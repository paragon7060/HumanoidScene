type PoseArray = [number, number, number, number, number, number, number];
const PROTOCOL_VERSION = 2;
const FRAME_HEADER_BYTES = 32;

type EyeViewPacket = {
  eye: 'left' | 'right';
  pose: PoseArray;
  projection_matrix: number[];
};

type ControllerPacket = {
  grip: PoseArray;
  thumbstick: [number, number]; // WebXR: +X right, +Y down
  trigger: number;
};

type TrackingPacket = {
  type: 'tracking';
  protocol_version: number;
  sequence: number;
  timestamp_ms: number;
  head: PoseArray | null;
  left_hand: Record<string, PoseArray> | null;
  right_hand: Record<string, PoseArray> | null;
  left_controller: ControllerPacket | null;
  right_controller: ControllerPacket | null;
  views: EyeViewPacket[];
};

function transformToArray(transform: XRRigidTransform): PoseArray {
  const { position, orientation } = transform;
  return [
    position.x,
    position.y,
    position.z,
    orientation.w,
    orientation.x,
    orientation.y,
    orientation.z,
  ];
}

function poseFromSpace(frame: XRFrame, space: XRSpace, referenceSpace: XRReferenceSpace): PoseArray | null {
  const pose = frame.getPose(space, referenceSpace);
  return pose ? transformToArray(pose.transform) : null;
}

function controllerHand(wrist: PoseArray): Record<string, PoseArray> {
  const [x, y, z, qw, qx, qy, qz] = wrist;
  return {
    wrist,
    thumb_tip: [x - 0.025, y + 0.015, z - 0.035, qw, qx, qy, qz],
    index_tip: [x + 0.025, y + 0.015, z - 0.035, qw, qx, qy, qz],
  };
}

export function trackedController(
  frame: XRFrame,
  inputSource: XRInputSource,
  referenceSpace: XRReferenceSpace
): ControllerPacket | null {
  // Hand tracking can also expose a gamepad; do not interpret it as a stick.
  const gamepad = inputSource.gamepad;
  if (inputSource.hand || !inputSource.gripSpace || !gamepad || gamepad.mapping !== 'xr-standard') return null;
  const grip = poseFromSpace(frame, inputSource.gripSpace, referenceSpace);
  if (!grip) return null;
  const axis = (index: number): number => {
    const value = gamepad.axes[index] ?? 0;
    return Number.isFinite(value) ? Math.max(-1, Math.min(1, value)) : 0;
  };
  const trigger = gamepad.buttons[0]?.value ?? 0;
  return {
    grip,
    // xr-standard axes 0/1 are the touchpad, 2/3 are the thumbstick.
    thumbstick: [axis(2), axis(3)],
    trigger: Number.isFinite(trigger) ? Math.max(0, Math.min(1, trigger)) : 0,
  };
}

function trackedHand(
  frame: XRFrame,
  inputSource: XRInputSource,
  referenceSpace: XRReferenceSpace
): Record<string, PoseArray> | null {
  if (inputSource.hand) {
    const joints: Array<[XRHandJoint, string]> = [
      ['wrist', 'wrist'],
      ['thumb-tip', 'thumb_tip'],
      ['index-finger-tip', 'index_tip'],
    ];
    const output: Record<string, PoseArray> = {};
    for (const [jointName, outputName] of joints) {
      const jointSpace = inputSource.hand.get(jointName);
      if (jointSpace) {
        const pose = poseFromSpace(frame, jointSpace, referenceSpace);
        if (pose) output[outputName] = pose;
      }
    }
    return output.wrist ? output : null;
  }
  if (inputSource.gripSpace) {
    const wrist = poseFromSpace(frame, inputSource.gripSpace, referenceSpace);
    return wrist ? controllerHand(wrist) : null;
  }
  return null;
}

export class LocalIsaacLabBridge {
  private socket: WebSocket | null = null;
  private latestBitmap: ImageBitmap | null = null;
  private decodingImage = false;
  private program: WebGLProgram;
  private texture: WebGLTexture;
  private vertexBuffer: WebGLBuffer;
  private positionLocation: number;
  private eyeIndexLocation: WebGLUniformLocation;
  private trackingSequence = 0;
  private metricsStartedAt = performance.now();
  private receivedFrames = 0;
  private renderedFrames = 0;
  private droppedFrames = 0;
  private poseLatencyTotalMs = 0;
  private poseLatencySamples = 0;
  private decodeTotalMs = 0;
  private decodeSamples = 0;
  private latestServerFps = 0;

  constructor(
    private gl: WebGL2RenderingContext,
    private referenceSpace: XRReferenceSpace
  ) {
    this.program = this.createProgram();
    const texture = gl.createTexture();
    const vertexBuffer = gl.createBuffer();
    if (!texture || !vertexBuffer) throw new Error('Failed to allocate local IsaacLab renderer');
    this.texture = texture;
    this.vertexBuffer = vertexBuffer;
    this.positionLocation = gl.getAttribLocation(this.program, 'a_position');
    const eyeIndexLocation = gl.getUniformLocation(this.program, 'u_eye_index');
    if (!eyeIndexLocation) throw new Error('Failed to find stereo eye shader uniform');
    this.eyeIndexLocation = eyeIndexLocation;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
      gl.STATIC_DRAW
    );
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA,
      1,
      1,
      0,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      new Uint8Array([8, 12, 18, 255])
    );
  }

  async connect(url: string): Promise<void> {
    await new Promise<void>((resolve, reject) => {
      const socket = new WebSocket(url);
      socket.binaryType = 'arraybuffer';
      socket.addEventListener(
        'open',
        () => {
          socket.send(JSON.stringify({ type: 'client_hello', protocol_version: PROTOCOL_VERSION }));
          resolve();
        },
        { once: true }
      );
      socket.addEventListener('error', () => reject(new Error(`Cannot connect to ${url}`)), {
        once: true,
      });
      socket.addEventListener('message', event => {
        if (event.data instanceof ArrayBuffer) void this.acceptCameraFrame(event.data);
        else if (event.data instanceof Blob) void event.data.arrayBuffer().then(data => this.acceptCameraFrame(data));
        else if (typeof event.data === 'string') this.acceptControlMessage(event.data);
      });
      this.socket = socket;
    });
  }

  onXRFrame(timestamp: DOMHighResTimeStamp, frame: XRFrame, baseLayer: XRWebGLLayer): void {
    this.sendTracking(timestamp, frame);
    this.render(frame, baseLayer);
    this.reportMetrics(timestamp);
  }

  close(): void {
    this.socket?.close();
    this.socket = null;
    this.latestBitmap?.close();
    this.latestBitmap = null;
    this.gl.deleteTexture(this.texture);
    this.gl.deleteBuffer(this.vertexBuffer);
    this.gl.deleteProgram(this.program);
  }

  private sendTracking(timestamp: DOMHighResTimeStamp, frame: XRFrame): void {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) return;
    const viewerPose = frame.getViewerPose(this.referenceSpace);
    const packet: TrackingPacket = {
      type: 'tracking',
      protocol_version: PROTOCOL_VERSION,
      sequence: ++this.trackingSequence,
      timestamp_ms: timestamp,
      head: viewerPose ? transformToArray(viewerPose.transform) : null,
      left_hand: null,
      right_hand: null,
      left_controller: null,
      right_controller: null,
      views: viewerPose
        ? viewerPose.views
            .filter(view => view.eye === 'left' || view.eye === 'right')
            .map(view => ({
              eye: view.eye as 'left' | 'right',
              pose: transformToArray(view.transform),
              projection_matrix: Array.from(view.projectionMatrix),
            }))
        : [],
    };
    for (const inputSource of frame.session.inputSources) {
      if (inputSource.handedness !== 'left' && inputSource.handedness !== 'right') continue;
      const hand = trackedHand(frame, inputSource, this.referenceSpace);
      const controller = trackedController(frame, inputSource, this.referenceSpace);
      if (inputSource.handedness === 'left') {
        packet.left_hand = hand;
        packet.left_controller = controller;
      } else {
        packet.right_hand = hand;
        packet.right_controller = controller;
      }
    }
    this.socket.send(JSON.stringify(packet));
  }

  private async acceptCameraFrame(data: ArrayBuffer): Promise<void> {
    this.receivedFrames += 1;
    if (data.byteLength < FRAME_HEADER_BYTES) {
      this.failProtocol('Frame is shorter than the protocol header.');
      return;
    }
    const header = new DataView(data, 0, FRAME_HEADER_BYTES);
    const magic = String.fromCharCode(
      header.getUint8(0), header.getUint8(1), header.getUint8(2), header.getUint8(3)
    );
    const version = header.getUint8(4);
    const headerBytes = header.getUint16(6, true);
    if (magic !== 'KVR2' || version !== PROTOCOL_VERSION || headerBytes !== FRAME_HEADER_BYTES) {
      this.failProtocol(`Unsupported frame protocol magic=${magic} version=${version}.`);
      return;
    }
    const poseTimestampMs = header.getFloat64(16, true);
    this.latestServerFps = header.getFloat32(28, true);
    if (Number.isFinite(poseTimestampMs) && poseTimestampMs > 0) {
      this.poseLatencyTotalMs += Math.max(0, performance.now() - poseTimestampMs);
      this.poseLatencySamples += 1;
    }
    if (this.decodingImage) {
      this.droppedFrames += 1;
      return;
    }
    this.decodingImage = true;
    const decodeStartedAt = performance.now();
    try {
      const bitmap = await createImageBitmap(
        new Blob([data.slice(headerBytes)], { type: 'image/jpeg' })
      );
      this.latestBitmap?.close();
      this.latestBitmap = bitmap;
      this.decodeTotalMs += performance.now() - decodeStartedAt;
      this.decodeSamples += 1;
    } finally {
      this.decodingImage = false;
    }
  }

  private acceptControlMessage(message: string): void {
    try {
      const payload = JSON.parse(message) as Record<string, unknown>;
      if (payload.type === 'protocol_error') {
        this.failProtocol(
          `Server expects protocol ${payload.expected}; browser is ${PROTOCOL_VERSION}. Re-run setup_quest_browser.sh --patch-only.`
        );
      } else if (payload.type === 'server_hello' && payload.protocol_version !== PROTOCOL_VERSION) {
        this.failProtocol(`Server hello used protocol ${payload.protocol_version}.`);
      }
    } catch (error) {
      console.warn('Ignoring malformed Kuavo bridge control message', error);
    }
  }

  private failProtocol(message: string): void {
    console.error(`[KUAVO XR PROTOCOL] ${message}`);
    this.socket?.close(1002, 'Kuavo XR protocol mismatch');
  }

  private reportMetrics(now: DOMHighResTimeStamp): void {
    const elapsedMs = now - this.metricsStartedAt;
    if (elapsedMs < 1000) return;
    const elapsedSeconds = elapsedMs / 1000;
    const receivedFps = this.receivedFrames / elapsedSeconds;
    const renderedFps = this.renderedFrames / elapsedSeconds;
    const poseToFrameMs = this.poseLatencySamples
      ? this.poseLatencyTotalMs / this.poseLatencySamples
      : Number.NaN;
    const decodeMs = this.decodeSamples ? this.decodeTotalMs / this.decodeSamples : Number.NaN;
    const metrics = {
      type: 'client_metrics',
      protocol_version: PROTOCOL_VERSION,
      received_fps: receivedFps,
      rendered_fps: renderedFps,
      pose_to_frame_ms: poseToFrameMs,
      decode_ms: decodeMs,
      dropped_frames: this.droppedFrames,
    };
    if (this.socket?.readyState === WebSocket.OPEN) this.socket.send(JSON.stringify(metrics));
    console.info(
      `[KUAVO XR] receive=${receivedFps.toFixed(1)} fps render=${renderedFps.toFixed(1)} fps ` +
      `pose-to-frame=${poseToFrameMs.toFixed(1)} ms decode=${decodeMs.toFixed(1)} ms ` +
      `server=${this.latestServerFps.toFixed(1)} fps dropped=${this.droppedFrames}`
    );
    this.metricsStartedAt = now;
    this.receivedFrames = 0;
    this.renderedFrames = 0;
    this.droppedFrames = 0;
    this.poseLatencyTotalMs = 0;
    this.poseLatencySamples = 0;
    this.decodeTotalMs = 0;
    this.decodeSamples = 0;
  }

  private render(frame: XRFrame, baseLayer: XRWebGLLayer): void {
    const viewerPose = frame.getViewerPose(this.referenceSpace);
    if (!viewerPose) return;
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, baseLayer.framebuffer);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.CULL_FACE);
    gl.useProgram(this.program);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
    gl.enableVertexAttribArray(this.positionLocation);
    gl.vertexAttribPointer(this.positionLocation, 2, gl.FLOAT, false, 0, 0);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    if (this.latestBitmap) {
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, this.latestBitmap);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
      this.latestBitmap.close();
      this.latestBitmap = null;
      this.renderedFrames += 1;
    }
    for (const view of viewerPose.views) {
      const viewport = baseLayer.getViewport(view);
      if (!viewport) continue;
      gl.viewport(viewport.x, viewport.y, viewport.width, viewport.height);
      gl.uniform1f(this.eyeIndexLocation, view.eye === 'right' ? 1.0 : 0.0);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
  }

  private createProgram(): WebGLProgram {
    const vertexSource = `#version 300 es
      in vec2 a_position;
      out vec2 v_uv;
      void main() {
        v_uv = a_position * 0.5 + 0.5;
        gl_Position = vec4(a_position, 0.0, 1.0);
      }`;
    const fragmentSource = `#version 300 es
      precision mediump float;
      in vec2 v_uv;
      uniform sampler2D u_texture;
      uniform float u_eye_index;
      out vec4 out_color;
      void main() {
        vec2 atlas_uv = vec2(v_uv.x * 0.5 + u_eye_index * 0.5, v_uv.y);
        out_color = texture(u_texture, atlas_uv);
      }`;
    const compile = (type: number, source: string): WebGLShader => {
      const shader = this.gl.createShader(type);
      if (!shader) throw new Error('Failed to allocate WebGL shader');
      this.gl.shaderSource(shader, source);
      this.gl.compileShader(shader);
      if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
        throw new Error(this.gl.getShaderInfoLog(shader) || 'WebGL shader compilation failed');
      }
      return shader;
    };
    const vertex = compile(this.gl.VERTEX_SHADER, vertexSource);
    const fragment = compile(this.gl.FRAGMENT_SHADER, fragmentSource);
    const program = this.gl.createProgram();
    if (!program) throw new Error('Failed to allocate WebGL program');
    this.gl.attachShader(program, vertex);
    this.gl.attachShader(program, fragment);
    this.gl.linkProgram(program);
    this.gl.deleteShader(vertex);
    this.gl.deleteShader(fragment);
    if (!this.gl.getProgramParameter(program, this.gl.LINK_STATUS)) {
      throw new Error(this.gl.getProgramInfoLog(program) || 'WebGL program link failed');
    }
    return program;
  }
}
