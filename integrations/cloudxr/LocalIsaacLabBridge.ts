type PoseArray = [number, number, number, number, number, number, number];

type TrackingPacket = {
  type: 'tracking';
  timestamp_ms: number;
  head: PoseArray | null;
  left_hand: Record<string, PoseArray> | null;
  right_hand: Record<string, PoseArray> | null;
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
      socket.addEventListener('open', () => resolve(), { once: true });
      socket.addEventListener('error', () => reject(new Error(`Cannot connect to ${url}`)), {
        once: true,
      });
      socket.addEventListener('message', event => {
        if (event.data instanceof ArrayBuffer) void this.acceptCameraFrame(event.data);
        else if (event.data instanceof Blob) void event.data.arrayBuffer().then(data => this.acceptCameraFrame(data));
      });
      this.socket = socket;
    });
  }

  onXRFrame(timestamp: DOMHighResTimeStamp, frame: XRFrame, baseLayer: XRWebGLLayer): void {
    this.sendTracking(timestamp, frame);
    this.render(frame, baseLayer);
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
      timestamp_ms: timestamp,
      head: viewerPose ? transformToArray(viewerPose.transform) : null,
      left_hand: null,
      right_hand: null,
    };
    for (const inputSource of frame.session.inputSources) {
      if (inputSource.handedness !== 'left' && inputSource.handedness !== 'right') continue;
      const hand = trackedHand(frame, inputSource, this.referenceSpace);
      if (inputSource.handedness === 'left') packet.left_hand = hand;
      else packet.right_hand = hand;
    }
    this.socket.send(JSON.stringify(packet));
  }

  private async acceptCameraFrame(data: ArrayBuffer): Promise<void> {
    if (this.decodingImage) return;
    this.decodingImage = true;
    try {
      const bitmap = await createImageBitmap(new Blob([data], { type: 'image/jpeg' }));
      this.latestBitmap?.close();
      this.latestBitmap = bitmap;
    } finally {
      this.decodingImage = false;
    }
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
    }
    for (const view of viewerPose.views) {
      const viewport = baseLayer.getViewport(view);
      if (!viewport) continue;
      gl.viewport(viewport.x, viewport.y, viewport.width, viewport.height);
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
      out vec4 out_color;
      void main() { out_color = texture(u_texture, v_uv); }`;
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
