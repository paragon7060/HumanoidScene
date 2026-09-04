#!/usr/bin/env node
/** Build photo-derived UV atlases and non-physical visual overrides for each box. */

const fs = require("node:fs");
const path = require("node:path");
const sharp = require("sharp");

const repoRoot = path.resolve(__dirname, "..");
const assetDir = path.join(repoRoot, "src", "kuavo_isaaclab_scene", "assets");
const textureDir = path.join(assetDir, "textures", "boxes");
const generatedDir = path.join(textureDir, "generated");

const ATLAS_SIZE = 2048;
const CELL_SIZE = 512;
const CELL_GUTTER = 12;

// Crop limits follow the photographed fold lines.  The left photographed panel
// is a D x H side wall and the right panel is a W x H front wall.
const boxSpecs = {
  small: {
    asset: "SmallBox",
    dimensions: { width: 0.266, depth: 0.185, height: 0.130, flap: 0.100 },
    image: "small.png",
    crop: { sideX: 0, sideWidth: 466, frontX: 493, frontWidth: 692, top: 279, bodyHeight: 325, imageHeight: 900 },
  },
  medium: {
    asset: "MediumBox",
    dimensions: { width: 0.320, depth: 0.220, height: 0.185, flap: 0.110 },
    image: "medium.png",
    crop: { sideX: 0, sideWidth: 513, frontX: 542, frontWidth: 751, top: 281, bodyHeight: 323, imageHeight: 896 },
  },
  large: {
    asset: "LargeBox",
    dimensions: { width: 0.380, depth: 0.260, height: 0.230, flap: 0.130 },
    image: "large.png",
    crop: { sideX: 0, sideWidth: 486, frontX: 500, frontWidth: 694, top: 285, bodyHeight: 352, imageHeight: 894 },
  },
  xlarge: {
    asset: "XLargeBox",
    dimensions: { width: 0.400, depth: 0.320, height: 0.285, flap: 0.155 },
    image: "xlarge.png",
    crop: { sideX: 0, sideWidth: 501, frontX: 512, frontWidth: 615, top: 254, bodyHeight: 389, imageHeight: 895 },
  },
};

const tileCells = {
  front: [0, 0],
  back: [1, 0],
  right: [2, 0],
  left: [3, 0],
  bottom: [0, 1],
  flapFront: [1, 1],
  flapBack: [2, 1],
  flapRight: [3, 1],
  flapLeft: [0, 2],
  innerFront: [1, 2],
  innerSide: [2, 2],
  innerBottom: [3, 2],
  innerFlapFront: [0, 3],
  innerFlapSide: [1, 3],
  innerEdge: [2, 3],
};

function crop(left, top, width, height) {
  return { left, top, width, height };
}

function photoSources(spec) {
  const c = spec.crop;
  const bottomTop = c.top + c.bodyHeight;
  return {
    front: { file: spec.image, region: crop(c.frontX, c.top, c.frontWidth, c.bodyHeight) },
    back: { file: spec.image, region: crop(c.frontX, c.top, c.frontWidth, c.bodyHeight) },
    right: { file: spec.image, region: crop(c.sideX, c.top, c.sideWidth, c.bodyHeight) },
    left: { file: spec.image, region: crop(c.sideX, c.top, c.sideWidth, c.bodyHeight) },
    bottom: { file: "bottom.png", region: crop(0, 0, 1448, 1086) },
    flapFront: { file: spec.image, region: crop(c.frontX, 0, c.frontWidth, c.top) },
    flapBack: {
      file: spec.image,
      region: crop(c.frontX, bottomTop, c.frontWidth, c.imageHeight - bottomTop),
      flip: true,
    },
    flapRight: { file: spec.image, region: crop(c.sideX, 0, c.sideWidth, c.top) },
    flapLeft: {
      file: spec.image,
      region: crop(c.sideX, bottomTop, c.sideWidth, c.imageHeight - bottomTop),
      flip: true,
    },
  };
}

function targetRatios({ width, depth, height, flap }) {
  return {
    front: width / height,
    back: width / height,
    right: depth / height,
    left: depth / height,
    bottom: width / depth,
    flapFront: width / flap,
    flapBack: width / flap,
    flapRight: depth / flap,
    flapLeft: depth / flap,
    innerFront: width / height,
    innerSide: depth / height,
    innerBottom: width / depth,
    innerFlapFront: width / flap,
    innerFlapSide: depth / flap,
    innerEdge: 1,
  };
}

function fittedSize(aspect) {
  const maximum = CELL_SIZE - 2 * CELL_GUTTER;
  if (aspect >= 1) {
    return { width: maximum, height: Math.max(1, Math.round(maximum / aspect)) };
  }
  return { width: Math.max(1, Math.round(maximum * aspect)), height: maximum };
}

async function transformedTile(source, width, height) {
  let pipeline = sharp(path.join(textureDir, source.file)).extract(source.region);
  if (source.flip) pipeline = pipeline.flip();
  if (source.flop) pipeline = pipeline.flop();
  return pipeline
    .resize(width, height, { fit: "fill", kernel: sharp.kernel.lanczos3 })
    .png()
    .toBuffer();
}

async function buildAtlas(boxType, spec) {
  const ratios = targetRatios(spec.dimensions);
  const sources = photoSources(spec);
  const insideSource = {
    file: "bottom.png",
    // Real, clean photographed cardboard above the taped bottom seam.
    region: crop(300, 16, 400, 400),
  };
  const composites = [];
  const uvRects = {};

  for (const [name, cell] of Object.entries(tileCells)) {
    const { width, height } = fittedSize(ratios[name]);
    const source = sources[name] || insideSource;
    const tile = await transformedTile(source, width, height);
    const padded = await sharp(tile)
      .extend({
        top: CELL_GUTTER,
        bottom: CELL_GUTTER,
        left: CELL_GUTTER,
        right: CELL_GUTTER,
        extendWith: "copy",
      })
      .png()
      .toBuffer();
    const outerLeft = cell[0] * CELL_SIZE + Math.floor((CELL_SIZE - width - 2 * CELL_GUTTER) / 2);
    const outerTop = cell[1] * CELL_SIZE + Math.floor((CELL_SIZE - height - 2 * CELL_GUTTER) / 2);
    const left = outerLeft + CELL_GUTTER;
    const top = outerTop + CELL_GUTTER;
    composites.push({ input: padded, left: outerLeft, top: outerTop });
    uvRects[name] = {
      u0: left / ATLAS_SIZE,
      u1: (left + width) / ATLAS_SIZE,
      v0: 1 - (top + height) / ATLAS_SIZE,
      v1: 1 - top / ATLAS_SIZE,
    };
  }

  const output = path.join(generatedDir, `${boxType}_atlas.png`);
  await sharp({
    create: {
      width: ATLAS_SIZE,
      height: ATLAS_SIZE,
      channels: 3,
      background: { r: 174, g: 135, b: 88 },
    },
  })
    .composite(composites)
    .png({ compressionLevel: 9, adaptiveFiltering: true })
    .toFile(output);
  return { output, uvRects };
}

function uvCorners(rect) {
  return {
    bl: [rect.u0, rect.v0],
    br: [rect.u1, rect.v0],
    tr: [rect.u1, rect.v1],
    tl: [rect.u0, rect.v1],
  };
}

function faceUv(rect, faceIndex) {
  const { bl, br, tr, tl } = uvCorners(rect);
  const layouts = {
    0: [bl, br, tr, tl],
    1: [bl, tl, tr, br],
    2: [bl, tl, tr, br],
    3: [br, bl, tl, tr],
    4: [br, tr, tl, bl],
    5: [bl, br, tr, tl],
  };
  return layouts[faceIndex];
}

function formatUv(values) {
  return values.map(([u, v]) => `(${u.toFixed(8)}, ${v.toFixed(8)})`).join(", ");
}

function bodyMeshOverride(name, tiles, uvRects) {
  const values = [];
  for (let face = 0; face < 6; face += 1) {
    values.push(...faceUv(uvRects[tiles[face]], face));
  }
  return `            over Mesh "${name}"
            {
                rel material:binding = </Root/Looks/BoxMaterial>
                texCoord2f[] primvars:st = [${formatUv(values)}] (
                    interpolation = "faceVarying"
                )
            }`;
}

function flapSurface(name, outerTile, innerTile, uvRects) {
  const outer = uvCorners(uvRects[outerTile]);
  const inner = uvCorners(uvRects[innerTile]);
  const definitions = {
    front: {
      points: "(-0.5, 0.501, -0.5), (-0.5, 0.501, 0.5), (0.5, 0.501, 0.5), (0.5, 0.501, -0.5), (-0.5, -0.501, -0.5), (0.5, -0.501, -0.5), (0.5, -0.501, 0.5), (-0.5, -0.501, 0.5)",
      uv: [outer.bl, outer.tl, outer.tr, outer.br, inner.br, inner.bl, inner.tl, inner.tr],
    },
    back: {
      points: "(-0.5, -0.501, -0.5), (0.5, -0.501, -0.5), (0.5, -0.501, 0.5), (-0.5, -0.501, 0.5), (-0.5, 0.501, -0.5), (-0.5, 0.501, 0.5), (0.5, 0.501, 0.5), (0.5, 0.501, -0.5)",
      uv: [outer.br, outer.bl, outer.tl, outer.tr, inner.bl, inner.tl, inner.tr, inner.br],
    },
    right: {
      points: "(0.501, -0.5, -0.5), (0.501, 0.5, -0.5), (0.501, 0.5, 0.5), (0.501, -0.5, 0.5), (-0.501, -0.5, -0.5), (-0.501, -0.5, 0.5), (-0.501, 0.5, 0.5), (-0.501, 0.5, -0.5)",
      uv: [outer.bl, outer.br, outer.tr, outer.tl, inner.br, inner.tr, inner.tl, inner.bl],
    },
    left: {
      points: "(-0.501, -0.5, -0.5), (-0.501, -0.5, 0.5), (-0.501, 0.5, 0.5), (-0.501, 0.5, -0.5), (0.501, -0.5, -0.5), (0.501, 0.5, -0.5), (0.501, 0.5, 0.5), (0.501, -0.5, 0.5)",
      uv: [outer.br, outer.tr, outer.tl, outer.bl, inner.bl, inner.br, inner.tr, inner.tl],
    },
  };
  const d = definitions[name];
  return `        over Cube "flap_${name}"
        {
            def Mesh "texture_surfaces"
            {
                uniform bool doubleSided = 0
                int[] faceVertexCounts = [4, 4]
                int[] faceVertexIndices = [0, 1, 2, 3, 4, 5, 6, 7]
                point3f[] points = [${d.points}]
                rel material:binding = </Root/Looks/BoxMaterial>
                texCoord2f[] primvars:st = [${formatUv(d.uv)}] (
                    interpolation = "faceVarying"
                )
            }
        }`;
}

function wrapperText(boxType, spec, uvRects) {
  const asset = spec.asset;
  const edge = "innerEdge";
  const bodyMeshes = [
    bodyMeshOverride("bottom", ["innerBottom", "bottom", edge, edge, edge, edge], uvRects),
    bodyMeshOverride("wall_front", [edge, edge, "front", "innerFront", edge, edge], uvRects),
    bodyMeshOverride("wall_back", [edge, edge, "innerFront", "back", edge, edge], uvRects),
    bodyMeshOverride("wall_left", [edge, edge, edge, edge, "innerSide", "right"], uvRects),
    bodyMeshOverride("wall_right", [edge, edge, edge, edge, "left", "innerSide"], uvRects),
  ].join("\n\n");
  const flapMeshes = [
    flapSurface("front", "flapFront", "innerFlapFront", uvRects),
    flapSurface("back", "flapBack", "innerFlapFront", uvRects),
    flapSurface("right", "flapRight", "innerFlapSide", uvRects),
    flapSurface("left", "flapLeft", "innerFlapSide", uvRects),
  ].join("\n\n");

  return `#usda 1.0
(
    defaultPrim = "Root"
    upAxis = "Z"
    customLayerData = {
        string sourceGeometry = "${asset}_physical.usda"
        string sourcePhoto = "textures/boxes/${spec.image}"
        string textureAtlas = "textures/boxes/generated/${boxType}_atlas.png"
    }
)

def Xform "Root" (
    prepend references = @./${asset}_physical.usda@</Root>
)
{
    def Scope "Looks"
    {
        def Material "BoxMaterial"
        {
            token outputs:surface.connect = </Root/Looks/BoxMaterial/PreviewSurface.outputs:surface>

            def Shader "PreviewSurface"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor.connect = </Root/Looks/BoxMaterial/Albedo.outputs:rgb>
                float inputs:metallic = 0
                float inputs:roughness = 0.82
                token outputs:surface
            }

            def Shader "Albedo"
            {
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @./textures/boxes/generated/${boxType}_atlas.png@
                token inputs:sourceColorSpace = "sRGB"
                float2 inputs:st.connect = </Root/Looks/BoxMaterial/PrimvarReader.outputs:result>
                float3 outputs:rgb
            }

            def Shader "PrimvarReader"
            {
                uniform token info:id = "UsdPrimvarReader_float2"
                token inputs:varname = "st"
                float2 outputs:result
            }
        }
    }

    over Xform "${asset}"
    {
        over Xform "Body"
        {
${bodyMeshes}
        }

${flapMeshes}
    }
}
`;
}

async function main() {
  fs.mkdirSync(generatedDir, { recursive: true });
  for (const [boxType, spec] of Object.entries(boxSpecs)) {
    const { output, uvRects } = await buildAtlas(boxType, spec);
    const wrapper = path.join(assetDir, `${spec.asset}_atlas.usda`);
    fs.writeFileSync(wrapper, wrapperText(boxType, spec, uvRects), "utf8");
    console.log(`wrote ${path.relative(repoRoot, output)}`);
    console.log(`wrote ${path.relative(repoRoot, wrapper)}`);
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
