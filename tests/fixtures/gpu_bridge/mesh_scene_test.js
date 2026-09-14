/**
 * tests/fixtures/gpu_bridge/mesh_scene_test.js
 *
 * Real browser proof for dynamic Three.js Mesh rendering through
 * Gray's production mesh_adapter -> compiled Wasm f3d_build_mesh_packet /
 * f3d_build_canvas_mesh_packet -> WebGpuBridgeHost.
 *
 * Suites:
 * 1. testMeshScene (Offscreen pass readback):
 *    - Exercises Gray's actual production adapter (prepareMeshPacket, canAdmitMesh).
 *    - No mock adapter or fallback extraction; missing production adapter MUST fail immediately.
 *    - Independent direct WebGPU oracle reference constructed directly from real Three.js
 *      attributes (getX/getY/getZ), matrix multiplication, projection, and material.
 *    - Does NOT use candidate-extracted data as oracle input.
 *    - Checkpoints: baseline match, translation mutation match, color mutation match,
 *      stale cached packet mismatch negative, and admission rejection negatives.
 *
 * 2. testVisibleCanvasMeshScene (Visible canvas pass):
 *    - Actual canvas output vs independent direct WebGPU rendering in preferredCanvasFormat.
 *    - Fresh currentTexture across submissions.
 *    - Transform and color mutation observed on canvas texture.
 *    - Negative control: stale initial packet fails to match mutated reference.
 */

import { WebGpuBridgeHost } from "./bridge_runtime.js";

/**
 * Genuinely independent WebGPU oracle reference.
 * Issues raw WebGPU API calls directly; does NOT use F3D Wasm or bridge decoders.
 */
export async function directMeshReference(device, options) {
  const {
    positions,
    indices,
    modelView,
    projection,
    webglDepth = true,
    color,
    width = 64,
    height = 64,
    format = "rgba8unorm",
    hasDepth = false,
    depthFormat = "depth24plus",
    depthWriteEnabled = true,
    depthCompare = "less",
    depthClearValue = 1.0,
    outputSrgb = false,
    draws = null,
  } = options;

  const bytesPerRow = Math.ceil((width * 4) / 256) * 256;
  const target = device.createTexture({
    size: [width, height, 1],
    format,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  const depthTexture = hasDepth
    ? device.createTexture({
        size: [width, height, 1],
        format: depthFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      })
    : null;
  const readback = device.createBuffer({
    size: bytesPerRow * height,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const remapDepthWgsl = webglDepth
    ? "clip.z = (clip.z + clip.w) * 0.5;"
    : "";

  // Linear-sRGB to sRGB OETF transfer function per Three.js r186 ColorSpaceFunctions.js:38-48
  const srgbFunctionWgsl = outputSrgb
    ? `
      fn srgb_transfer_oetf(color: vec3<f32>) -> vec3<f32> {
        let clamped = max(color, vec3<f32>(0.0));
        let a = pow(clamped, vec3<f32>(0.41666)) * 1.055 - vec3<f32>(0.055);
        let b = color * 12.92;
        return select(a, b, color <= vec3<f32>(0.0031308));
      }
    `
    : "";

  const fragmentReturnWgsl = outputSrgb
    ? `
        let srgb_rgb = srgb_transfer_oetf(uniforms.color.rgb);
        return vec4<f32>(srgb_rgb, uniforms.color.a);
    `
    : `
        return uniforms.color;
    `;

  const shaderModule = device.createShaderModule({
    code: `
      struct MeshUniforms {
        model_view: mat4x4<f32>,
        projection: mat4x4<f32>,
        color: vec4<f32>,
      };

      @group(0) @binding(0)
      var<uniform> uniforms: MeshUniforms;

      struct VertexInput {
        @location(0) position: vec3<f32>,
        @location(1) uv: vec2<f32>,
      };

      struct VertexOutput {
        @builtin(position) position: vec4<f32>,
      };

      @vertex
      fn vs_main(in: VertexInput) -> VertexOutput {
        var out: VertexOutput;
        var mv_pos = uniforms.model_view * vec4<f32>(in.position, 1.0);
        var clip = uniforms.projection * mv_pos;
        ${remapDepthWgsl}
        out.position = clip;
        return out;
      }

      ${srgbFunctionWgsl}

      @fragment
      fn fs_main() -> @location(0) vec4<f32> {
        ${fragmentReturnWgsl}
      }
    `,
  });

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: "uniform" },
      },
    ],
  });

  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
  });

  const pipelineCache = new Map();
  const getPipeline = (
    cullMode = "none",
    frontFace = "ccw",
    itemDepthWriteEnabled = depthWriteEnabled,
    itemDepthCompare = depthCompare,
    itemWriteMask = 0xF
  ) => {
    const key = `${cullMode}:${frontFace}:${itemDepthWriteEnabled}:${itemDepthCompare}:${itemWriteMask}`;
    if (!pipelineCache.has(key)) {
      const p = device.createRenderPipeline({
        layout: pipelineLayout,
        vertex: {
          module: shaderModule,
          entryPoint: "vs_main",
          buffers: [
            {
              arrayStride: 20,
              attributes: [
                { shaderLocation: 0, offset: 0, format: "float32x3" },
                { shaderLocation: 1, offset: 12, format: "float32x2" },
              ],
            },
          ],
        },
        fragment: {
          module: shaderModule,
          entryPoint: "fs_main",
          targets: [{ format, writeMask: itemWriteMask }],
        },
        primitive: {
          topology: "triangle-list",
          cullMode,
          frontFace,
        },
        depthStencil: hasDepth
          ? {
              format: depthFormat,
              depthWriteEnabled: itemDepthWriteEnabled,
              depthCompare: itemDepthCompare,
            }
          : undefined,
      });
      pipelineCache.set(key, p);
    }
    return pipelineCache.get(key);
  };

  const drawList = (draws && draws.length > 0)
    ? draws
    : [{ positions, indices, modelView, color, projection }];

  const preparedDraws = [];
  for (const item of drawList) {
    const itemPositions = item.positions;
    const itemIndices = item.indices;
    const itemModelView = item.modelView;
    const itemColor = item.color;
    const itemProjection = item.projection || projection;

    const vCount = itemPositions.length / 3;
    const vData = new Float32Array(vCount * 5);
    for (let i = 0; i < vCount; i++) {
      vData[i * 5 + 0] = itemPositions[i * 3 + 0];
      vData[i * 5 + 1] = itemPositions[i * 3 + 1];
      vData[i * 5 + 2] = itemPositions[i * 3 + 2];
      vData[i * 5 + 3] = 0.0;
      vData[i * 5 + 4] = 0.0;
    }

    const vBuffer = device.createBuffer({
      size: Math.max(vData.byteLength, 16),
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(vBuffer, 0, vData);

    let iBuffer = null;
    const itemIsIndexed = itemIndices && itemIndices.length > 0;
    if (itemIsIndexed) {
      iBuffer = device.createBuffer({
        size: Math.max(itemIndices.byteLength, 16),
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(iBuffer, 0, itemIndices);
    }

    const uData = new Float32Array(64); // 256 bytes
    uData.set(new Float32Array(itemModelView), 0);
    uData.set(new Float32Array(itemProjection), 16);
    uData.set(new Float32Array(itemColor), 32);

    const uBuffer = device.createBuffer({
      size: 256,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(uBuffer, 0, uData);

    const bGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [{ binding: 0, resource: { buffer: uBuffer } }],
    });

    const itemCullMode = item.cullMode || options.cullMode || "none";
    const itemFrontFace = item.frontFace || options.frontFace || "ccw";
    const itemDepthWriteEnabled = (item.depthWriteEnabled !== undefined)
      ? item.depthWriteEnabled
      : depthWriteEnabled;
    const itemDepthCompare = item.depthCompare || depthCompare;
    const itemWriteMask = (item.writeMask !== undefined)
      ? item.writeMask
      : (item.colorWrite === false ? 0x0 : (options.writeMask !== undefined ? options.writeMask : 0xF));
    preparedDraws.push({
      vertexBuffer: vBuffer,
      indexBuffer: iBuffer,
      uniformBuffer: uBuffer,
      bindGroup: bGroup,
      vertexCount: vCount,
      isIndexed: itemIsIndexed,
      indicesLength: itemIsIndexed ? itemIndices.length : 0,
      cullMode: itemCullMode,
      frontFace: itemFrontFace,
      depthWriteEnabled: itemDepthWriteEnabled,
      depthCompare: itemDepthCompare,
      writeMask: itemWriteMask,
    });
  }

  device.pushErrorScope("validation");
  let scopeOpen = true;
  try {
    const encoder = device.createCommandEncoder();
    const passDesc = {
      colorAttachments: [
        {
          view: target.createView(),
          loadOp: "clear",
          storeOp: "store",
          clearValue: [0, 0, 0, 1],
        },
      ],
    };
    if (hasDepth) {
      passDesc.depthStencilAttachment = {
        view: depthTexture.createView(),
        depthClearValue,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      };
    }
    const pass = encoder.beginRenderPass(passDesc);
    let currentPipelineKey = null;
    for (const draw of preparedDraws) {
      const pipeKey = `${draw.cullMode}:${draw.frontFace}:${draw.depthWriteEnabled}:${draw.depthCompare}:${draw.writeMask}`;
      if (pipeKey !== currentPipelineKey) {
        pass.setPipeline(getPipeline(draw.cullMode, draw.frontFace, draw.depthWriteEnabled, draw.depthCompare, draw.writeMask));
        currentPipelineKey = pipeKey;
      }
      pass.setBindGroup(0, draw.bindGroup);
      pass.setVertexBuffer(0, draw.vertexBuffer);
      if (draw.isIndexed) {
        pass.setIndexBuffer(draw.indexBuffer, "uint32");
        pass.drawIndexed(draw.indicesLength, 1, 0, 0, 0);
      } else {
        pass.draw(draw.vertexCount, 1, 0, 0);
      }
    }
    pass.end();

    encoder.copyTextureToBuffer(
      { texture: target },
      { buffer: readback, bytesPerRow },
      [width, height, 1]
    );

    device.queue.submit([encoder.finish()]);

    const validation = device.popErrorScope();
    scopeOpen = false;
    const error = await validation;
    if (error) {
      throw new Error(`Direct mesh reference validation error: ${error.message}`);
    }

    await readback.mapAsync(GPUMapMode.READ);
    const pixels = new Uint8Array(readback.getMappedRange().slice(0));
    readback.unmap();
    return pixels;
  } finally {
    if (scopeOpen) await device.popErrorScope();
    target.destroy();
    readback.destroy();
    if (depthTexture) depthTexture.destroy();
    for (const draw of preparedDraws) {
      draw.vertexBuffer.destroy();
      if (draw.indexBuffer) draw.indexBuffer.destroy();
      draw.uniformBuffer.destroy();
    }
  }
}

export function differs(a, b) {
  if (a.length !== b.length) return true;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return true;
  }
  return false;
}

export function nextFrame() {
  return new Promise(resolve => {
    if (typeof requestAnimationFrame === "function") {
      requestAnimationFrame(() => resolve());
    } else {
      setTimeout(resolve, 16);
    }
  });
}

/**
 * Builds reference inputs INDEPENDENTLY from actual Three.js scene objects.
 * Does NOT consume candidate adapter extraction.
 * Reads position attribute directly via getX/getY/getZ.
 * Computes model-view matrix directly via camera.matrixWorldInverse.clone().multiply(mesh.matrixWorld).
 * Reads projection matrix and material directly from source objects.
 */
function buildIndependentReferenceInput(mesh, camera, width = 64, height = 64, options = {}) {
  const posAttr = mesh.geometry.attributes.position;
  const vertexCount = posAttr.count;
  const positions = new Float32Array(vertexCount * 3);
  for (let i = 0; i < vertexCount; i++) {
    positions[i * 3 + 0] = posAttr.getX(i);
    positions[i * 3 + 1] = posAttr.getY(i);
    positions[i * 3 + 2] = posAttr.getZ(i);
  }

  let indices = new Uint32Array(0);
  if (mesh.geometry.index) {
    const rawIndices = mesh.geometry.index.array;
    indices = new Uint32Array(rawIndices.length);
    for (let i = 0; i < rawIndices.length; i++) {
      indices[i] = rawIndices[i];
    }
  }

  // Independent matrix multiplication: camera.matrixWorldInverse.clone().multiply(mesh.matrixWorld)
  const mvi = camera.matrixWorldInverse.clone();
  const mw = mesh.matrixWorld.clone();
  const mv = mvi.multiply(mw);
  const modelView = new Float64Array(16);
  for (let i = 0; i < 16; i++) {
    modelView[i] = mv.elements[i];
  }

  // Raw projection matrix elements directly from camera
  const projection = new Float64Array(16);
  for (let i = 0; i < 16; i++) {
    projection[i] = camera.projectionMatrix.elements[i];
  }

  // Coordinate system: true if WebGL clip z needs remapping
  const webglDepth = camera.coordinateSystem !== 2001; // 2001 = THREE.WebGPUCoordinateSystem

  // Raw material color directly from material
  const mat = mesh.material;
  const color = new Float32Array([
    mat.color.r,
    mat.color.g,
    mat.color.b,
    mat.opacity !== undefined ? mat.opacity : 1.0,
  ]);

  const side = (mat && typeof mat.side === "number") ? mat.side : 0;
  const isReflected = (typeof mesh.matrixWorld?.determinantAffine === "function")
    ? (mesh.matrixWorld.determinantAffine() < 0)
    : false;
  const flipSided = (side === 1) ? !isReflected : isReflected;
  const frontFace = flipSided ? "cw" : "ccw";
  const cullMode = (side === 2) ? "none" : "back";

  // Material Depth Settings per pinned Three.js r186 semantics
  const depthTest = (mat && mat.depthTest !== false);
  const rawDepthWrite = (mat && mat.depthWrite !== false);
  const rawDepthFunc = (mat && mat.depthFunc !== undefined) ? mat.depthFunc : 3; // 3 = LessEqualDepth

  let depthWriteEnabled = rawDepthWrite;
  if (!depthTest && rawDepthWrite) {
    const backend = options?.sourceBackend?.toLowerCase();
    if (backend === "webgl") {
      depthWriteEnabled = false;
    } else if (backend === "webgpu") {
      depthWriteEnabled = true;
    } else {
      throw new Error("AMBIGUOUS_DEPTH_PAIR: Material with depthTest=false and depthWrite=true is ambiguous across backends; provide options.sourceBackend (\"webgl\" | \"webgpu\")");
    }
  }

  const THREE_DEPTH_FUNC_TO_COMPARE_STRING = {
    0: "never",
    1: "always",
    2: "less",
    3: "less-equal",
    4: "equal",
    5: "greater-equal",
    6: "greater",
    7: "not-equal",
  };

  const depthCompare = depthTest
    ? (THREE_DEPTH_FUNC_TO_COMPARE_STRING[rawDepthFunc] || "less")
    : "always";

  const colorWrite = (mat && mat.colorWrite !== false);
  const writeMask = colorWrite ? 0xF : 0x0;

  return {
    positions,
    indices,
    modelView,
    projection,
    webglDepth,
    color,
    colorWrite,
    writeMask,
    width,
    height,
    cullMode,
    frontFace,
    depthTest,
    depthWriteEnabled,
    depthCompare,
  };
}

/**
 * Builds reference batch inputs INDEPENDENTLY from an ordered array of Three.js Mesh instances.
 * Extracts per-mesh positions, indices, modelView, and colors directly from source objects.
 */
function buildIndependentBatchReferenceInput(meshes, camera, width = 64, height = 64, extraOptions = {}) {
  const draws = meshes.map((m) => {
    const single = buildIndependentReferenceInput(m, camera, width, height, extraOptions);
    return {
      positions: single.positions,
      indices: single.indices,
      modelView: single.modelView,
      color: single.color,
      cullMode: single.cullMode,
      frontFace: single.frontFace,
      depthTest: single.depthTest,
      depthWriteEnabled: single.depthWriteEnabled,
      depthCompare: single.depthCompare,
      colorWrite: single.colorWrite,
      writeMask: single.writeMask,
    };
  });
  const first = buildIndependentReferenceInput(meshes[0], camera, width, height, extraOptions);
  return {
    draws,
    projection: first.projection,
    webglDepth: first.webglDepth,
    width,
    height,
    ...extraOptions,
  };
}

async function loadProductionThree(customThree) {
  let THREE = customThree || (typeof window !== "undefined" ? window.THREE : null);
  if (!THREE) {
    try {
      THREE = await import("../../../upstream/three.js/build/three.module.js");
    } catch (e1) {
      try {
        THREE = await import("/upstream/three.js/build/three.module.js");
      } catch (e2) {
        throw new Error(`FATAL: Upstream Three.js could not be loaded: ${e1.message} / ${e2.message}`);
      }
    }
  }
  return THREE;
}

async function loadProductionAdapter(customAdapter) {
  let adapter = customAdapter || (typeof window !== "undefined" ? window.__f3dMeshAdapter : null);
  if (!adapter) {
    try {
      adapter = await import("../../../tools/compat/mesh_adapter.mjs");
    } catch (e1) {
      try {
        adapter = await import("/tools/compat/mesh_adapter.mjs");
      } catch (e2) {
        throw new Error(
          `FATAL: Production tools/compat/mesh_adapter.mjs could not be loaded. ` +
          `Fallback forbidden by root review: ${e1.message} / ${e2.message}`
        );
      }
    }
  }
  if (typeof adapter.prepareMeshPacket !== "function") {
    throw new Error("FATAL: Production mesh_adapter.mjs is missing required prepareMeshPacket export");
  }
  if (typeof adapter.canAdmitMesh !== "function") {
    throw new Error("FATAL: Production mesh_adapter.mjs is missing required canAdmitMesh export");
  }
  return adapter;
}

/**
 * Retained Three.js WebGLRenderer reference midpoint oracle.
 * Renders on an isolated canvas with production configuration:
 * antialias: false, NoToneMapping, outputColorSpace: SRGBColorSpace.
 * Reads center pixel via gl.readPixels to provide ground-truth Three.js parity.
 */
function renderRetainedWebGLReference(THREE, meshOrMeshes, camera, width = 64, height = 64) {
  if (typeof document === "undefined" || typeof document.createElement !== "function") {
    throw new Error("Retained WebGLRenderer reference oracle requires a DOM document environment");
  }
  const refCanvas = document.createElement("canvas");
  refCanvas.width = width;
  refCanvas.height = height;

  const renderer = new THREE.WebGLRenderer({
    canvas: refCanvas,
    antialias: false,
  });
  try {
    renderer.setSize(width, height, false);
    renderer.toneMapping = THREE.NoToneMapping;
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    const scene = new THREE.Scene();
    const meshList = Array.isArray(meshOrMeshes) ? meshOrMeshes : [meshOrMeshes];
    for (const m of meshList) {
      scene.add(m);
    }
    renderer.render(scene, camera);

    const gl = renderer.getContext();
    const pixel = new Uint8Array(4);
    // In WebGL, (0,0) is bottom-left. For canvas center (x=32, y=32),
    // WebGL row is height - 1 - y = 31.
    const glX = Math.floor(width / 2);
    const glY = height - 1 - Math.floor(height / 2);
    gl.readPixels(glX, glY, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel);
    return [pixel[0], pixel[1], pixel[2], pixel[3]];
  } finally {
    renderer.dispose();
    if (typeof refCanvas.remove === "function") {
      refCanvas.remove();
    }
  }
}

/**
 * Suite 1: Offscreen dynamic mesh scene verification suite.
 */
export async function testMeshScene(bridgeHost, wasmExports, customThree = null, customAdapter = null) {
  // 1. Strict production Wasm export check
  const buildMeshFn =
    wasmExports.f3d_build_mesh_packet ||
    wasmExports.gpu_bridge_build_mesh_packet;

  if (typeof buildMeshFn !== "function") {
    throw new Error("Missing required Wasm export f3d_build_mesh_packet / gpu_bridge_build_mesh_packet");
  }

  // 2. Strict production Three.js and adapter loading (no fallbacks)
  const THREE = await loadProductionThree(customThree);
  const adapter = await loadProductionAdapter(customAdapter);

  const width = 64;
  const height = 64;
  const bytesPerRow = Math.ceil((width * 4) / 256) * 256;

  // 3. Construct real Three.js Scene, Mesh, and Camera
  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array([
    -0.5, -0.5, 0.0,
     0.5, -0.5, 0.0,
     0.0,  0.5, 0.0,
  ]);
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));

  const material = new THREE.MeshBasicMaterial({
    color: 0xff0000, // Red
    side: THREE.DoubleSide,
    depthTest: false,
    depthWrite: false,
    transparent: false,
    opacity: 1.0,
  });

  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.set(0, 0, 0);

  const camera = new THREE.PerspectiveCamera(45, 1.0, 0.1, 100.0);
  camera.position.set(0, 0, 3);
  camera.lookAt(0, 0, 0);

  // ---------------------------------------------------------------------------
  // Checkpoint 1: Baseline Match against Independent Direct WebGPU Reference
  // ---------------------------------------------------------------------------
  mesh.updateMatrixWorld(true);
  camera.updateMatrixWorld(true);
  camera.updateProjectionMatrix();

  const { packetBytes: baselinePacket } = adapter.prepareMeshPacket(
    mesh,
    camera,
    width,
    height,
    wasmExports
  );
  await bridgeHost.executePacket(baselinePacket);
  const baselineCandidatePixels = await bridgeHost.readbackBuffer(20, bytesPerRow * height);

  const refInput0 = buildIndependentReferenceInput(mesh, camera, width, height);
  const refPixels0 = await directMeshReference(bridgeHost.device, refInput0);

  if (differs(baselineCandidatePixels, refPixels0)) {
    throw new Error("Baseline candidate mesh pixels differ from independent direct WebGPU reference");
  }

  const centerIdx = 32 * bytesPerRow + 32 * 4;
  const centerPixel = [
    baselineCandidatePixels[centerIdx],
    baselineCandidatePixels[centerIdx + 1],
    baselineCandidatePixels[centerIdx + 2],
    baselineCandidatePixels[centerIdx + 3],
  ];
  if (centerPixel[0] < 200 || centerPixel[1] > 50 || centerPixel[2] > 50 || centerPixel[3] !== 255) {
    throw new Error(`Baseline center pixel expected Red, got [${centerPixel}]`);
  }

  const cornerIdx = 2 * bytesPerRow + 2 * 4;
  const cornerPixel = [
    baselineCandidatePixels[cornerIdx],
    baselineCandidatePixels[cornerIdx + 1],
    baselineCandidatePixels[cornerIdx + 2],
    baselineCandidatePixels[cornerIdx + 3],
  ];
  if (cornerPixel[0] !== 0 || cornerPixel[1] !== 0 || cornerPixel[2] !== 0 || cornerPixel[3] !== 255) {
    throw new Error(`Baseline corner pixel expected Clear Black [0,0,0,255], got [${cornerPixel}]`);
  }

  // ---------------------------------------------------------------------------
  // Checkpoint 2: Translation Mutation Checkpoint
  // ---------------------------------------------------------------------------
  mesh.position.x = 0.4;
  mesh.updateMatrixWorld(true);

  const { packetBytes: transPacket } = adapter.prepareMeshPacket(
    mesh,
    camera,
    width,
    height,
    wasmExports
  );
  await bridgeHost.executePacket(transPacket);
  const transCandidatePixels = await bridgeHost.readbackBuffer(20, bytesPerRow * height);

  const refInput1 = buildIndependentReferenceInput(mesh, camera, width, height);
  const refPixels1 = await directMeshReference(bridgeHost.device, refInput1);

  if (differs(transCandidatePixels, refPixels1)) {
    throw new Error("Translated candidate mesh pixels differ from independent direct WebGPU reference");
  }
  if (!differs(transCandidatePixels, baselineCandidatePixels)) {
    throw new Error("Translation mutation produced identical pixels to baseline; dynamic matrix update failed");
  }

  // ---------------------------------------------------------------------------
  // Checkpoint 3: Color Mutation Checkpoint
  // ---------------------------------------------------------------------------
  mesh.material.color.setRGB(0.0, 1.0, 0.0);

  const { packetBytes: greenPacket } = adapter.prepareMeshPacket(
    mesh,
    camera,
    width,
    height,
    wasmExports
  );
  await bridgeHost.executePacket(greenPacket);
  const greenCandidatePixels = await bridgeHost.readbackBuffer(20, bytesPerRow * height);

  const refInput2 = buildIndependentReferenceInput(mesh, camera, width, height);
  const refPixels2 = await directMeshReference(bridgeHost.device, refInput2);

  if (differs(greenCandidatePixels, refPixels2)) {
    throw new Error("Color-mutated candidate mesh pixels differ from independent direct WebGPU reference");
  }
  if (!differs(greenCandidatePixels, transCandidatePixels)) {
    throw new Error("Color mutation produced identical pixels to previous frame; dynamic color update failed");
  }

  // ---------------------------------------------------------------------------
  // Checkpoint 4: Negative Control - Stale Cached Packet Mismatch
  // ---------------------------------------------------------------------------
  const staleHost = new WebGpuBridgeHost();
  try {
    await staleHost.negotiateAndCreateDevice({ requiredFeatures: [] });
    await staleHost.executePacket(baselinePacket);
    const stalePixels = await staleHost.readbackBuffer(20, bytesPerRow * height);
    if (!differs(stalePixels, refPixels2)) {
      throw new Error("Negative control failed: stale baseline packet falsely matched current mutated reference");
    }
  } finally {
    staleHost.destroyDevice();
  }

  // ---------------------------------------------------------------------------
  // Checkpoint 5: Negative Controls - Admission Rejections via Production canAdmitMesh
  // ---------------------------------------------------------------------------
  // 1. Negative control: depthTest=false with depthWrite=true without options.sourceBackend
  // is ambiguous across backends (WebGL suppresses writes, WebGPU permits writes) and must be refused.
  mesh.material.depthTest = false;
  mesh.material.depthWrite = true;
  const ambiguousAdmission = adapter.canAdmitMesh(mesh, camera);
  if (ambiguousAdmission.admitted) {
    throw new Error("Negative control failed: depthTest=false + depthWrite=true without options.sourceBackend was admitted");
  }
  if (ambiguousAdmission.reason !== adapter.ADMISSION_REJECTION.AMBIGUOUS_DEPTH_PAIR) {
    throw new Error(`Negative control failed: expected AMBIGUOUS_DEPTH_PAIR, got ${ambiguousAdmission.reason}`);
  }
  mesh.material.depthWrite = false;

  // 2. Negative control: polygonOffset is an unsupported material feature
  mesh.material.polygonOffset = true;
  const polygonAdmission = adapter.canAdmitMesh(mesh, camera);
  if (polygonAdmission.admitted) {
    throw new Error("Negative control failed: polygonOffset=true was admitted");
  }
  mesh.material.polygonOffset = false;

  // 3. Negative control: stencilWrite is an unsupported material feature
  mesh.material.stencilWrite = true;
  const stencilAdmission = adapter.canAdmitMesh(mesh, camera);
  if (stencilAdmission.admitted) {
    throw new Error("Negative control failed: stencilWrite=true was admitted");
  }
  mesh.material.stencilWrite = false;

  mesh.material.transparent = true;
  const transparentAdmission = adapter.canAdmitMesh(mesh, camera);
  if (transparentAdmission.admitted) {
    throw new Error("Negative control failed: transparent=true was admitted in opaque slice");
  }
  mesh.material.transparent = false;

  mesh.material.map = {};
  const textureAdmission = adapter.canAdmitMesh(mesh, camera);
  if (textureAdmission.admitted) {
    throw new Error("Negative control failed: textured material was admitted in untextured slice");
  }
  mesh.material.map = null;

  mesh.material.side = THREE.FrontSide;
  const frontAdmission = adapter.canAdmitMesh(mesh, camera);
  if (!frontAdmission.admitted) {
    throw new Error(`Negative control failed: FrontSide was rejected: ${frontAdmission.reason}`);
  }

  mesh.material.side = THREE.BackSide;
  const backAdmission = adapter.canAdmitMesh(mesh, camera);
  if (!backAdmission.admitted) {
    throw new Error(`Negative control failed: BackSide was rejected: ${backAdmission.reason}`);
  }

  mesh.material.side = 999;
  const sideAdmission = adapter.canAdmitMesh(mesh, camera);
  if (sideAdmission.admitted) {
    throw new Error("Negative control failed: invalid side=999 was admitted");
  }
  mesh.material.side = THREE.DoubleSide;

  return "Three.js Mesh dynamic rendering matches independent direct WebGPU reference (getX/getY/getZ/matrices) for baseline, translation, and color mutations; Gray's prepareMeshPacket exercised directly; stale packet and unsupported features correctly rejected";
}

/**
 * Suite 2: Visible canvas dynamic mesh scene verification suite.
 * Exercises visible canvas pass through f3d_build_canvas_mesh_packet /
 * gpu_bridge_build_canvas_mesh_packet and Gray's adapter.
 */
export async function testVisibleCanvasMeshScene(bridgeHost, wasmExports, canvasContext, canvas, customThree = null, customAdapter = null) {
  if (!canvasContext) {
    throw new Error("Visible canvas test requires a real GPUCanvasContext");
  }

  const buildCanvasMeshFn =
    wasmExports.f3d_build_canvas_mesh_packet ||
    wasmExports.gpu_bridge_build_canvas_mesh_packet;

  if (typeof buildCanvasMeshFn !== "function") {
    throw new Error("Missing required Wasm export f3d_build_canvas_mesh_packet / gpu_bridge_build_canvas_mesh_packet");
  }

  // Ensure wasmExports exposes f3d_build_canvas_mesh_packet for adapter calls
  if (typeof wasmExports.f3d_build_canvas_mesh_packet !== "function" && typeof wasmExports.gpu_bridge_build_canvas_mesh_packet === "function") {
    wasmExports.f3d_build_canvas_mesh_packet = wasmExports.gpu_bridge_build_canvas_mesh_packet;
  }

  const THREE = await loadProductionThree(customThree);
  const adapter = await loadProductionAdapter(customAdapter);

  if (typeof adapter.prepareCanvasMeshPacket !== "function") {
    throw new Error("FATAL: Production mesh_adapter.mjs is missing required prepareCanvasMeshPacket export");
  }
  if (typeof adapter.renderMesh !== "function") {
    throw new Error("FATAL: Production mesh_adapter.mjs is missing required renderMesh export");
  }

  const device = bridgeHost.device;
  const canvasFormat = navigator.gpu ? navigator.gpu.getPreferredCanvasFormat() : "bgra8unorm";

  canvasContext.configure({
    device,
    format: canvasFormat,
    alphaMode: "opaque",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });

  const width = canvas.width || 64;
  const height = canvas.height || 64;
  const bytesPerRow = Math.ceil((width * 4) / 256) * 256;

  const canvasReadback = device.createBuffer({
    size: bytesPerRow * height,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  function nextFrame() {
    return new Promise(resolve => {
      if (typeof requestAnimationFrame === "function") {
        requestAnimationFrame(() => resolve());
      } else {
        setTimeout(resolve, 16);
      }
    });
  }

  /**
   * Executes a prepared canvas packet and synchronously submits the readback copy
   * within the SAME turn before any await, preventing swapchain texture expiration.
   */
  async function executeAndReadCanvas(packetBytes) {
    const currentTexture = canvasContext.getCurrentTexture();

    // Start bridge execution WITHOUT awaiting (encode and queue.submit are synchronous)
    const execPromise = bridgeHost.executePacket(packetBytes, canvasContext);

    // In the SAME turn, immediately encode and submit the copy from currentTexture to readback buffer
    const copyEncoder = device.createCommandEncoder();
    copyEncoder.copyTextureToBuffer(
      { texture: currentTexture },
      { buffer: canvasReadback, bytesPerRow },
      [width, height, 1]
    );
    device.queue.submit([copyEncoder.finish()]);

    // Now await execution error scopes and buffer map
    await execPromise;
    await canvasReadback.mapAsync(GPUMapMode.READ);
    const pixels = new Uint8Array(canvasReadback.getMappedRange().slice(0));
    canvasReadback.unmap();
    return pixels;
  }

  try {
    // 1. Construct real Three.js Scene, Mesh, and Camera
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array([
      -0.5, -0.5, 0.0,
       0.5, -0.5, 0.0,
       0.0,  0.5, 0.0,
    ]);
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));

    const material = new THREE.MeshBasicMaterial({
      color: 0xff0000, // Red
      side: THREE.DoubleSide,
      depthTest: false,
      depthWrite: false,
      transparent: false,
      opacity: 1.0,
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(0, 0, 0);

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100.0);
    camera.position.set(0, 0, 3);
    camera.lookAt(0, 0, 0);

    mesh.updateMatrixWorld(true);
    camera.updateMatrixWorld(true);
    camera.updateProjectionMatrix();

    // -------------------------------------------------------------------------
    // Negative Admission Controls
    // -------------------------------------------------------------------------
    // Negative 1: bridgeHost.executePacket with canvas target requires canvasContext
    const { packetBytes: baselinePacket } = adapter.prepareCanvasMeshPacket(
      mesh,
      camera,
      width,
      height,
      wasmExports
    );
    let absentContextRejected = false;
    try {
      await bridgeHost.executePacket(baselinePacket, null);
    } catch (e) {
      if (e.message.includes("canvas target requires a canvas context")) {
        absentContextRejected = true;
      }
    }
    if (!absentContextRejected) {
      throw new Error("Negative control failed: canvas packet execution without canvasContext was not rejected");
    }

    // Negative 2: adapter.renderMesh with canvasContext strictly refuses when wasmModule lacks f3d_build_canvas_mesh_packet
    let missingExportRefused = false;
    try {
      await adapter.renderMesh(bridgeHost, mesh, camera, canvasContext, {});
    } catch (e) {
      if (e.message.includes("does not export f3d_build_canvas_mesh_packet")) {
        missingExportRefused = true;
      }
    }
    if (!missingExportRefused) {
      throw new Error("Negative control failed: renderMesh did not refuse canvasContext when Wasm module lacked f3d_build_canvas_mesh_packet");
    }

    // -------------------------------------------------------------------------
    // Checkpoint 1: Baseline Canvas Render vs Independent Direct Reference
    // -------------------------------------------------------------------------
    await nextFrame();
    const canvasPixels1 = await executeAndReadCanvas(baselinePacket);

    const refInput1 = buildIndependentReferenceInput(mesh, camera, width, height);
    refInput1.format = canvasFormat;
    refInput1.outputSrgb = true;
    const refPixels1 = await directMeshReference(device, refInput1);

    if (differs(canvasPixels1, refPixels1)) {
      throw new Error("Baseline canvas mesh pixels differ from independent direct WebGPU reference in canvas format");
    }

    // Verify center pixel has Red > 200 in canvas format (handles bgra vs rgba)
    const centerIdx = 32 * bytesPerRow + 32 * 4;
    const rCh = canvasFormat.startsWith("bgra") ? 2 : 0;
    const gCh = 1;
    const bCh = canvasFormat.startsWith("bgra") ? 0 : 2;
    if (canvasPixels1[centerIdx + rCh] < 200 || canvasPixels1[centerIdx + gCh] > 50 || canvasPixels1[centerIdx + bCh] > 50) {
      throw new Error(`Baseline canvas center pixel expected Red, got [${canvasPixels1.slice(centerIdx, centerIdx + 4)}]`);
    }

    // -------------------------------------------------------------------------
    // Checkpoint 2: Real rAF Frame Advance & Translation Mutation
    // -------------------------------------------------------------------------
    mesh.position.x = 0.4;
    mesh.updateMatrixWorld(true);

    const { packetBytes: transPacket } = adapter.prepareCanvasMeshPacket(
      mesh,
      camera,
      width,
      height,
      wasmExports
    );
    await nextFrame();
    const canvasPixels2 = await executeAndReadCanvas(transPacket);

    const refInput2 = buildIndependentReferenceInput(mesh, camera, width, height);
    refInput2.format = canvasFormat;
    refInput2.outputSrgb = true;
    const refPixels2 = await directMeshReference(device, refInput2);

    if (differs(canvasPixels2, refPixels2)) {
      throw new Error("Translated canvas mesh pixels differ from independent direct WebGPU reference");
    }
    if (!differs(canvasPixels2, canvasPixels1)) {
      throw new Error("Translation mutation on canvas produced identical pixels to baseline; dynamic matrix update failed");
    }

    // -------------------------------------------------------------------------
    // Checkpoint 3: Real rAF Frame Advance & Color Mutation on Canvas
    // -------------------------------------------------------------------------
    mesh.material.color.setRGB(0.0, 1.0, 0.0); // Green

    const { packetBytes: greenPacket } = adapter.prepareCanvasMeshPacket(
      mesh,
      camera,
      width,
      height,
      wasmExports
    );
    await nextFrame();
    const canvasPixels3 = await executeAndReadCanvas(greenPacket);

    const refInput3 = buildIndependentReferenceInput(mesh, camera, width, height);
    refInput3.format = canvasFormat;
    refInput3.outputSrgb = true;
    const refPixels3 = await directMeshReference(device, refInput3);

    if (differs(canvasPixels3, refPixels3)) {
      throw new Error("Color-mutated canvas mesh pixels differ from independent direct WebGPU reference");
    }
    if (!differs(canvasPixels3, canvasPixels2)) {
      throw new Error("Color mutation on canvas produced identical pixels to previous frame");
    }

    // -------------------------------------------------------------------------
    // Checkpoint 4: Real rAF Frame Advance & Negative Stale Canvas Packet Mismatch
    // -------------------------------------------------------------------------
    // Submit the baseline red packet from Checkpoint 1 on the canvas swapchain
    // while the scene reference is the mutated green mesh:
    await nextFrame();
    const staleCanvasPixels = await executeAndReadCanvas(baselinePacket);

    if (!differs(staleCanvasPixels, refPixels3)) {
      throw new Error("Negative control failed: stale baseline canvas packet falsely matched current mutated reference");
    }

    // -------------------------------------------------------------------------
    // Checkpoint 5: Verify Production adapter.renderMesh with Actual Canvas Pixel Check
    // -------------------------------------------------------------------------
    await nextFrame();
    const currentTexture5 = canvasContext.getCurrentTexture();
    const renderResultPromise = adapter.renderMesh(
      bridgeHost,
      mesh,
      camera,
      canvasContext,
      wasmExports,
      { width, height }
    );
    const copyEncoder5 = device.createCommandEncoder();
    copyEncoder5.copyTextureToBuffer(
      { texture: currentTexture5 },
      { buffer: canvasReadback, bytesPerRow },
      [width, height, 1]
    );
    device.queue.submit([copyEncoder5.finish()]);

    const renderResult = await renderResultPromise;
    await canvasReadback.mapAsync(GPUMapMode.READ);
    const renderMeshPixels = new Uint8Array(canvasReadback.getMappedRange().slice(0));
    canvasReadback.unmap();

    if (!renderResult || renderResult.target !== "canvas") {
      throw new Error(`Expected renderMesh target to be 'canvas', got '${renderResult?.target}'`);
    }
    if (differs(renderMeshPixels, refPixels3)) {
      throw new Error("renderMesh positive canvas output differs from expected reference pixels");
    }

    // -------------------------------------------------------------------------
    // Checkpoint 6: Retained WebGLRenderer Reference Oracle Parity (Midtone sRGB)
    // -------------------------------------------------------------------------
    // Pinned Three.js r186 WebGPURenderer applies outputColorSpace sRGB encoding
    // to canvas outputs. We verify candidate Wasm canvas execution against real
    // retained WebGLRenderer on its own isolated canvas/context.
    // Checkpoint 2 moved the triangle away from the center sample. Restore it
    // so this comparison observes fragment output, rather than two clear pixels.
    mesh.position.x = 0;
    mesh.updateMatrixWorld(true);
    mesh.material.color.setRGB(0.5, 0.5, 0.5);

    const { packetBytes: midtonePacket } = adapter.prepareCanvasMeshPacket(
      mesh,
      camera,
      width,
      height,
      wasmExports
    );
    await nextFrame();
    const candidateMidtonePixels = await executeAndReadCanvas(midtonePacket);

    // Run real retained Three.js WebGLRenderer oracle on its own isolated canvas
    const oracleCenter = renderRetainedWebGLReference(THREE, mesh, camera, width, height);
    if (oracleCenter.slice(0, 3).some(channel => Math.abs(channel - 188) > 2)) {
      throw new Error(`Midtone reference did not cover the sample with sRGB gray: [${oracleCenter}]`);
    }

    // Read candidate center pixel (handling BGRA vs RGBA preferredCanvasFormat)
    const candR = candidateMidtonePixels[centerIdx + rCh];
    const candG = candidateMidtonePixels[centerIdx + gCh];
    const candB = candidateMidtonePixels[centerIdx + bCh];
    const candA = candidateMidtonePixels[centerIdx + 3];

    // Tolerance 2 match against actual WebGLRenderer readPixels oracle
    const diffR = Math.abs(candR - oracleCenter[0]);
    const diffG = Math.abs(candG - oracleCenter[1]);
    const diffB = Math.abs(candB - oracleCenter[2]);
    const diffA = Math.abs(candA - oracleCenter[3]);

    if (diffR > 2 || diffG > 2 || diffB > 2 || diffA > 2) {
      throw new Error(
        `Checkpoint 6 failed: Candidate canvas midtone pixel [${candR}, ${candG}, ${candB}, ${candA}] ` +
        `differs from retained WebGLRenderer reference oracle [${oracleCenter}] by > tolerance 2 ` +
        `(diffs: R=${diffR}, G=${diffG}, B=${diffB}, A=${diffA})`
      );
    }

    // Planted negative control: raw linear byte (128 for 0.5) must strictly fail sRGB parity
    const rawLinearByte = Math.round(0.5 * 255); // 128
    const linearMismatch = Math.abs(rawLinearByte - oracleCenter[0]);
    if (linearMismatch <= 2) {
      throw new Error(
        `Planted negative failed: raw linear byte ${rawLinearByte} was not rejected by sRGB oracle ${oracleCenter[0]}`
      );
    }

    return "Visible canvas Three.js Mesh rendering matches independent direct WebGPU reference (preferredCanvasFormat) across rAF-separated presentation frames; translation and color mutations observed on swapchain; stale packet and absent canvasContext correctly rejected; Gray's prepareCanvasMeshPacket and renderMesh (with canvas pixel verification) verified directly; retained WebGLRenderer midtone sRGB reference oracle matches candidate within tolerance 2 and rejects planted linear byte";
  } finally {
    canvasReadback.destroy();
  }
}

/**
 * Executes dynamic Three.js Mesh depth mutations through production mesh_adapter -> Wasm exports.
 *
 * Verification Requirements (§6.1, §6.7, §8.5):
 * 1. Real THREE.Mesh depthTest, depthWrite, and depthFunc mutation through actual Wasm new exports.
 * 2. Independent direct-WebGPU oracle reference for exact bit-for-bit comparison.
 * 3. Compares clear vs visible for NeverDepth (clear/suppressed) vs AlwaysDepth/LessDepth (visible).
 * 4. Verifies depth-write-disabled behavior against independent direct WebGPU reference.
 * 5. Strict missing-export failure on required functions.
 *
 * @param {WebGpuBridgeHost} bridgeHost
 * @param {object} wasmExports
 * @param {GPUCanvasContext} [canvasContext]
 * @param {HTMLCanvasElement} [canvas]
 * @returns {Promise<string>}
 */
export async function testMeshDepthScene(bridgeHost, wasmExports, canvasContext = null, canvas = null) {
  const buildMeshDepthFn =
    wasmExports?.f3d_build_mesh_depth_packet ||
    wasmExports?.gpu_bridge_build_mesh_depth_packet;

  if (typeof buildMeshDepthFn !== "function") {
    throw new Error(
      "Missing required Wasm mesh depth export: f3d_build_mesh_depth_packet / gpu_bridge_build_mesh_depth_packet"
    );
  }

  const THREE = await loadProductionThree();
  const adapter = await loadProductionAdapter();

  const device = bridgeHost.device;
  const width = 64;
  const height = 64;
  const bytesPerRow = Math.ceil((width * 4) / 256) * 256;
  const centerIdx = 32 * bytesPerRow + 32 * 4;

  // Geometry: triangle covering center, in z = -2.0 plane
  const geomNear = new THREE.BufferGeometry();
  const posNear = new Float32Array([
    -1.0, -1.0, -2.0,
     1.0, -1.0, -2.0,
     0.0,  1.0, -2.0,
  ]);
  geomNear.setAttribute("position", new THREE.BufferAttribute(posNear, 3));

  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
  camera.position.set(0, 0, 0);
  camera.lookAt(0, 0, -1);
  camera.updateMatrixWorld(true);
  camera.updateProjectionMatrix();

  const greenMaterial = new THREE.MeshBasicMaterial({
    color: 0x00ff00,
    side: THREE.DoubleSide,
    depthTest: true,
    depthWrite: true,
    depthFunc: THREE.LessDepth,
  });
  const meshNear = new THREE.Mesh(geomNear, greenMaterial);
  meshNear.updateMatrixWorld(true);

  // Helper to extract raw direct reference parameters from real Three objects
  function getRawDirectOptions(targetMesh, targetCamera, overrides = {}) {
    const pos = targetMesh.geometry.attributes.position.array;
    const mv = targetCamera.matrixWorldInverse.clone().multiply(targetMesh.matrixWorld).elements;
    const proj = targetCamera.projectionMatrix.elements;
    const col = [
      targetMesh.material.color.r,
      targetMesh.material.color.g,
      targetMesh.material.color.b,
      targetMesh.material.opacity ?? 1.0,
    ];
    return {
      positions: pos,
      indices: null,
      modelView: mv,
      projection: proj,
      webglDepth: true,
      color: col,
      width,
      height,
      depthFormat: "depth24plus",
      ...overrides,
    };
  }

  // ---------------------------------------------------------------------------
  // Checkpoint 1: Standard LessDepth -> Mesh Visible (Green)
  // ---------------------------------------------------------------------------
  meshNear.material.depthFunc = THREE.LessDepth;
  meshNear.material.depthTest = true;
  meshNear.material.depthWrite = true;
  meshNear.material.needsUpdate = true;

  const { packetBytes: packetLess } = adapter.prepareMeshDepthPacket
    ? adapter.prepareMeshDepthPacket(meshNear, camera, width, height, wasmExports)
    : adapter.prepareMeshPacket(meshNear, camera, width, height, wasmExports);

  await bridgeHost.executePacket(packetLess);
  const candidateLess = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
  const refLess = await directMeshReference(
    device,
    getRawDirectOptions(meshNear, camera, {
      hasDepth: true,
      depthCompare: "less",
      depthWriteEnabled: true,
    })
  );

  if (differs(candidateLess, refLess)) {
    throw new Error("Checkpoint 1 failed: LessDepth candidate differs from direct WebGPU reference");
  }
  const centerLess = [
    candidateLess[centerIdx],
    candidateLess[centerIdx + 1],
    candidateLess[centerIdx + 2],
    candidateLess[centerIdx + 3],
  ];
  if (centerLess[0] > 50 || centerLess[1] < 200 || centerLess[2] > 50) {
    throw new Error(`Checkpoint 1 failed: expected visible Green, got [${centerLess}]`);
  }

  // ---------------------------------------------------------------------------
  // Checkpoint 2: NeverDepth Mutation -> Depth Test Rejects, Clear Color Remains
  // ---------------------------------------------------------------------------
  meshNear.material.depthFunc = THREE.NeverDepth;
  meshNear.material.needsUpdate = true;

  const { packetBytes: packetNever } = adapter.prepareMeshDepthPacket
    ? adapter.prepareMeshDepthPacket(meshNear, camera, width, height, wasmExports)
    : adapter.prepareMeshPacket(meshNear, camera, width, height, wasmExports);

  await bridgeHost.executePacket(packetNever);
  const candidateNever = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
  const refNever = await directMeshReference(
    device,
    getRawDirectOptions(meshNear, camera, {
      hasDepth: true,
      depthCompare: "never",
      depthWriteEnabled: true,
    })
  );

  if (differs(candidateNever, refNever)) {
    throw new Error("Checkpoint 2 failed: NeverDepth candidate differs from direct WebGPU reference");
  }
  const centerNever = [
    candidateNever[centerIdx],
    candidateNever[centerIdx + 1],
    candidateNever[centerIdx + 2],
    candidateNever[centerIdx + 3],
  ];
  // Must remain clear color [0, 0, 0, 1] / [0, 0, 0, 255]
  if (centerNever[0] !== 0 || centerNever[1] !== 0 || centerNever[2] !== 0) {
    throw new Error(`Checkpoint 2 failed: expected clear black for NeverDepth, got [${centerNever}]`);
  }
  if (!differs(candidateNever, candidateLess)) {
    throw new Error("Checkpoint 2 failed: NeverDepth did not suppress mesh rendering");
  }

  // ---------------------------------------------------------------------------
  // Checkpoint 3: AlwaysDepth Mutation -> Depth Test Passes, Mesh Visible
  // ---------------------------------------------------------------------------
  meshNear.material.depthFunc = THREE.AlwaysDepth;
  meshNear.material.needsUpdate = true;

  const { packetBytes: packetAlways } = adapter.prepareMeshDepthPacket
    ? adapter.prepareMeshDepthPacket(meshNear, camera, width, height, wasmExports)
    : adapter.prepareMeshPacket(meshNear, camera, width, height, wasmExports);

  await bridgeHost.executePacket(packetAlways);
  const candidateAlways = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
  const refAlways = await directMeshReference(
    device,
    getRawDirectOptions(meshNear, camera, {
      hasDepth: true,
      depthCompare: "always",
      depthWriteEnabled: true,
    })
  );

  if (differs(candidateAlways, refAlways)) {
    throw new Error("Checkpoint 3 failed: AlwaysDepth candidate differs from direct WebGPU reference");
  }
  const centerAlways = [
    candidateAlways[centerIdx],
    candidateAlways[centerIdx + 1],
    candidateAlways[centerIdx + 2],
    candidateAlways[centerIdx + 3],
  ];
  if (centerAlways[0] > 50 || centerAlways[1] < 200 || centerAlways[2] > 50) {
    throw new Error(`Checkpoint 3 failed: expected visible Green for AlwaysDepth, got [${centerAlways}]`);
  }

  // ---------------------------------------------------------------------------
  // Checkpoint 4: depthTest = false Disambiguation via options.sourceBackend
  // ---------------------------------------------------------------------------
  meshNear.material.depthTest = false;
  meshNear.material.depthWrite = true;
  meshNear.material.depthFunc = THREE.NeverDepth; // Even with NeverDepth, depthTest=false overrides to Always
  meshNear.material.needsUpdate = true;

  // 4a: Ambiguous pair without options.sourceBackend MUST be refused
  let ambiguousRefused = false;
  try {
    if (adapter.prepareMeshDepthPacket) {
      adapter.prepareMeshDepthPacket(meshNear, camera, width, height, wasmExports);
    } else {
      adapter.prepareMeshPacket(meshNear, camera, width, height, wasmExports);
    }
  } catch (e) {
    if (e.message.includes("ambiguous across backends") || e.message.includes("AMBIGUOUS_DEPTH_PAIR")) {
      ambiguousRefused = true;
    }
  }
  if (!ambiguousRefused) {
    throw new Error("Checkpoint 4 failed: depthTest=false + depthWrite=true without options.sourceBackend was not refused");
  }

  // 4b: Positive WebGPU backend: passes depthWrite=true directly (r186 WebGPUPipelineUtils.js:224)
  const { packetBytes: packetWebGPU } = adapter.prepareMeshDepthPacket
    ? adapter.prepareMeshDepthPacket(meshNear, camera, width, height, wasmExports, { sourceBackend: "webgpu" })
    : adapter.prepareMeshPacket(meshNear, camera, width, height, wasmExports, { sourceBackend: "webgpu" });

  await bridgeHost.executePacket(packetWebGPU);
  const candidateWebGPU = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
  const refWebGPU = await directMeshReference(
    device,
    getRawDirectOptions(meshNear, camera, {
      hasDepth: true,
      depthCompare: "always",
      depthWriteEnabled: true,
    })
  );

  if (differs(candidateWebGPU, refWebGPU)) {
    throw new Error("Checkpoint 4b failed: depthTest=false with sourceBackend='webgpu' differs from direct WebGPU reference");
  }
  const centerWebGPU = [
    candidateWebGPU[centerIdx],
    candidateWebGPU[centerIdx + 1],
    candidateWebGPU[centerIdx + 2],
    candidateWebGPU[centerIdx + 3],
  ];
  if (centerWebGPU[0] > 50 || centerWebGPU[1] < 200 || centerWebGPU[2] > 50) {
    throw new Error(`Checkpoint 4b failed: expected visible Green for sourceBackend='webgpu', got [${centerWebGPU}]`);
  }

  // 4c: Positive WebGL backend: disabled depth test suppresses writes in hardware (depthWriteEnabled = false)
  const { packetBytes: packetWebGL } = adapter.prepareMeshDepthPacket
    ? adapter.prepareMeshDepthPacket(meshNear, camera, width, height, wasmExports, { sourceBackend: "webgl" })
    : adapter.prepareMeshPacket(meshNear, camera, width, height, wasmExports, { sourceBackend: "webgl" });

  await bridgeHost.executePacket(packetWebGL);
  const candidateWebGL = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
  const refWebGL = await directMeshReference(
    device,
    getRawDirectOptions(meshNear, camera, {
      hasDepth: true,
      depthCompare: "always",
      depthWriteEnabled: false,
    })
  );

  if (differs(candidateWebGL, refWebGL)) {
    throw new Error("Checkpoint 4c failed: depthTest=false with sourceBackend='webgl' differs from direct WebGPU reference");
  }
  const centerWebGL = [
    candidateWebGL[centerIdx],
    candidateWebGL[centerIdx + 1],
    candidateWebGL[centerIdx + 2],
    candidateWebGL[centerIdx + 3],
  ];
  if (centerWebGL[0] > 50 || centerWebGL[1] < 200 || centerWebGL[2] > 50) {
    throw new Error(`Checkpoint 4c failed: expected visible Green for sourceBackend='webgl', got [${centerWebGL}]`);
  }

  // ---------------------------------------------------------------------------
  // Checkpoint 5: Depth Write Disabled Behavior Verification
  // ---------------------------------------------------------------------------
  meshNear.material.depthTest = true;
  meshNear.material.depthWrite = false;
  meshNear.material.depthFunc = THREE.LessDepth;
  meshNear.material.needsUpdate = true;

  const { packetBytes: packetWriteDisabled } = adapter.prepareMeshDepthPacket
    ? adapter.prepareMeshDepthPacket(meshNear, camera, width, height, wasmExports)
    : adapter.prepareMeshPacket(meshNear, camera, width, height, wasmExports);

  await bridgeHost.executePacket(packetWriteDisabled);
  const candidateWriteDisabled = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
  const refWriteDisabled = await directMeshReference(
    device,
    getRawDirectOptions(meshNear, camera, {
      hasDepth: true,
      depthCompare: "less",
      depthWriteEnabled: false,
    })
  );

  if (differs(candidateWriteDisabled, refWriteDisabled)) {
    throw new Error("Checkpoint 5 failed: depthWrite=false candidate differs from direct WebGPU reference");
  }

  // ---------------------------------------------------------------------------
  // Checkpoint 6: Visible Canvas Mesh with Depth (when canvasContext is provided)
  // ---------------------------------------------------------------------------
  if (canvasContext) {
    const buildCanvasDepthFn =
      wasmExports?.f3d_build_canvas_mesh_depth_packet ||
      wasmExports?.gpu_bridge_build_canvas_mesh_depth_packet;

    if (typeof buildCanvasDepthFn !== "function") {
      throw new Error(
        "Visible canvas mesh depth requested, but Wasm module is missing required export: f3d_build_canvas_mesh_depth_packet / gpu_bridge_build_canvas_mesh_depth_packet"
      );
    }

    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();

    canvasContext.configure({
      device,
      format: canvasFormat,
      alphaMode: "opaque",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    await nextFrame();

    const { packetBytes: canvasDepthPacket } = adapter.prepareCanvasMeshDepthPacket
      ? adapter.prepareCanvasMeshDepthPacket(meshNear, camera, width, height, wasmExports)
      : adapter.prepareCanvasMeshPacket(meshNear, camera, width, height, wasmExports);

    const canvasReadback = device.createBuffer({
      size: bytesPerRow * height,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const currentTexture = canvasContext.getCurrentTexture();
    const execPromise = bridgeHost.executePacket(canvasDepthPacket, canvasContext);

    const copyEncoder = device.createCommandEncoder();
    copyEncoder.copyTextureToBuffer(
      { texture: currentTexture },
      { buffer: canvasReadback, bytesPerRow },
      [width, height, 1]
    );
    device.queue.submit([copyEncoder.finish()]);

    await execPromise;
    await canvasReadback.mapAsync(GPUMapMode.READ);
    const canvasPixels = new Uint8Array(canvasReadback.getMappedRange().slice(0));
    canvasReadback.unmap();
    canvasReadback.destroy();

    const refCanvas = await directMeshReference(
      device,
      getRawDirectOptions(meshNear, camera, {
        hasDepth: true,
        depthCompare: "less",
        depthWriteEnabled: false,
        format: canvasFormat,
        outputSrgb: true,
      })
    );

    if (differs(canvasPixels, refCanvas)) {
      const diffs = [];
      let maxDiff = 0;
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const idx = y * bytesPerRow + x * 4;
          const c = [canvasPixels[idx], canvasPixels[idx + 1], canvasPixels[idx + 2], canvasPixels[idx + 3]];
          const r = [refCanvas[idx], refCanvas[idx + 1], refCanvas[idx + 2], refCanvas[idx + 3]];
          const dR = Math.abs(c[0] - r[0]);
          const dG = Math.abs(c[1] - r[1]);
          const dB = Math.abs(c[2] - r[2]);
          const dA = Math.abs(c[3] - r[3]);
          const d = Math.max(dR, dG, dB, dA);
          if (d > 0) {
            if (d > maxDiff) maxDiff = d;
            if (diffs.length < 8) {
              diffs.push(`(${x},${y}): cand=[${c}] ref=[${r}] diff=[${dR},${dG},${dB},${dA}]`);
            }
          }
        }
      }
      throw new Error(
        `Checkpoint 6 failed: Visible canvas mesh depth pixels differ from direct WebGPU reference in canvas format (${canvasFormat}). ` +
        `Matched settings: depthFormat=depth24plus, depthCompare=less, depthWrite=false, outputSrgb=true. ` +
        `Max component diff: ${maxDiff}. Sample differing pixels: ${diffs.join("; ")}`
      );
    }
  }

  return "Real THREE.Mesh depthTest/depthWrite/depthFunc mutation verified (depth24plus): LessDepth produces visible mesh matching direct WebGPU reference; NeverDepth suppresses rendering (clear color preserved); AlwaysDepth renders visible mesh; depthTest=false without sourceBackend refused (AMBIGUOUS_DEPTH_PAIR); sourceBackend='webgpu' enables depthWrite flag; sourceBackend='webgl' suppresses depthWrite flag; depthWrite=false flag and selected cases verified against independent oracle (single-mesh draw verifies pipeline write flags, multi-object occlusion verified by static depth fixture)" + (canvasContext ? "; visible canvas mesh depth verified against direct reference" : "");
}

/**
 * Suite 3b: Dynamic Three.js MeshToonMaterial with DirectionalLight verification suite.
 *
 * Verification Requirements:
 * 1. One real Three.js r186 Mesh with MeshToonMaterial (no gradientMap) and one DirectionalLight,
 *    rendered by the existing upstream reference path (renderRetainedWebGLReference).
 * 2. Derive candidate inputs from the SAME objects with citations:
 *    - mesh.matrixWorld (upstream/three.js/src/core/Object3D.js:1176)
 *    - camera.projectionMatrix (upstream/three.js/src/cameras/PerspectiveCamera.js:337)
 *    - camera.matrixWorldInverse (upstream/three.js/src/cameras/Camera.js:122)
 *    - source normal matrix Matrix3 (upstream/three.js/src/nodes/accessors/ModelNode.js:112 and upstream/three.js/src/math/Matrix3.js:352)
 *    - material.color (upstream/three.js/src/materials/MeshToonMaterial.js:34)
 *    - view-space light direction (upstream/three.js/src/renderers/webgl/WebGLLights.js:579-582 and upstream/three.js/src/nodes/accessors/Lights.js:139)
 *    - light.color * intensity (upstream/three.js/src/renderers/webgl/WebGLLights.js:578)
 *    - material.side (upstream/three.js/src/constants.js:1-3)
 * 3. Checks (EXPECTED VALUES ONLY, tolerance 2):
 *    - Frame 1: Candidate probe pixels match upstream reference within tolerance 2.
 *    - Frame 2: Change light direction; Frame 2 candidate matches upstream frame 2 AND differs from frame 1.
 *    - Checkpoint 3: DoubleSide orientation handling matches upstream reference.
 *    - Checkpoint 4 (canvas): Candidate visible swapchain matches retained WebGLRenderer sRGB reference.
 *
 * @param {WebGpuBridgeHost} bridgeHost
 * @param {Record<string, Function>} wasmExports
 * @param {GPUCanvasContext} [canvasContext]
 * @param {HTMLCanvasElement} [canvas]
 * @returns {Promise<string>}
 */
export async function testToonMeshScene(bridgeHost, wasmExports, canvasContext = null, canvas = null) {
  const buildToonMeshFn =
    wasmExports?.f3d_build_toon_mesh_packet ||
    wasmExports?.gpu_bridge_build_toon_mesh_packet;

  if (typeof buildToonMeshFn !== "function") {
    throw new Error(
      "Missing required Wasm toon export: f3d_build_toon_mesh_packet / gpu_bridge_build_toon_mesh_packet"
    );
  }

  const THREE = await loadProductionThree();
  const adapter = await loadProductionAdapter();

  const width = 64;
  const height = 64;
  const bytesPerRow = Math.ceil((width * 4) / 256) * 256;
  const centerIdx = 32 * bytesPerRow + 32 * 4;

  // Real Three.js r186 geometry: triangle covering center (32, 32) in z = -2.0 plane
  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array([
    -1.0, -1.0, -2.0,
     1.0, -1.0, -2.0,
     0.0,  1.0, -2.0,
  ]);
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.computeVertexNormals();

  const uvs = new Float32Array([0.0, 0.0, 1.0, 0.0, 0.5, 1.0]);
  geometry.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));

  const indices = new Uint32Array([0, 1, 2]);
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));

  // Real Three.js r186 MeshToonMaterial (no gradientMap)
  const material = new THREE.MeshToonMaterial({
    color: 0x2288cc,
    side: THREE.FrontSide,
  });

  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.set(0, 0, 0);

  // Real Three.js r186 DirectionalLight
  const light = new THREE.DirectionalLight(0xffffff, 1.0);
  light.position.set(0, 0, 1);
  light.target.position.set(0, 0, 0);

  // Real Three.js r186 PerspectiveCamera
  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
  camera.position.set(0, 0, 0);
  camera.lookAt(0, 0, -1);

  // Ensure initial world matrices and projections are clean
  mesh.updateMatrixWorld(true);
  light.updateMatrixWorld(true);
  light.target.updateMatrixWorld(true);
  camera.updateMatrixWorld(true);
  camera.updateProjectionMatrix();

  // Helper to build candidate packet from the SAME Three.js objects with exact citations
  function buildCandidatePacket(targetMesh, targetLight, targetCamera, overrides = {}) {
    const pos = targetMesh.geometry.attributes.position.array;
    const norm = targetMesh.geometry.attributes.normal.array;
    const uv = targetMesh.geometry.attributes.uv ? targetMesh.geometry.attributes.uv.array : new Float32Array(0);
    const ind = targetMesh.geometry.index ? new Uint32Array(targetMesh.geometry.index.array) : new Uint32Array(0);

    // 1. mesh.matrixWorld (upstream/three.js/src/core/Object3D.js:1176)
    const modelWorld = new Float64Array(targetMesh.matrixWorld.elements);

    // 2. camera.projectionMatrix (upstream/three.js/src/cameras/PerspectiveCamera.js:337)
    const projection = new Float64Array(targetCamera.projectionMatrix.elements);

    // 3. camera.matrixWorldInverse (upstream/three.js/src/cameras/Camera.js:122)
    const cameraView = new Float64Array(targetCamera.matrixWorldInverse.elements);

    // 4. source normal matrix Matrix3 (upstream/three.js/src/nodes/accessors/ModelNode.js:112 and upstream/three.js/src/math/Matrix3.js:352)
    const normalMatrix3 = new THREE.Matrix3().getNormalMatrix(targetMesh.matrixWorld);
    const modelNormalMatrix = new Float64Array(normalMatrix3.elements);

    // 5. material.color (upstream/three.js/src/materials/MeshToonMaterial.js:34)
    const color = new Float32Array([
      targetMesh.material.color.r,
      targetMesh.material.color.g,
      targetMesh.material.color.b,
      targetMesh.material.opacity ?? 1.0,
    ]);

    // 6. view-space light direction (upstream/three.js/src/renderers/webgl/WebGLLights.js:579-582 and upstream/three.js/src/nodes/accessors/Lights.js:139)
    const lightWorldPos = new THREE.Vector3().setFromMatrixPosition(targetLight.matrixWorld);
    const targetWorldPos = new THREE.Vector3().setFromMatrixPosition(targetLight.target.matrixWorld);
    const lightDirView = lightWorldPos.sub(targetWorldPos).transformDirection(targetCamera.matrixWorldInverse).normalize();
    const lightDirection = new Float32Array([lightDirView.x, lightDirView.y, lightDirView.z, 0.0]);

    // 7. light.color * intensity (upstream/three.js/src/renderers/webgl/WebGLLights.js:578)
    const lightColor = new Float32Array([
      targetLight.color.r * targetLight.intensity,
      targetLight.color.g * targetLight.intensity,
      targetLight.color.b * targetLight.intensity,
      1.0,
    ]);

    // 8. material.side (upstream/three.js/src/constants.js:1-3)
    const side = targetMesh.material.side ?? THREE.FrontSide;

    const webglDepth = overrides.webglDepth !== undefined ? overrides.webglDepth : true;
    const canvasMode = overrides.canvas !== undefined ? overrides.canvas : false;

    return buildToonMeshFn(
      pos,
      norm,
      uv,
      ind,
      modelWorld,
      projection,
      cameraView,
      modelNormalMatrix,
      color,
      lightDirection,
      lightColor,
      side,
      width,
      height,
      webglDepth,
      canvasMode
    );
  }

  // Helper to execute upstream reference via existing renderRetainedWebGLReference
  function renderUpstreamReference(targetMesh, targetLight, targetCamera) {
    const prevOnBefore = targetMesh.onBeforeRender;
    targetMesh.onBeforeRender = (renderer, scene, cam, geom, mat, grp) => {
      renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
      if (typeof prevOnBefore === "function") {
        prevOnBefore.call(targetMesh, renderer, scene, cam, geom, mat, grp);
      }
    };
    try {
      return renderRetainedWebGLReference(
        THREE,
        [targetMesh, targetLight, targetLight.target],
        targetCamera,
        width,
        height
      );
    } finally {
      targetMesh.onBeforeRender = prevOnBefore;
    }
  }

  // ---------------------------------------------------------------------------
  // Checkpoint 1: Frame 1 - Baseline Parity with Upstream WebGLRenderer Reference
  // ---------------------------------------------------------------------------
  mesh.updateMatrixWorld(true);
  light.updateMatrixWorld(true);
  light.target.updateMatrixWorld(true);
  camera.updateMatrixWorld(true);
  camera.updateProjectionMatrix();

  const ref1 = renderUpstreamReference(mesh, light, camera);

  const packet1 = buildCandidatePacket(mesh, light, camera);
  await bridgeHost.executePacket(packet1);
  const candidatePixels1 = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
  const candProbe1 = [
    candidatePixels1[centerIdx],
    candidatePixels1[centerIdx + 1],
    candidatePixels1[centerIdx + 2],
    candidatePixels1[centerIdx + 3],
  ];

  const diffR1 = Math.abs(candProbe1[0] - ref1[0]);
  const diffG1 = Math.abs(candProbe1[1] - ref1[1]);
  const diffB1 = Math.abs(candProbe1[2] - ref1[2]);
  const diffA1 = Math.abs(candProbe1[3] - ref1[3]);

  if (diffR1 > 2 || diffG1 > 2 || diffB1 > 2 || diffA1 > 2) {
    throw new Error(
      `Checkpoint 1 failed: Candidate toon probe [${candProbe1}] differs from upstream reference [${ref1}] by > tolerance 2 ` +
      `(diffs: R=${diffR1}, G=${diffG1}, B=${diffB1}, A=${diffA1})`
    );
  }

  // ---------------------------------------------------------------------------
  // Checkpoint 2: Frame 2 - Mutate Light Direction
  // ---------------------------------------------------------------------------
  light.position.set(0, 0, -5);
  light.updateMatrixWorld(true);
  light.target.updateMatrixWorld(true);

  const ref2 = renderUpstreamReference(mesh, light, camera);

  const packet2 = buildCandidatePacket(mesh, light, camera);
  await bridgeHost.executePacket(packet2);
  const candidatePixels2 = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
  const candProbe2 = [
    candidatePixels2[centerIdx],
    candidatePixels2[centerIdx + 1],
    candidatePixels2[centerIdx + 2],
    candidatePixels2[centerIdx + 3],
  ];

  const diffR2 = Math.abs(candProbe2[0] - ref2[0]);
  const diffG2 = Math.abs(candProbe2[1] - ref2[1]);
  const diffB2 = Math.abs(candProbe2[2] - ref2[2]);
  const diffA2 = Math.abs(candProbe2[3] - ref2[3]);

  if (diffR2 > 2 || diffG2 > 2 || diffB2 > 2 || diffA2 > 2) {
    throw new Error(
      `Checkpoint 2 failed: Frame 2 candidate toon probe [${candProbe2}] differs from upstream reference [${ref2}] by > tolerance 2 ` +
      `(diffs: R=${diffR2}, G=${diffG2}, B=${diffB2}, A=${diffA2})`
    );
  }

  const frameDiff = Math.max(
    Math.abs(candProbe2[0] - candProbe1[0]),
    Math.abs(candProbe2[1] - candProbe1[1]),
    Math.abs(candProbe2[2] - candProbe1[2])
  );
  if (frameDiff <= 2) {
    throw new Error(
      `Checkpoint 2 failed: Frame 2 candidate [${candProbe2}] did not differ from Frame 1 [${candProbe1}] after light direction change (maxDiff=${frameDiff})`
    );
  }

  // ---------------------------------------------------------------------------
  // Checkpoint 3: Frame 3 - Material Side Handling (DoubleSide)
  // ---------------------------------------------------------------------------
  mesh.material.side = THREE.DoubleSide;
  mesh.material.needsUpdate = true;

  const ref3Double = renderUpstreamReference(mesh, light, camera);

  const packet3Double = buildCandidatePacket(mesh, light, camera);
  await bridgeHost.executePacket(packet3Double);
  const candidatePixels3Double = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
  const candProbe3Double = [
    candidatePixels3Double[centerIdx],
    candidatePixels3Double[centerIdx + 1],
    candidatePixels3Double[centerIdx + 2],
    candidatePixels3Double[centerIdx + 3],
  ];

  const diffR3D = Math.abs(candProbe3Double[0] - ref3Double[0]);
  const diffG3D = Math.abs(candProbe3Double[1] - ref3Double[1]);
  const diffB3D = Math.abs(candProbe3Double[2] - ref3Double[2]);
  const diffA3D = Math.abs(candProbe3Double[3] - ref3Double[3]);

  if (diffR3D > 2 || diffG3D > 2 || diffB3D > 2 || diffA3D > 2) {
    throw new Error(
      `Checkpoint 3 failed: DoubleSide candidate toon probe [${candProbe3Double}] differs from upstream reference [${ref3Double}] by > tolerance 2 ` +
      `(diffs: R=${diffR3D}, G=${diffG3D}, B=${diffB3D}, A=${diffA3D})`
    );
  }

  // Restore material side
  mesh.material.side = THREE.FrontSide;
  mesh.material.needsUpdate = true;

  // ---------------------------------------------------------------------------
  // Checkpoint 4: Visible Canvas sRGB Presentation (when canvasContext is provided)
  // ---------------------------------------------------------------------------
  let canvasFormat = null;
  if (canvasContext) {
    canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    canvasContext.configure({
      device: bridgeHost.device,
      format: canvasFormat,
      alphaMode: "opaque",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    await nextFrame();

    // Use the frame-1 scene (FrontSide, original light)
    mesh.material.side = THREE.FrontSide;
    mesh.material.needsUpdate = true;
    light.position.set(0, 0, 1);
    light.target.position.set(0, 0, 0);
    mesh.updateMatrixWorld(true);
    light.updateMatrixWorld(true);
    light.target.updateMatrixWorld(true);
    camera.updateMatrixWorld(true);
    camera.updateProjectionMatrix();

    // Build candidate with explicit canvas argument set to true
    const packet4Canvas = buildCandidatePacket(mesh, light, camera, { canvas: true });

    // Read pixels the same way testVisibleCanvasMeshScene does
    const canvasReadback = bridgeHost.device.createBuffer({
      size: bytesPerRow * height,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    let canvasPixels;
    try {
      const currentTexture = canvasContext.getCurrentTexture();
      const execPromise = bridgeHost.executePacket(packet4Canvas, canvasContext);

      const copyEncoder = bridgeHost.device.createCommandEncoder();
      copyEncoder.copyTextureToBuffer(
        { texture: currentTexture },
        { buffer: canvasReadback, bytesPerRow },
        [width, height, 1]
      );
      bridgeHost.device.queue.submit([copyEncoder.finish()]);

      await execPromise;
      await canvasReadback.mapAsync(GPUMapMode.READ);
      canvasPixels = new Uint8Array(canvasReadback.getMappedRange().slice(0));
      canvasReadback.unmap();
    } finally {
      canvasReadback.destroy();
    }

    // Apply bgra/rgba channel mapping from getPreferredCanvasFormat
    const rCh = canvasFormat.startsWith("bgra") ? 2 : 0;
    const gCh = 1;
    const bCh = canvasFormat.startsWith("bgra") ? 0 : 2;

    const candCanvasR = canvasPixels[centerIdx + rCh];
    const candCanvasG = canvasPixels[centerIdx + gCh];
    const candCanvasB = canvasPixels[centerIdx + bCh];
    const candCanvasA = canvasPixels[centerIdx + 3];

    // Compare center probe with renderRetainedWebGLReference using SRGBColorSpace (default, NOT Linear override)
    const refCanvasSRGB = renderRetainedWebGLReference(
      THREE,
      [mesh, light, light.target],
      camera,
      width,
      height
    );

    const diffCanvasR = Math.abs(candCanvasR - refCanvasSRGB[0]);
    const diffCanvasG = Math.abs(candCanvasG - refCanvasSRGB[1]);
    const diffCanvasB = Math.abs(candCanvasB - refCanvasSRGB[2]);
    const diffCanvasA = Math.abs(candCanvasA - refCanvasSRGB[3]);

    if (diffCanvasR > 2 || diffCanvasG > 2 || diffCanvasB > 2 || diffCanvasA > 2) {
      throw new Error(
        `Checkpoint 4 failed: Candidate canvas toon probe [${candCanvasR}, ${candCanvasG}, ${candCanvasB}, ${candCanvasA}] ` +
        `differs from retained WebGLRenderer reference oracle (${canvasFormat}, sRGB) [${refCanvasSRGB}] by > tolerance 2 ` +
        `(diffs: R=${diffCanvasR}, G=${diffCanvasG}, B=${diffCanvasB}, A=${diffCanvasA})`
      );
    }
  }

  const baseResult = "Real THREE.Mesh with MeshToonMaterial and DirectionalLight verified against upstream WebGLRenderer reference oracle (tolerance 2): Frame 1 candidate matches upstream reference on identical inputs; Frame 2 light direction change matches upstream frame 2 and strictly differs from frame 1; DoubleSide orientation handling verified against upstream reference; offscreen rgba8unorm only; canvas path not exercised";
  return canvasContext
    ? `${baseResult}; canvas sRGB checkpoint verified (${canvasFormat})`
    : baseResult;
}

/**
 * Suite 4: Variable-length multi-mesh production batch scene verification suite.
 *
 * Verification Requirements:
 * 1. Variable-length multi-mesh production path via prepareMeshBatchPacket / renderMeshBatch.
 * 2. Actual two overlapping meshes with DISTINCT near/far colors:
 *    - Near quad at z = -2.0, Green [0, 1, 0, 1]
 *    - Far quad at z = -4.0, Red [1, 0, 0, 1]
 * 3. Proves depth writes:
 *    - Checkpoint 1: Near-first, far-second with depthWrite = true -> near color wins (Green occluding Red).
 *      Matches independent direct WebGPU reference bit-for-bit.
 *    - Checkpoint 2: Near-first, far-second with depthWrite = false -> far color overwrites near (Red).
 *      Matches independent direct WebGPU reference bit-for-bit, and strictly diverges from Checkpoint 1 (Green vs Red).
 *    - Checkpoint 3: Far-first, near-second with depthWrite = true -> near color overwrites far (Green).
 *      Matches independent direct WebGPU reference bit-for-bit.
 * 4. Checkpoint 4: Proves immutable snapshots with distinct dynamic transforms and colors.
 * 5. Checkpoint 5: Negative controls / refusal testing:
 *    - 5a: Refusal of empty mesh batch (EMPTY_MESH_BATCH).
 *    - 5b: Legacy-export refusal of mixed depth settings (INCOMPATIBLE_BATCH_DEPTH).
 *    - 5c: Refusal of inadmissible mesh in batch with exact index.
 * 6. Checkpoint 6: Retained WebGLRenderer multi-mesh oracle parity.
 * 7. Checkpoint 7: Visible canvas multi-mesh batch presentation (when canvasContext is provided).
 *
 * @param {WebGpuBridgeHost} bridgeHost
 * @param {object} wasmExports
 * @param {GPUCanvasContext} [canvasContext]
 * @param {HTMLCanvasElement} [canvas]
 * @returns {Promise<string>}
 */
export async function testMultiMeshBatchScene(bridgeHost, wasmExports, canvasContext = null, canvas = null) {
  const buildBatchFn =
    wasmExports?.f3d_build_mesh_batch_packet ||
    wasmExports?.gpu_bridge_build_mesh_batch_packet;

  if (typeof buildBatchFn !== "function") {
    throw new Error(
      "Missing required Wasm mesh batch export: f3d_build_mesh_batch_packet / gpu_bridge_build_mesh_batch_packet"
    );
  }

  const THREE = await loadProductionThree();
  const adapter = await loadProductionAdapter();

  if (typeof adapter.prepareMeshBatchPacket !== "function") {
    throw new Error("Missing required prepareMeshBatchPacket in production mesh_adapter.mjs");
  }
  if (typeof adapter.renderMeshBatch !== "function") {
    throw new Error("Missing required renderMeshBatch in production mesh_adapter.mjs");
  }
  if (typeof adapter.renderScene !== "function") {
    throw new Error("Missing required renderScene in production mesh_adapter.mjs");
  }

  const device = bridgeHost.device;
  const width = 64;
  const height = 64;
  const bytesPerRow = Math.ceil((width * 4) / 256) * 256;
  const centerIdx = 32 * bytesPerRow + 32 * 4;

  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
  camera.position.set(0, 0, 0);
  camera.lookAt(0, 0, -1);
  camera.updateMatrixWorld(true);
  camera.updateProjectionMatrix();

  // Near triangle: z = -2.0, Green (unindexed 3 vertices, strictly enclosing center in solid interior)
  const geomNear = new THREE.BufferGeometry();
  const posNear = new Float32Array([
    -2.0, -1.5, -2.0,
     2.0, -1.5, -2.0,
     0.0,  2.0, -2.0,
  ]);
  geomNear.setAttribute("position", new THREE.BufferAttribute(posNear, 3));

  const matNear = new THREE.MeshBasicMaterial({
    color: 0x00ff00, // Green [0, 1, 0, 1]
    side: THREE.DoubleSide,
    depthTest: true,
    depthWrite: true,
    depthFunc: THREE.LessDepth,
  });
  const meshNear = new THREE.Mesh(geomNear, matNear);
  meshNear.updateMatrixWorld(true);

  // Far triangle: z = -4.0, Red (indexed 3 vertices, strictly enclosing center in solid interior)
  const geomFar = new THREE.BufferGeometry();
  const posFar = new Float32Array([
    -3.0, -2.5, -4.0,
     3.0, -2.5, -4.0,
     0.0,  3.5, -4.0,
  ]);
  geomFar.setAttribute("position", new THREE.BufferAttribute(posFar, 3));
  geomFar.setIndex([0, 1, 2]);

  const matFar = new THREE.MeshBasicMaterial({
    color: 0xff0000, // Red [1, 0, 0, 1]
    side: THREE.DoubleSide,
    depthTest: true,
    depthWrite: true,
    depthFunc: THREE.LessDepth,
  });
  const meshFar = new THREE.Mesh(geomFar, matFar);
  meshFar.updateMatrixWorld(true);

  // ---------------------------------------------------------------------------
  // Checkpoint 1: Near-first, far-second with depthWrite = true -> Green
  // ---------------------------------------------------------------------------
  meshNear.material.depthWrite = true;
  meshFar.material.depthWrite = true;
  meshNear.material.needsUpdate = true;
  meshFar.material.needsUpdate = true;

  await adapter.renderMeshBatch(
    bridgeHost,
    [meshNear, meshFar],
    camera,
    null,
    wasmExports,
    { width, height }
  );
  const candidate1 = await bridgeHost.readbackBuffer(20, bytesPerRow * height);

  const ref1 = await directMeshReference(
    device,
    buildIndependentBatchReferenceInput(
      [meshNear, meshFar],
      camera,
      width,
      height,
      {
        hasDepth: true,
        depthFormat: "depth24plus",
        depthWriteEnabled: true,
        depthCompare: "less",
      }
    )
  );

  // Assert demonstrably covered non-clear pixel (must not pass from agreeing on clear pixels)
  if (candidate1[centerIdx] === 0 && candidate1[centerIdx + 1] === 0 && candidate1[centerIdx + 2] === 0) {
    throw new Error(
      "Checkpoint 1 failed: Candidate center pixel is clear color (black); triangles must demonstrably cover the sample"
    );
  }
  if (ref1[centerIdx] === 0 && ref1[centerIdx + 1] === 0 && ref1[centerIdx + 2] === 0) {
    throw new Error(
      "Checkpoint 1 failed: Reference center pixel is clear color (black); triangles must demonstrably cover the sample"
    );
  }

  if (differs(candidate1, ref1)) {
    throw new Error(
      "Checkpoint 1 failed: Near-first, far-second with depthWrite=true differs from independent direct WebGPU reference"
    );
  }

  if (candidate1[centerIdx + 1] < 240 || candidate1[centerIdx] > 15) {
    throw new Error(
      `Checkpoint 1 failed: Expected Green center pixel for depthWrite=true occlusion, got [${candidate1[centerIdx]}, ${candidate1[centerIdx + 1]}, ${candidate1[centerIdx + 2]}]`
    );
  }

  // ---------------------------------------------------------------------------
  // Checkpoint 2: Near-first, far-second with depthWrite = false -> Red
  // ---------------------------------------------------------------------------
  meshNear.material.depthWrite = false;
  meshFar.material.depthWrite = false;
  meshNear.material.needsUpdate = true;
  meshFar.material.needsUpdate = true;

  await adapter.renderMeshBatch(
    bridgeHost,
    [meshNear, meshFar],
    camera,
    null,
    wasmExports,
    { width, height }
  );
  const candidate2 = await bridgeHost.readbackBuffer(20, bytesPerRow * height);

  const ref2 = await directMeshReference(
    device,
    buildIndependentBatchReferenceInput(
      [meshNear, meshFar],
      camera,
      width,
      height,
      {
        hasDepth: true,
        depthFormat: "depth24plus",
        depthWriteEnabled: false,
        depthCompare: "less",
      }
    )
  );

  // Assert demonstrably covered non-clear pixel (must not pass from agreeing on clear pixels)
  if (candidate2[centerIdx] === 0 && candidate2[centerIdx + 1] === 0 && candidate2[centerIdx + 2] === 0) {
    throw new Error(
      "Checkpoint 2 failed: Candidate center pixel is clear color (black); triangles must demonstrably cover the sample"
    );
  }
  if (ref2[centerIdx] === 0 && ref2[centerIdx + 1] === 0 && ref2[centerIdx + 2] === 0) {
    throw new Error(
      "Checkpoint 2 failed: Reference center pixel is clear color (black); triangles must demonstrably cover the sample"
    );
  }

  if (differs(candidate2, ref2)) {
    throw new Error(
      "Checkpoint 2 failed: Near-first, far-second with depthWrite=false differs from independent direct WebGPU reference"
    );
  }

  if (candidate2[centerIdx] < 240 || candidate2[centerIdx + 1] > 15) {
    throw new Error(
      `Checkpoint 2 failed: Expected Red center pixel for depthWrite=false overwrite, got [${candidate2[centerIdx]}, ${candidate2[centerIdx + 1]}, ${candidate2[centerIdx + 2]}]`
    );
  }

  // Explicit proof of depth write: candidate1 (Green) and candidate2 (Red) MUST strictly differ!
  if (!differs(candidate1, candidate2)) {
    throw new Error(
      "Checkpoint 2 failed: depthWrite=false output must strictly diverge from depthWrite=true output to prove depth writes"
    );
  }

  // Planted negative control: depthWrite=true must strictly reject planted wrong-write (depthWrite=false, Red) output
  if (!differs(candidate1, ref2)) {
    throw new Error(
      "Planted negative failed: depthWrite=true candidate output falsely matched planted wrong-write (depthWrite=false) reference"
    );
  }
  const plantedDiffG1 = Math.abs(candidate1[centerIdx + 1] - ref2[centerIdx + 1]);
  const plantedDiffR1 = Math.abs(candidate1[centerIdx] - ref2[centerIdx]);
  if (plantedDiffG1 < 200 || plantedDiffR1 < 200) {
    throw new Error(
      `Planted negative failed: depthWrite=true did not strictly diverge from planted wrong-write reference (diffG=${plantedDiffG1}, diffR=${plantedDiffR1})`
    );
  }

  // Planted negative control: depthWrite=false must strictly reject planted wrong-write (depthWrite=true, Green) output
  if (!differs(candidate2, ref1)) {
    throw new Error(
      "Planted negative failed: depthWrite=false candidate output falsely matched planted wrong-write (depthWrite=true) reference"
    );
  }
  const plantedDiffG2 = Math.abs(candidate2[centerIdx + 1] - ref1[centerIdx + 1]);
  const plantedDiffR2 = Math.abs(candidate2[centerIdx] - ref1[centerIdx]);
  if (plantedDiffG2 < 200 || plantedDiffR2 < 200) {
    throw new Error(
      `Planted negative failed: depthWrite=false did not strictly diverge from planted wrong-write reference (diffG=${plantedDiffG2}, diffR=${plantedDiffR2})`
    );
  }

  // ---------------------------------------------------------------------------
  // Checkpoint 3: Far-first, near-second with depthWrite = true -> Green
  // ---------------------------------------------------------------------------
  meshNear.material.depthWrite = true;
  meshFar.material.depthWrite = true;
  meshNear.material.needsUpdate = true;
  meshFar.material.needsUpdate = true;

  await adapter.renderMeshBatch(
    bridgeHost,
    [meshFar, meshNear],
    camera,
    null,
    wasmExports,
    { width, height }
  );
  const candidate3 = await bridgeHost.readbackBuffer(20, bytesPerRow * height);

  const ref3 = await directMeshReference(
    device,
    buildIndependentBatchReferenceInput(
      [meshFar, meshNear],
      camera,
      width,
      height,
      {
        hasDepth: true,
        depthFormat: "depth24plus",
        depthWriteEnabled: true,
        depthCompare: "less",
      }
    )
  );

  // Assert demonstrably covered non-clear pixel
  if (candidate3[centerIdx] === 0 && candidate3[centerIdx + 1] === 0 && candidate3[centerIdx + 2] === 0) {
    throw new Error(
      "Checkpoint 3 failed: Candidate center pixel is clear color (black); triangles must demonstrably cover the sample"
    );
  }

  if (differs(candidate3, ref3)) {
    throw new Error(
      "Checkpoint 3 failed: Far-first, near-second with depthWrite=true differs from independent direct WebGPU reference"
    );
  }

  if (candidate3[centerIdx + 1] < 240 || candidate3[centerIdx] > 15) {
    throw new Error(
      `Checkpoint 3 failed: Expected Green center pixel for far-first/near-second depth test, got [${candidate3[centerIdx]}, ${candidate3[centerIdx + 1]}, ${candidate3[centerIdx + 2]}]`
    );
  }

  // ---------------------------------------------------------------------------
  // Checkpoint 4: Distinct Dynamic Transforms, Immutable Snapshots, & GPU Output
  // ---------------------------------------------------------------------------
  // Shared centered triangle in local space: [-0.4, -0.4, 0], [0.4, -0.4, 0], [0, 0.4, 0]
  // In camera frustum (fov 45, aspect 1, z=-3 gives half-width ~1.2426):
  // meshA at (-0.5, 0, -3) maps interior center to pixel (x=19, y=32).
  // meshB at (+0.5, 0, -3) maps interior center to pixel (x=45, y=32).
  // Midpoint between them (x=32, y=32) remains clear background.
  const geomShared = new THREE.BufferGeometry();
  geomShared.setAttribute("position", new THREE.BufferAttribute(new Float32Array([
    -0.4, -0.4, 0.0,
     0.4, -0.4, 0.0,
     0.0,  0.4, 0.0,
  ]), 3));

  const matA = new THREE.MeshBasicMaterial({
    color: 0x0000ff, // Blue [0, 0, 1, 1]
    side: THREE.DoubleSide,
    depthTest: true,
    depthWrite: true,
    depthFunc: THREE.LessDepth,
  });
  const meshA = new THREE.Mesh(geomShared, matA);
  meshA.position.set(-0.5, 0.0, -3.0);
  meshA.updateMatrixWorld(true);

  const matB = new THREE.MeshBasicMaterial({
    color: 0xffff00, // Yellow [1, 1, 0, 1]
    side: THREE.DoubleSide,
    depthTest: true,
    depthWrite: true,
    depthFunc: THREE.LessDepth,
  });
  const meshB = new THREE.Mesh(geomShared, matB);
  meshB.position.set(0.5, 0.0, -3.0);
  meshB.updateMatrixWorld(true);

  // 1. Build INDEPENDENT reference inputs BEFORE scene mutation
  const independentRef4 = await directMeshReference(
    device,
    buildIndependentBatchReferenceInput(
      [meshA, meshB],
      camera,
      width,
      height,
      {
        hasDepth: true,
        depthFormat: "depth24plus",
        depthWriteEnabled: true,
        depthCompare: "less",
      }
    )
  );

  // 2. Prepare candidate batch packet (extracts snapshots with distinct modelViews and colors)
  const batchPrep4 = adapter.prepareMeshBatchPacket(
    [meshA, meshB],
    camera,
    width,
    height,
    wasmExports
  );

  if (batchPrep4.snapshots.length !== 2) {
    throw new Error(`Checkpoint 4 failed: Expected 2 snapshots, got ${batchPrep4.snapshots.length}`);
  }
  const snapA = batchPrep4.snapshots[0];
  const snapB = batchPrep4.snapshots[1];

  // Verify distinct transforms in snapshot records (m12 = model-view x translation)
  if (Math.abs(snapA.modelView[12] - (-0.5)) > 0.01) {
    throw new Error(`Checkpoint 4 failed: Snapshot A modelView x is not -0.5, got ${snapA.modelView[12]}`);
  }
  if (Math.abs(snapB.modelView[12] - 0.5) > 0.01) {
    throw new Error(`Checkpoint 4 failed: Snapshot B modelView x is not +0.5, got ${snapB.modelView[12]}`);
  }
  if (Math.abs(snapA.modelView[12] - snapB.modelView[12]) < 0.5) {
    throw new Error("Checkpoint 4 failed: Snapshots A and B do not have distinct model-view transforms");
  }

  // Verify distinct colors in snapshot records
  if (snapA.color[2] < 0.9 || snapA.color[0] > 0.1) {
    throw new Error("Checkpoint 4 failed: Snapshot A color is not Blue");
  }
  if (snapB.color[0] < 0.9 || snapB.color[1] < 0.9) {
    throw new Error("Checkpoint 4 failed: Snapshot B color is not Yellow");
  }

  // 3. Mutate scene transforms and colors BEFORE executing packet to catch clobber/stale reread
  meshA.position.set(100.0, 100.0, 100.0);
  meshA.material.color.setHex(0x000000);
  meshA.updateMatrixWorld(true);

  meshB.position.set(-100.0, -100.0, -100.0);
  meshB.material.color.setHex(0xffffff);
  meshB.updateMatrixWorld(true);

  // 4. Execute packet on GPU and read back full candidate pixels
  await bridgeHost.executePacket(batchPrep4.packetBytes);
  const candidate4 = await bridgeHost.readbackBuffer(20, bytesPerRow * height);

  // 5. Compare full candidate pixels against independent direct WebGPU reference
  if (differs(candidate4, independentRef4)) {
    throw new Error(
      "Checkpoint 4 failed: Multi-mesh candidate output with distinct transforms differs from independent direct WebGPU reference"
    );
  }

  // 6. Assert known covered interior pixels at y=32:
  // Left interior pixel (Mesh A, x=19, y=32)
  const leftIdx = 32 * bytesPerRow + 19 * 4;
  // Right interior pixel (Mesh B, x=45, y=32)
  const rightIdx = 32 * bytesPerRow + 45 * 4;
  // Midpoint pixel between separated triangles (x=32, y=32)
  const midIdx = 32 * bytesPerRow + 32 * 4;

  // Left pixel must be Blue (R <= 15, G <= 15, B >= 240)
  if (candidate4[leftIdx + 2] < 240 || candidate4[leftIdx] > 15 || candidate4[leftIdx + 1] > 15) {
    throw new Error(
      `Checkpoint 4 failed: Expected Blue interior pixel for Mesh A at (19,32), got [${candidate4[leftIdx]}, ${candidate4[leftIdx + 1]}, ${candidate4[leftIdx + 2]}]`
    );
  }

  // Right pixel must be Yellow (R >= 240, G >= 240, B <= 15)
  if (candidate4[rightIdx] < 240 || candidate4[rightIdx + 1] < 240 || candidate4[rightIdx + 2] > 15) {
    throw new Error(
      `Checkpoint 4 failed: Expected Yellow interior pixel for Mesh B at (45,32), got [${candidate4[rightIdx]}, ${candidate4[rightIdx + 1]}, ${candidate4[rightIdx + 2]}]`
    );
  }

  // Midpoint between triangles must be clear background (black [0, 0, 0])
  if (candidate4[midIdx] !== 0 || candidate4[midIdx + 1] !== 0 || candidate4[midIdx + 2] !== 0) {
    throw new Error(
      `Checkpoint 4 failed: Expected clear background between separated triangles at (32,32), got [${candidate4[midIdx]}, ${candidate4[midIdx + 1]}, ${candidate4[midIdx + 2]}]`
    );
  }

  // 7. Uniform aliasing / transform clobber guards:
  // Left pixel must strictly not be yellow, and right pixel must strictly not be blue
  if (candidate4[leftIdx] > 50 || candidate4[leftIdx + 1] > 50) {
    throw new Error("Checkpoint 4 failed: Left pixel shows uniform aliasing with Mesh B (Yellow clobbered Mesh A)");
  }
  if (candidate4[rightIdx + 2] > 50) {
    throw new Error("Checkpoint 4 failed: Right pixel shows uniform aliasing with Mesh A (Blue clobbered Mesh B)");
  }

  // 8. Assert snapshots remained immutable despite post-prepare scene mutations
  if (snapA.color[2] < 0.9 || snapB.color[0] < 0.9) {
    throw new Error("Checkpoint 4 failed: Post-prepare scene mutation corrupted previous snapshot records");
  }

  // ---------------------------------------------------------------------------
  // Checkpoint 5: Negative controls / refusal testing
  // ---------------------------------------------------------------------------
  // 5a: Refusal of empty mesh batch
  let emptyRejected = false;
  try {
    adapter.prepareMeshBatchPacket([], camera, width, height, wasmExports);
  } catch (err) {
    if (
      err.reason === "EMPTY_MESH_BATCH" ||
      err.message.includes("EMPTY_MESH_BATCH") ||
      err.message.includes(adapter.ADMISSION_REJECTION?.EMPTY_MESH_BATCH) ||
      err.message.includes("non-empty array")
    ) {
      emptyRejected = true;
    }
  }
  if (!emptyRejected) {
    throw new Error("Checkpoint 5a failed: Empty mesh batch was not rejected with EMPTY_MESH_BATCH");
  }

  // 5b: Ambiguous settings still refuse; older exports still refuse mixed depth.
  // These are real legacy functions, with the new capability deliberately absent.
  const legacyDepthExports = {
    f3d_build_mesh_batch_packet: wasmExports.f3d_build_mesh_batch_packet,
    f3d_build_mesh_batch_cull_packet: wasmExports.f3d_build_mesh_batch_cull_packet,
  };
  const meshIncompatA = meshNear.clone();
  meshIncompatA.material = matNear.clone();
  meshIncompatA.material.depthTest = true;

  const meshIncompatB = meshFar.clone();
  meshIncompatB.material = matFar.clone();
  meshIncompatB.material.depthTest = false;

  // 5b-1: Refusal of ambiguous depthTest=false + depthWrite=true without sourceBackend
  let ambiguityRejected = false;
  try {
    adapter.prepareMeshBatchPacket([meshIncompatA, meshIncompatB], camera, width, height, wasmExports);
  } catch (err) {
    if (
      err.reason === "AMBIGUOUS_DEPTH_PAIR" ||
      err.message.includes("AMBIGUOUS_DEPTH_PAIR") ||
      err.message.includes(adapter.ADMISSION_REJECTION?.AMBIGUOUS_DEPTH_PAIR) ||
      err.message.includes("ambiguous across backends")
    ) {
      ambiguityRejected = true;
    }
  }
  if (!ambiguityRejected) {
    throw new Error("Checkpoint 5b failed: Ambiguous depthTest=false + depthWrite=true without sourceBackend was not rejected with AMBIGUOUS_DEPTH_PAIR");
  }

  // 5b-2: Refusal of incompatible depthTest with explicit sourceBackend: 'webgpu'
  let depthTestMismatchRejected = false;
  try {
    adapter.prepareMeshBatchPacket([meshIncompatA, meshIncompatB], camera, width, height, legacyDepthExports, { sourceBackend: "webgpu" });
  } catch (err) {
    const isDepthReason =
      err.reason === "INCOMPATIBLE_BATCH_DEPTH" ||
      err.message.includes("INCOMPATIBLE_BATCH_DEPTH") ||
      err.message.includes(adapter.ADMISSION_REJECTION?.INCOMPATIBLE_BATCH_DEPTH) ||
      err.message.includes("incompatible depth settings");
    if (isDepthReason) {
      depthTestMismatchRejected = true;
    }
  }
  if (!depthTestMismatchRejected) {
    throw new Error("Checkpoint 5b failed: Mismatched depthTest was not rejected with INCOMPATIBLE_BATCH_DEPTH");
  }

  meshIncompatB.material.depthTest = true;
  meshIncompatB.material.depthWrite = false;
  let depthWriteMismatchRejected = false;
  try {
    adapter.prepareMeshBatchPacket([meshIncompatA, meshIncompatB], camera, width, height, legacyDepthExports);
  } catch (err) {
    const isDepthReason =
      err.reason === "INCOMPATIBLE_BATCH_DEPTH" ||
      err.message.includes("INCOMPATIBLE_BATCH_DEPTH") ||
      err.message.includes(adapter.ADMISSION_REJECTION?.INCOMPATIBLE_BATCH_DEPTH) ||
      err.message.includes("incompatible depth settings");
    if (isDepthReason) {
      depthWriteMismatchRejected = true;
    }
  }
  if (!depthWriteMismatchRejected) {
    throw new Error("Checkpoint 5b failed: Mismatched depthWrite was not rejected with INCOMPATIBLE_BATCH_DEPTH");
  }

  // 5b-3: Real wasmExports with f3d_build_mesh_batch_cull_depth_packet admits mixed depth
  let newExportAdmitted = false;
  try {
    const admittedBatch = adapter.prepareMeshBatchPacket(
      [meshIncompatA, meshIncompatB],
      camera,
      width,
      height,
      wasmExports,
      { sourceBackend: "webgpu" }
    );
    if (admittedBatch && admittedBatch.packetBytes && admittedBatch.packetBytes.length > 0) {
      newExportAdmitted = true;
    }
  } catch (err) {
    newExportAdmitted = false;
  }
  if (!newExportAdmitted) {
    throw new Error("Checkpoint 5b failed: Real wasmExports with cull_depth export must admit mixed depth settings");
  }

  // 5c: Refusal of inadmissible mesh in batch with exact index
  const meshInvisible = meshFar.clone();
  meshInvisible.visible = false;
  let invisibleRejected = false;
  try {
    adapter.prepareMeshBatchPacket([meshNear, meshInvisible], camera, width, height, wasmExports);
  } catch (err) {
    const isNotVisible =
      err.reason === "NOT_VISIBLE" ||
      err.message.includes("NOT_VISIBLE") ||
      err.message.includes(adapter.ADMISSION_REJECTION?.NOT_VISIBLE) ||
      err.message.includes("visible = false");
    if (err.message.includes("Mesh batch admission rejected at index 1") && isNotVisible) {
      invisibleRejected = true;
    }
  }
  if (!invisibleRejected) {
    throw new Error("Checkpoint 5c failed: Inadmissible invisible mesh was not rejected with exact index 1 NOT_VISIBLE");
  }

  // 5d: Refusal of colorWrite=false on older exports lacking f3d_build_mesh_batch_cull_depth_color_packet
  const legacyColorExports = {
    f3d_build_mesh_batch_packet: wasmExports.f3d_build_mesh_batch_packet,
    f3d_build_mesh_batch_cull_packet: wasmExports.f3d_build_mesh_batch_cull_packet,
    f3d_build_mesh_batch_cull_depth_packet: wasmExports.f3d_build_mesh_batch_cull_depth_packet,
  };
  const meshOccluder5d = meshNear.clone();
  meshOccluder5d.material = matNear.clone();
  meshOccluder5d.material.colorWrite = false;
  let colorWriteLegacyRejected = false;
  try {
    adapter.prepareMeshBatchPacket([meshOccluder5d], camera, width, height, legacyColorExports);
  } catch (err) {
    if (
      err.reason === "INCOMPATIBLE_COLOR_WRITE" ||
      err.message.includes("INCOMPATIBLE_COLOR_WRITE") ||
      err.message.includes(adapter.ADMISSION_REJECTION?.INCOMPATIBLE_COLOR_WRITE)
    ) {
      colorWriteLegacyRejected = true;
    }
  }
  if (!colorWriteLegacyRejected) {
    throw new Error("Checkpoint 5d failed: colorWrite=false was not rejected with INCOMPATIBLE_COLOR_WRITE on legacy exports");
  }

  // 5d-2: Real wasmExports with f3d_build_mesh_batch_cull_depth_color_packet admits colorWrite=false
  let colorWriteAdmitted = false;
  try {
    const admittedColorBatch = adapter.prepareMeshBatchPacket(
      [meshOccluder5d],
      camera,
      width,
      height,
      wasmExports
    );
    if (admittedColorBatch && admittedColorBatch.packetBytes && admittedColorBatch.packetBytes.length > 0) {
      colorWriteAdmitted = true;
    }
  } catch (err) {
    colorWriteAdmitted = false;
  }
  if (!colorWriteAdmitted) {
    throw new Error("Checkpoint 5d failed: Real wasmExports with cull_depth_color export must admit colorWrite=false");
  }

  // ---------------------------------------------------------------------------
  // Checkpoint 6: Retained WebGLRenderer multi-mesh oracle parity
  // ---------------------------------------------------------------------------
  meshNear.material.depthWrite = true;
  meshFar.material.depthWrite = true;
  meshNear.material.needsUpdate = true;
  meshFar.material.needsUpdate = true;

  const retainedOracle = renderRetainedWebGLReference(
    THREE,
    [meshNear, meshFar],
    camera,
    width,
    height
  );

  // Assert demonstrably covered non-clear pixel from retained WebGL oracle
  if (retainedOracle[0] === 0 && retainedOracle[1] === 0 && retainedOracle[2] === 0) {
    throw new Error(
      `Checkpoint 6 failed: Retained WebGLRenderer oracle sampled clear color (black); triangles must demonstrably cover center`
    );
  }

  if (retainedOracle[1] < 240 || retainedOracle[0] > 15) {
    throw new Error(
      `Checkpoint 6 failed: Retained WebGLRenderer oracle center is not Green, got [${retainedOracle[0]}, ${retainedOracle[1]}, ${retainedOracle[2]}]`
    );
  }

  // Planted negative control: Retained oracle (Green) must strictly reject planted wrong-write (Red, ref2)
  const oracleWrongDiff = Math.abs(retainedOracle[1] - ref2[centerIdx + 1]);
  if (oracleWrongDiff < 200) {
    throw new Error(
      `Planted negative failed: Retained WebGLRenderer oracle falsely matched planted wrong-write reference (diffG=${oracleWrongDiff})`
    );
  }

  const oracleDiffG = Math.abs(candidate1[centerIdx + 1] - retainedOracle[1]);
  const oracleDiffR = Math.abs(candidate1[centerIdx] - retainedOracle[0]);
  if (oracleDiffG > 2 || oracleDiffR > 2) {
    throw new Error(
      `Checkpoint 6 failed: Candidate center pixel [${candidate1[centerIdx]}, ${candidate1[centerIdx + 1]}] differs from retained WebGLRenderer oracle [${retainedOracle[0]}, ${retainedOracle[1]}] by more than tolerance 2`
    );
  }

  // ---------------------------------------------------------------------------
  // Checkpoint 7: Visible canvas presentation (when canvasContext is provided)
  // ---------------------------------------------------------------------------
  if (canvasContext) {
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();

    canvasContext.configure({
      device,
      format: canvasFormat,
      alphaMode: "opaque",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    await nextFrame();

    const canvasReadback = device.createBuffer({
      size: bytesPerRow * height,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const currentTexture = canvasContext.getCurrentTexture();
    const execPromise = adapter.renderMeshBatch(
      bridgeHost,
      [meshNear, meshFar],
      camera,
      canvasContext,
      wasmExports
    );

    const copyEncoder = device.createCommandEncoder();
    copyEncoder.copyTextureToBuffer(
      { texture: currentTexture },
      { buffer: canvasReadback, bytesPerRow },
      [width, height, 1]
    );
    device.queue.submit([copyEncoder.finish()]);

    await execPromise;
    await canvasReadback.mapAsync(GPUMapMode.READ);
    const canvasPixels = new Uint8Array(canvasReadback.getMappedRange().slice(0));
    canvasReadback.unmap();
    canvasReadback.destroy();

    const refCanvas = await directMeshReference(
      device,
      buildIndependentBatchReferenceInput(
        [meshNear, meshFar],
        camera,
        width,
        height,
        {
          hasDepth: true,
          depthFormat: "depth24plus",
          depthWriteEnabled: true,
          depthCompare: "less",
          format: canvasFormat,
          outputSrgb: true,
        }
      )
    );

    if (differs(canvasPixels, refCanvas)) {
      const diffs = [];
      let maxDiff = 0;
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const idx = y * bytesPerRow + x * 4;
          const c = [canvasPixels[idx], canvasPixels[idx + 1], canvasPixels[idx + 2], canvasPixels[idx + 3]];
          const r = [refCanvas[idx], refCanvas[idx + 1], refCanvas[idx + 2], refCanvas[idx + 3]];
          const dR = Math.abs(c[0] - r[0]);
          const dG = Math.abs(c[1] - r[1]);
          const dB = Math.abs(c[2] - r[2]);
          const dA = Math.abs(c[3] - r[3]);
          const d = Math.max(dR, dG, dB, dA);
          if (d > 0) {
            if (d > maxDiff) maxDiff = d;
            if (diffs.length < 8) {
              diffs.push(`(${x},${y}): cand=[${c}] ref=[${r}] diff=[${dR},${dG},${dB},${dA}]`);
            }
          }
        }
      }
      throw new Error(
        `Checkpoint 7 failed: Visible canvas multi-mesh batch pixels differ from direct WebGPU reference in canvas format (${canvasFormat}). ` +
        `Matched settings: depthFormat=depth24plus, depthCompare=less, depthWrite=true, outputSrgb=true. ` +
        `Max component diff: ${maxDiff}. Sample differing pixels: ${diffs.join("; ")}`
      );
    }
  }

  // ---------------------------------------------------------------------------
  // Checkpoint 8: Scene-level hierarchy, admission filtering, and canvas execution via renderScene
  // ---------------------------------------------------------------------------
  // Ruby's renderScene (mesh_adapter.mjs:1050+):
  // - A THREE.Scene with a translated THREE.Group (position: 0.2, -0.1, -3.0)
  // - Group contains 3 MeshBasicMaterial meshes at distinct positions/colors:
  //     Mesh 1: Red (Near, z=-2.5 in world space)
  //     Mesh 2: Green (Far, z=-3.5 in world space)
  //     Mesh 3: Blue (Distinct, z=-3.0 in world space)
  // - Group also contains:
  //     1 InstancedMesh (must be refused with UNSUPPORTED_MESH_SUBCLASS)
  //     1 invisible mesh (visible=false, must be refused with NOT_VISIBLE)
  // - Asserts renderScene returns admitted=3 and refused=2 carrying reason codes
  // - Reads back real canvas pixels
  // - Asserts each mesh's expected sRGB color at its projected center (tolerance 2)
  // - Asserts correct occlusion where Mesh 1 and Mesh 2 overlap (depth24plus)
  // - Asserts a planted negative (wrong expected color must strictly fail)

  const scene8 = new THREE.Scene();
  const group8 = new THREE.Group();
  group8.position.set(0.2, -0.1, -3.0);
  scene8.add(group8);

  // Mesh 1: Red, Near (local: -0.3, 0.2, 0.5 -> world: -0.1, 0.1, -2.5)
  const geom8A = new THREE.BufferGeometry();
  geom8A.setAttribute("position", new THREE.BufferAttribute(new Float32Array([
    -0.3, -0.2, 0.0,
     0.3, -0.2, 0.0,
     0.0,  0.3, 0.0,
  ]), 3));
  const mat8A = new THREE.MeshBasicMaterial({
    color: 0xff0000, // Red [1, 0, 0, 1]
    side: THREE.DoubleSide,
    depthTest: true,
    depthWrite: true,
    depthFunc: THREE.LessDepth,
  });
  const mesh8A = new THREE.Mesh(geom8A, mat8A);
  mesh8A.position.set(-0.3, 0.2, 0.5);
  group8.add(mesh8A);

  // Mesh 2: Green, Far (local: 0.0, 0.2, -0.5 -> world: 0.2, 0.1, -3.5)
  const geom8B = new THREE.BufferGeometry();
  geom8B.setAttribute("position", new THREE.BufferAttribute(new Float32Array([
    -0.3, -0.2, 0.0,
     0.3, -0.2, 0.0,
     0.0,  0.3, 0.0,
  ]), 3));
  const mat8B = new THREE.MeshBasicMaterial({
    color: 0x00ff00, // Green [0, 1, 0, 1]
    side: THREE.DoubleSide,
    depthTest: true,
    depthWrite: true,
    depthFunc: THREE.LessDepth,
  });
  const mesh8B = new THREE.Mesh(geom8B, mat8B);
  mesh8B.position.set(0.0, 0.2, -0.5);
  group8.add(mesh8B);

  // Mesh 3: Blue, Distinct (local: 0.2, -0.3, 0.0 -> world: 0.4, -0.4, -3.0)
  const geom8C = new THREE.BufferGeometry();
  geom8C.setAttribute("position", new THREE.BufferAttribute(new Float32Array([
    -0.25, -0.25, 0.0,
     0.25, -0.25, 0.0,
     0.0,   0.25, 0.0,
  ]), 3));
  const mat8C = new THREE.MeshBasicMaterial({
    color: 0x0000ff, // Blue [0, 0, 1, 1]
    side: THREE.DoubleSide,
    depthTest: true,
    depthWrite: true,
    depthFunc: THREE.LessDepth,
  });
  const mesh8C = new THREE.Mesh(geom8C, mat8C);
  mesh8C.position.set(0.2, -0.3, 0.0);
  group8.add(mesh8C);

  // Legitimately culled 1: Invisible mesh (visible=false -> culling parity)
  const invisGeom8 = new THREE.BufferGeometry();
  invisGeom8.setAttribute("position", new THREE.BufferAttribute(new Float32Array([
    -0.1, -0.1, 0.0,
     0.1, -0.1, 0.0,
     0.0,  0.1, 0.0,
  ]), 3));
  const invisMat8 = new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide });
  const invisibleMesh8 = new THREE.Mesh(invisGeom8, invisMat8);
  invisibleMesh8.visible = false;
  group8.add(invisibleMesh8);

  // Legitimately culled 2: Layer-filtered mesh (camera is layer 0, mesh is layer 2)
  const layerGeom8 = new THREE.BufferGeometry();
  layerGeom8.setAttribute("position", new THREE.BufferAttribute(new Float32Array([
    -0.1, -0.1, 0.0,
     0.1, -0.1, 0.0,
     0.0,  0.1, 0.0,
  ]), 3));
  const layerMat8 = new THREE.MeshBasicMaterial({ color: 0xffff00, side: THREE.DoubleSide });
  const layerCulledMesh8 = new THREE.Mesh(layerGeom8, layerMat8);
  layerCulledMesh8.layers.set(2);
  group8.add(layerCulledMesh8);

  let activeCanvasContext8 = canvasContext;
  if (!activeCanvasContext8 && typeof OffscreenCanvas !== "undefined") {
    try {
      const offCanvas8 = new OffscreenCanvas(width, height);
      activeCanvasContext8 = offCanvas8.getContext("webgpu");
    } catch (_) {}
  }

  let sceneResult8;
  let canvasPixels8 = null;

  if (activeCanvasContext8) {
    const canvasFormat8 = navigator.gpu.getPreferredCanvasFormat();
    activeCanvasContext8.configure({
      device,
      format: canvasFormat8,
      alphaMode: "opaque",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    await nextFrame();

    const canvasReadback8 = device.createBuffer({
      size: bytesPerRow * height,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const currentTexture8 = activeCanvasContext8.getCurrentTexture();
    const renderScenePromise8 = adapter.renderScene(
      bridgeHost,
      scene8,
      camera,
      activeCanvasContext8,
      wasmExports,
      { width, height }
    );

    const copyEncoder8 = device.createCommandEncoder();
    copyEncoder8.copyTextureToBuffer(
      { texture: currentTexture8 },
      { buffer: canvasReadback8, bytesPerRow },
      [width, height, 1]
    );
    device.queue.submit([copyEncoder8.finish()]);

    sceneResult8 = await renderScenePromise8;
    await canvasReadback8.mapAsync(GPUMapMode.READ);
    canvasPixels8 = new Uint8Array(canvasReadback8.getMappedRange().slice(0));
    canvasReadback8.unmap();
    canvasReadback8.destroy();
  } else {
    sceneResult8 = await adapter.renderScene(
      bridgeHost,
      scene8,
      camera,
      null,
      wasmExports,
      { width, height }
    );
  }

  // 1. Assert renderScene returns admitted=3 for supported hierarchy
  if (!Array.isArray(sceneResult8.admitted) || sceneResult8.admitted.length !== 3) {
    throw new Error(
      `Checkpoint 8 failed: Expected 3 admitted meshes in scene, got ${sceneResult8.admitted?.length}`
    );
  }
  const expectedAdmittedUuids8 = [mesh8A.uuid, mesh8B.uuid, mesh8C.uuid];
  for (const uuid of expectedAdmittedUuids8) {
    if (!sceneResult8.admitted.includes(uuid)) {
      throw new Error(`Checkpoint 8 failed: Admitted list missing expected mesh UUID ${uuid}`);
    }
  }

  // 2. Assert culled meshes (invisible and layer-filtered) were legitimately ignored (refused.length === 0)
  if (!Array.isArray(sceneResult8.refused) || sceneResult8.refused.length !== 0) {
    throw new Error(
      `Checkpoint 8 failed: Expected 0 refused meshes for clean scene with culled items, got ${sceneResult8.refused?.length}`
    );
  }

  // 3. Real canvas pixel assertions: centers, occlusion, planted negative
  if (canvasPixels8) {
    const canvasFormat8 = navigator.gpu.getPreferredCanvasFormat();
    const rCh8 = canvasFormat8.startsWith("bgra") ? 2 : 0;
    const gCh8 = 1;
    const bCh8 = canvasFormat8.startsWith("bgra") ? 0 : 2;

    function getRgba8(pixels, x, y) {
      const idx = y * bytesPerRow + x * 4;
      return [
        pixels[idx + rCh8],
        pixels[idx + gCh8],
        pixels[idx + bCh8],
        pixels[idx + 3],
      ];
    }

    // Projected centers on 64x64 canvas:
    // Mesh 1 (Near Red) projected center: (28, 28)
    const c1_8 = getRgba8(canvasPixels8, 28, 28);
    // Mesh 2 (Far Green) projected center: (36, 29)
    const c2_8 = getRgba8(canvasPixels8, 36, 29);
    // Mesh 3 (Distinct Blue) projected center: (42, 42)
    const c3_8 = getRgba8(canvasPixels8, 42, 42);
    // Overlap point: (33, 31) covers both Mesh 1 (Near) and Mesh 2 (Far)
    const cOverlap8 = getRgba8(canvasPixels8, 33, 31);

    // a) Assert each mesh's expected sRGB color at its projected center (tolerance 2)
    // Mesh 1: expected sRGB Red [255, 0, 0, 255]
    if (Math.abs(c1_8[0] - 255) > 2 || c1_8[1] > 2 || c1_8[2] > 2 || Math.abs(c1_8[3] - 255) > 2) {
      throw new Error(
        `Checkpoint 8 failed: Mesh 1 projected center (28, 28) expected sRGB Red [255, 0, 0, 255] within tolerance 2, got [${c1_8}]`
      );
    }

    // Mesh 2: expected sRGB Green [0, 255, 0, 255]
    if (c2_8[0] > 2 || Math.abs(c2_8[1] - 255) > 2 || c2_8[2] > 2 || Math.abs(c2_8[3] - 255) > 2) {
      throw new Error(
        `Checkpoint 8 failed: Mesh 2 projected center (36, 29) expected sRGB Green [0, 255, 0, 255] within tolerance 2, got [${c2_8}]`
      );
    }

    // Mesh 3: expected sRGB Blue [0, 0, 255, 255]
    if (c3_8[0] > 2 || c3_8[1] > 2 || Math.abs(c3_8[2] - 255) > 2 || Math.abs(c3_8[3] - 255) > 2) {
      throw new Error(
        `Checkpoint 8 failed: Mesh 3 projected center (42, 42) expected sRGB Blue [0, 0, 255, 255] within tolerance 2, got [${c3_8}]`
      );
    }

    // b) Assert correct occlusion where two overlap (depth24plus): Near Red occludes Far Green
    if (Math.abs(cOverlap8[0] - 255) > 2 || cOverlap8[1] > 2 || cOverlap8[2] > 2 || Math.abs(cOverlap8[3] - 255) > 2) {
      throw new Error(
        `Checkpoint 8 failed: Overlap pixel (33, 31) expected Near Red [255, 0, 0, 255] (occluding Far Green) within tolerance 2, got [${cOverlap8}]`
      );
    }
    // Explicit guard: Far mesh green is occluded (green channel must strictly diverge from 255)
    if (Math.abs(cOverlap8[1] - 255) < 200) {
      throw new Error(
        `Checkpoint 8 failed: Overlap pixel (33, 31) showed Far mesh Green; occlusion by Near Red failed`
      );
    }

    // c) Planted negative control: wrong expected color must FAIL
    let plantedNegativePassed8 = false;
    const wrongExpectedColor8 = [0, 255, 0, 255]; // Green instead of Red at Mesh 1 center
    const plantedDiff8 = Math.max(
      Math.abs(c1_8[0] - wrongExpectedColor8[0]),
      Math.abs(c1_8[1] - wrongExpectedColor8[1]),
      Math.abs(c1_8[2] - wrongExpectedColor8[2])
    );
    if (plantedDiff8 <= 2) {
      throw new Error(
        `Checkpoint 8 planted negative failed: Mesh 1 Red center falsely matched wrong expected Green color [0, 255, 0, 255]`
      );
    }
    if (plantedDiff8 > 200) {
      plantedNegativePassed8 = true;
    }
    if (!plantedNegativePassed8) {
      throw new Error(
        `Checkpoint 8 planted negative failed: Wrong expected color did not strictly fail (diff=${plantedDiff8})`
      );
    }

    // Also verify against independent direct WebGPU reference
    const refSceneCanvas8 = await directMeshReference(
      device,
      buildIndependentBatchReferenceInput(
        [mesh8A, mesh8B, mesh8C],
        camera,
        width,
        height,
        {
          hasDepth: true,
          depthFormat: "depth24plus",
          depthWriteEnabled: true,
          depthCompare: "less",
          format: canvasFormat8,
          outputSrgb: true,
        }
      )
    );

    const refC1_8 = getRgba8(refSceneCanvas8, 28, 28);
    const refC2_8 = getRgba8(refSceneCanvas8, 36, 29);
    const refC3_8 = getRgba8(refSceneCanvas8, 42, 42);
    const refCOverlap8 = getRgba8(refSceneCanvas8, 33, 31);

    if (Math.abs(c1_8[0] - refC1_8[0]) > 2 || Math.abs(c1_8[1] - refC1_8[1]) > 2 || Math.abs(c1_8[2] - refC1_8[2]) > 2) {
      throw new Error(`Checkpoint 8 failed: Mesh 1 center differs from independent WebGPU reference`);
    }
    if (Math.abs(c2_8[0] - refC2_8[0]) > 2 || Math.abs(c2_8[1] - refC2_8[1]) > 2 || Math.abs(c2_8[2] - refC2_8[2]) > 2) {
      throw new Error(`Checkpoint 8 failed: Mesh 2 center differs from independent WebGPU reference`);
    }
    if (Math.abs(c3_8[0] - refC3_8[0]) > 2 || Math.abs(c3_8[1] - refC3_8[1]) > 2 || Math.abs(c3_8[2] - refC3_8[2]) > 2) {
      throw new Error(`Checkpoint 8 failed: Mesh 3 center differs from independent WebGPU reference`);
    }
    if (Math.abs(cOverlap8[0] - refCOverlap8[0]) > 2 || Math.abs(cOverlap8[1] - refCOverlap8[1]) > 2 || Math.abs(cOverlap8[2] - refCOverlap8[2]) > 2) {
      throw new Error(`Checkpoint 8 failed: Overlap pixel differs from independent WebGPU reference`);
    }
  }

  // ---------------------------------------------------------------------------
  // Checkpoint 8b: Visible-unsupported whole-scene refusal (no partial drawing per Root Mail 14355)
  // ---------------------------------------------------------------------------
  const sceneRefuse8 = new THREE.Scene();
  const groupRefuse8 = new THREE.Group();
  sceneRefuse8.add(groupRefuse8);

  // Add one valid supported mesh
  const validMesh8 = mesh8A.clone();
  validMesh8.material = mat8A.clone();
  groupRefuse8.add(validMesh8);

  // Add one visible unsupported renderable (InstancedMesh)
  const instGeom8 = new THREE.BufferGeometry();
  instGeom8.setAttribute("position", new THREE.BufferAttribute(new Float32Array([
    0, 0, 0,
    1, 0, 0,
    0, 1, 0,
  ]), 3));
  const instMat8 = new THREE.MeshBasicMaterial({ side: THREE.DoubleSide });
  const instancedMesh8 = new THREE.InstancedMesh(instGeom8, instMat8, 2);
  instancedMesh8.visible = true;
  groupRefuse8.add(instancedMesh8);

  const refuseResult8 = await adapter.renderScene(
    bridgeHost,
    sceneRefuse8,
    camera,
    activeCanvasContext8,
    wasmExports,
    { width, height }
  );

  // Must refuse WHOLE submission (admitted = 0, no partial scene rendering)
  if (!Array.isArray(refuseResult8.admitted) || refuseResult8.admitted.length !== 0) {
    throw new Error(
      `Checkpoint 8b failed: Visible unsupported content must refuse whole submission (admitted=0), got ${refuseResult8.admitted?.length}`
    );
  }

  if (!Array.isArray(refuseResult8.refused) || refuseResult8.refused.length === 0) {
    throw new Error("Checkpoint 8b failed: Expected at least 1 refused item for visible InstancedMesh");
  }

  const instancedRefusal8 = refuseResult8.refused.find(r => r.uuid === instancedMesh8.uuid);
  if (!instancedRefusal8) {
    throw new Error("Checkpoint 8b failed: Visible InstancedMesh was not recorded in refused list");
  }

  const isInstancedReason8 =
    instancedRefusal8.reason?.includes("UNSUPPORTED_MESH_SUBCLASS") ||
    instancedRefusal8.code === "UNSUPPORTED_MESH_SUBCLASS";
  if (!isInstancedReason8) {
    throw new Error(
      `Checkpoint 8b failed: InstancedMesh refusal does not carry UNSUPPORTED_MESH_SUBCLASS reason code, got: "${instancedRefusal8.reason}"`
    );
  }

  // ---------------------------------------------------------------------------
  // Checkpoint 9: Material side culling and reflected mesh winding parity
  // ---------------------------------------------------------------------------
  // 9a: FrontSide / BackSide / DoubleSide with both CCW and CW windings
  // CCW triangle: (-2.0, -1.5, -2.0), (2.0, -1.5, -2.0), (0.0, 2.0, -2.0)
  // CW triangle:  (0.0, 2.0, -2.0), (2.0, -1.5, -2.0), (-2.0, -1.5, -2.0)
  const geomCCW9 = new THREE.BufferGeometry();
  geomCCW9.setAttribute("position", new THREE.BufferAttribute(new Float32Array([
    -2.0, -1.5, -2.0,
     2.0, -1.5, -2.0,
     0.0,  2.0, -2.0,
  ]), 3));

  const geomCW9 = new THREE.BufferGeometry();
  geomCW9.setAttribute("position", new THREE.BufferAttribute(new Float32Array([
     0.0,  2.0, -2.0,
     2.0, -1.5, -2.0,
    -2.0, -1.5, -2.0,
  ]), 3));

  const matFront9 = new THREE.MeshBasicMaterial({ color: 0x00ff00, side: THREE.FrontSide, depthTest: true, depthWrite: true, depthFunc: THREE.LessDepth });
  const matBack9 = new THREE.MeshBasicMaterial({ color: 0x00ff00, side: THREE.BackSide, depthTest: true, depthWrite: true, depthFunc: THREE.LessDepth });
  const matDouble9 = new THREE.MeshBasicMaterial({ color: 0x00ff00, side: THREE.DoubleSide, depthTest: true, depthWrite: true, depthFunc: THREE.LessDepth });

  // 9a-1: FrontSide + CCW (det >= 0) -> Visible (Green)
  const meshFrontCCW = new THREE.Mesh(geomCCW9, matFront9);
  meshFrontCCW.updateMatrixWorld(true);
  await adapter.renderMeshBatch(bridgeHost, [meshFrontCCW], camera, null, wasmExports, { width, height });
  const cand9a1 = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
  const ref9a1 = await directMeshReference(device, buildIndependentBatchReferenceInput([meshFrontCCW], camera, width, height, { hasDepth: true, depthFormat: "depth24plus", depthWriteEnabled: true, depthCompare: "less" }));
  if (cand9a1[centerIdx + 1] === 0 || cand9a1[centerIdx] !== 0) {
    throw new Error(`Checkpoint 9a-1 failed: FrontSide CCW must be visible Green, got [${cand9a1[centerIdx]}, ${cand9a1[centerIdx+1]}, ${cand9a1[centerIdx+2]}]`);
  }
  if (differs(cand9a1, ref9a1)) {
    throw new Error("Checkpoint 9a-1 failed: FrontSide CCW candidate differs from direct WebGPU reference");
  }

  // 9a-2: FrontSide + CW (det >= 0) -> Culled (Clear color black)
  const meshFrontCW = new THREE.Mesh(geomCW9, matFront9);
  meshFrontCW.updateMatrixWorld(true);
  await adapter.renderMeshBatch(bridgeHost, [meshFrontCW], camera, null, wasmExports, { width, height });
  const cand9a2 = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
  const ref9a2 = await directMeshReference(device, buildIndependentBatchReferenceInput([meshFrontCW], camera, width, height, { hasDepth: true, depthFormat: "depth24plus", depthWriteEnabled: true, depthCompare: "less" }));
  if (cand9a2[centerIdx + 1] !== 0 || cand9a2[centerIdx] !== 0) {
    throw new Error(`Checkpoint 9a-2 failed: FrontSide CW must be culled (black), got [${cand9a2[centerIdx]}, ${cand9a2[centerIdx+1]}, ${cand9a2[centerIdx+2]}]`);
  }
  if (differs(cand9a2, ref9a2)) {
    throw new Error("Checkpoint 9a-2 failed: FrontSide CW candidate differs from direct WebGPU reference");
  }

  // 9a-3: BackSide + CCW (det >= 0) -> Culled (Clear color black)
  const meshBackCCW = new THREE.Mesh(geomCCW9, matBack9);
  meshBackCCW.updateMatrixWorld(true);
  await adapter.renderMeshBatch(bridgeHost, [meshBackCCW], camera, null, wasmExports, { width, height });
  const cand9a3 = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
  const ref9a3 = await directMeshReference(device, buildIndependentBatchReferenceInput([meshBackCCW], camera, width, height, { hasDepth: true, depthFormat: "depth24plus", depthWriteEnabled: true, depthCompare: "less" }));
  if (cand9a3[centerIdx + 1] !== 0 || cand9a3[centerIdx] !== 0) {
    throw new Error(`Checkpoint 9a-3 failed: BackSide CCW must be culled (black), got [${cand9a3[centerIdx]}, ${cand9a3[centerIdx+1]}, ${cand9a3[centerIdx+2]}]`);
  }
  if (differs(cand9a3, ref9a3)) {
    throw new Error("Checkpoint 9a-3 failed: BackSide CCW candidate differs from direct WebGPU reference");
  }

  // 9a-4: BackSide + CW (det >= 0) -> Visible (Green)
  const meshBackCW = new THREE.Mesh(geomCW9, matBack9);
  meshBackCW.updateMatrixWorld(true);
  await adapter.renderMeshBatch(bridgeHost, [meshBackCW], camera, null, wasmExports, { width, height });
  const cand9a4 = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
  const ref9a4 = await directMeshReference(device, buildIndependentBatchReferenceInput([meshBackCW], camera, width, height, { hasDepth: true, depthFormat: "depth24plus", depthWriteEnabled: true, depthCompare: "less" }));
  if (cand9a4[centerIdx + 1] === 0 || cand9a4[centerIdx] !== 0) {
    throw new Error(`Checkpoint 9a-4 failed: BackSide CW must be visible Green, got [${cand9a4[centerIdx]}, ${cand9a4[centerIdx+1]}, ${cand9a4[centerIdx+2]}]`);
  }
  if (differs(cand9a4, ref9a4)) {
    throw new Error("Checkpoint 9a-4 failed: BackSide CW candidate differs from direct WebGPU reference");
  }

  // 9a-5: DoubleSide + CCW and CW -> Both Visible
  const meshDoubleCCW = new THREE.Mesh(geomCCW9, matDouble9);
  const meshDoubleCW = new THREE.Mesh(geomCW9, matDouble9);
  meshDoubleCCW.updateMatrixWorld(true);
  meshDoubleCW.updateMatrixWorld(true);
  await adapter.renderMeshBatch(bridgeHost, [meshDoubleCCW], camera, null, wasmExports, { width, height });
  const cand9a5 = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
  const ref9a5 = await directMeshReference(device, buildIndependentBatchReferenceInput([meshDoubleCCW], camera, width, height, { hasDepth: true, depthFormat: "depth24plus", depthWriteEnabled: true, depthCompare: "less" }));
  if (cand9a5[centerIdx + 1] === 0) {
    throw new Error("Checkpoint 9a-5 failed: DoubleSide CCW must be visible Green");
  }
  if (differs(cand9a5, ref9a5)) {
    throw new Error("Checkpoint 9a-5 failed: DoubleSide CCW candidate differs from direct WebGPU reference");
  }

  // 9b: Reflected parent transform (group with scale.x = -1 -> determinantAffine() < 0)
  const groupReflected9 = new THREE.Group();
  groupReflected9.scale.set(-1, 1, 1);

  // 9b-1: Reflected FrontSide + CCW -> Screen CW, frontFace=CW -> Visible (Green)
  const meshReflFrontCCW = new THREE.Mesh(geomCCW9, matFront9);
  groupReflected9.add(meshReflFrontCCW);
  groupReflected9.updateMatrixWorld(true);
  await adapter.renderMeshBatch(bridgeHost, [meshReflFrontCCW], camera, null, wasmExports, { width, height });
  const cand9b1 = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
  const ref9b1 = await directMeshReference(device, buildIndependentBatchReferenceInput([meshReflFrontCCW], camera, width, height, { hasDepth: true, depthFormat: "depth24plus", depthWriteEnabled: true, depthCompare: "less" }));
  if (cand9b1[centerIdx + 1] === 0 || cand9b1[centerIdx] !== 0) {
    throw new Error(`Checkpoint 9b-1 failed: Reflected FrontSide CCW must be visible Green, got [${cand9b1[centerIdx]}, ${cand9b1[centerIdx+1]}, ${cand9b1[centerIdx+2]}]`);
  }
  if (differs(cand9b1, ref9b1)) {
    throw new Error("Checkpoint 9b-1 failed: Reflected FrontSide CCW candidate differs from direct WebGPU reference");
  }
  groupReflected9.remove(meshReflFrontCCW);

  // 9b-2: Reflected FrontSide + CW -> Screen CCW, frontFace=CW -> Culled (black)
  const meshReflFrontCW = new THREE.Mesh(geomCW9, matFront9);
  groupReflected9.add(meshReflFrontCW);
  groupReflected9.updateMatrixWorld(true);
  await adapter.renderMeshBatch(bridgeHost, [meshReflFrontCW], camera, null, wasmExports, { width, height });
  const cand9b2 = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
  const ref9b2 = await directMeshReference(device, buildIndependentBatchReferenceInput([meshReflFrontCW], camera, width, height, { hasDepth: true, depthFormat: "depth24plus", depthWriteEnabled: true, depthCompare: "less" }));
  if (cand9b2[centerIdx + 1] !== 0 || cand9b2[centerIdx] !== 0) {
    throw new Error(`Checkpoint 9b-2 failed: Reflected FrontSide CW must be culled (black), got [${cand9b2[centerIdx]}, ${cand9b2[centerIdx+1]}, ${cand9b2[centerIdx+2]}]`);
  }
  if (differs(cand9b2, ref9b2)) {
    throw new Error("Checkpoint 9b-2 failed: Reflected FrontSide CW candidate differs from direct WebGPU reference");
  }
  groupReflected9.remove(meshReflFrontCW);

  // 9b-3: Reflected BackSide + CCW -> Screen CW, frontFace=CCW -> Culled (black)
  const meshReflBackCCW = new THREE.Mesh(geomCCW9, matBack9);
  groupReflected9.add(meshReflBackCCW);
  groupReflected9.updateMatrixWorld(true);
  await adapter.renderMeshBatch(bridgeHost, [meshReflBackCCW], camera, null, wasmExports, { width, height });
  const cand9b3 = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
  const ref9b3 = await directMeshReference(device, buildIndependentBatchReferenceInput([meshReflBackCCW], camera, width, height, { hasDepth: true, depthFormat: "depth24plus", depthWriteEnabled: true, depthCompare: "less" }));
  if (cand9b3[centerIdx + 1] !== 0 || cand9b3[centerIdx] !== 0) {
    throw new Error(`Checkpoint 9b-3 failed: Reflected BackSide CCW must be culled (black), got [${cand9b3[centerIdx]}, ${cand9b3[centerIdx+1]}, ${cand9b3[centerIdx+2]}]`);
  }
  if (differs(cand9b3, ref9b3)) {
    throw new Error("Checkpoint 9b-3 failed: Reflected BackSide CCW candidate differs from direct WebGPU reference");
  }
  groupReflected9.remove(meshReflBackCCW);

  // 9b-4: Reflected BackSide + CW -> Screen CCW, frontFace=CCW -> Visible (Green)
  const meshReflBackCW = new THREE.Mesh(geomCW9, matBack9);
  groupReflected9.add(meshReflBackCW);
  groupReflected9.updateMatrixWorld(true);
  await adapter.renderMeshBatch(bridgeHost, [meshReflBackCW], camera, null, wasmExports, { width, height });
  const cand9b4 = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
  const ref9b4 = await directMeshReference(device, buildIndependentBatchReferenceInput([meshReflBackCW], camera, width, height, { hasDepth: true, depthFormat: "depth24plus", depthWriteEnabled: true, depthCompare: "less" }));
  if (cand9b4[centerIdx + 1] === 0 || cand9b4[centerIdx] !== 0) {
    throw new Error(`Checkpoint 9b-4 failed: Reflected BackSide CW must be visible Green, got [${cand9b4[centerIdx]}, ${cand9b4[centerIdx+1]}, ${cand9b4[centerIdx+2]}]`);
  }
  if (differs(cand9b4, ref9b4)) {
    throw new Error("Checkpoint 9b-4 failed: Reflected BackSide CW candidate differs from direct WebGPU reference");
  }
  groupReflected9.remove(meshReflBackCW);

  // 9c: Mixed-side multi-mesh batch in a single pass
  const leftIdx9 = 32 * bytesPerRow + 12 * 4;
  const rightIdx9 = 32 * bytesPerRow + 52 * 4;

  const geomLeft9 = new THREE.BufferGeometry();
  geomLeft9.setAttribute("position", new THREE.BufferAttribute(new Float32Array([
    -1.5, -1.5, -2.5,
    -0.3, -1.5, -2.5,
    -0.6,  1.5, -2.5,
  ]), 3));
  const matLeft9 = new THREE.MeshBasicMaterial({ color: 0x0000ff, side: THREE.DoubleSide, depthTest: true, depthWrite: true, depthFunc: THREE.LessDepth });
  const meshLeft9 = new THREE.Mesh(geomLeft9, matLeft9);
  meshLeft9.updateMatrixWorld(true);

  const geomCenter9 = new THREE.BufferGeometry();
  geomCenter9.setAttribute("position", new THREE.BufferAttribute(new Float32Array([
    -0.5, -1.0, -2.0,
     0.5, -1.0, -2.0,
     0.0,  1.0, -2.0,
  ]), 3));
  const matCenter9 = new THREE.MeshBasicMaterial({ color: 0x00ff00, side: THREE.FrontSide, depthTest: true, depthWrite: true, depthFunc: THREE.LessDepth });
  const meshCenter9 = new THREE.Mesh(geomCenter9, matCenter9);
  meshCenter9.updateMatrixWorld(true);

  const geomRight9 = new THREE.BufferGeometry();
  geomRight9.setAttribute("position", new THREE.BufferAttribute(new Float32Array([
    0.3, -1.5, -2.5,
    1.5, -1.5, -2.5,
    0.6,  1.5, -2.5,
  ]), 3));
  const matRight9 = new THREE.MeshBasicMaterial({ color: 0xff0000, side: THREE.FrontSide, depthTest: true, depthWrite: true, depthFunc: THREE.LessDepth });
  const meshRight9 = new THREE.Mesh(geomRight9, matRight9);
  meshRight9.updateMatrixWorld(true);

  // Culled mesh in foreground (BackSide CCW, Yellow)
  const matCulled9 = new THREE.MeshBasicMaterial({ color: 0xffff00, side: THREE.BackSide, depthTest: true, depthWrite: true, depthFunc: THREE.LessDepth });
  const meshCulled9 = new THREE.Mesh(geomCenter9, matCulled9);
  meshCulled9.position.set(0, 0, 0.5);
  meshCulled9.updateMatrixWorld(true);

  await adapter.renderMeshBatch(
    bridgeHost,
    [meshLeft9, meshCenter9, meshRight9, meshCulled9],
    camera,
    null,
    wasmExports,
    { width, height }
  );
  const cand9c = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
  const ref9c = await directMeshReference(
    device,
    buildIndependentBatchReferenceInput(
      [meshLeft9, meshCenter9, meshRight9, meshCulled9],
      camera,
      width,
      height,
      { hasDepth: true, depthFormat: "depth24plus", depthWriteEnabled: true, depthCompare: "less" }
    )
  );

  if (cand9c[leftIdx9 + 2] === 0 || cand9c[leftIdx9] !== 0) {
    throw new Error(`Checkpoint 9c failed: Mixed batch Left must be Blue, got [${cand9c[leftIdx9]}, ${cand9c[leftIdx9+1]}, ${cand9c[leftIdx9+2]}]`);
  }
  if (cand9c[centerIdx + 1] === 0 || cand9c[centerIdx] !== 0) {
    throw new Error(`Checkpoint 9c failed: Mixed batch Center must be Green, got [${cand9c[centerIdx]}, ${cand9c[centerIdx+1]}, ${cand9c[centerIdx+2]}]`);
  }
  if (cand9c[rightIdx9] === 0 || cand9c[rightIdx9 + 1] !== 0) {
    throw new Error(`Checkpoint 9c failed: Mixed batch Right must be Red, got [${cand9c[rightIdx9]}, ${cand9c[rightIdx9+1]}, ${cand9c[rightIdx9+2]}]`);
  }
  if (differs(cand9c, ref9c)) {
    throw new Error("Checkpoint 9c failed: Mixed-side batch candidate differs from direct WebGPU reference");
  }

  // 9d: Dynamic mutation across sequential frames
  const mutGeom9 = geomCCW9;
  const mutMat9 = new THREE.MeshBasicMaterial({ color: 0x00ff00, side: THREE.FrontSide, depthTest: true, depthWrite: true, depthFunc: THREE.LessDepth });
  const mutMesh9 = new THREE.Mesh(mutGeom9, mutMat9);
  mutMesh9.updateMatrixWorld(true);

  // Frame 1: FrontSide CCW -> Visible (Green)
  await adapter.renderMeshBatch(bridgeHost, [mutMesh9], camera, null, wasmExports, { width, height });
  const cand9d1 = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
  const ref9d1 = await directMeshReference(device, buildIndependentBatchReferenceInput([mutMesh9], camera, width, height, { hasDepth: true, depthFormat: "depth24plus", depthWriteEnabled: true, depthCompare: "less" }));
  if (cand9d1[centerIdx + 1] === 0 || cand9d1[centerIdx] !== 0) {
    throw new Error("Checkpoint 9d-1 failed: Mutated mesh frame 1 must be visible Green");
  }
  if (differs(cand9d1, ref9d1)) {
    throw new Error("Checkpoint 9d-1 failed: Mutated mesh frame 1 candidate differs from direct WebGPU reference");
  }

  // Frame 2: Mutate material.side to BackSide -> Culled (Black)
  mutMesh9.material.side = THREE.BackSide;
  mutMesh9.material.needsUpdate = true;
  await adapter.renderMeshBatch(bridgeHost, [mutMesh9], camera, null, wasmExports, { width, height });
  const cand9d2 = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
  const ref9d2 = await directMeshReference(device, buildIndependentBatchReferenceInput([mutMesh9], camera, width, height, { hasDepth: true, depthFormat: "depth24plus", depthWriteEnabled: true, depthCompare: "less" }));
  if (cand9d2[centerIdx + 1] !== 0 || cand9d2[centerIdx] !== 0) {
    throw new Error("Checkpoint 9d-2 failed: Mutated mesh frame 2 (side=BackSide) must be culled (black)");
  }
  if (differs(cand9d2, ref9d2)) {
    throw new Error("Checkpoint 9d-2 failed: Mutated mesh frame 2 candidate differs from direct WebGPU reference");
  }

  // Frame 3: Mutate material.side to FrontSide and scale.x to -1 -> Reflected FrontSide CCW stays Visible (Green)
  mutMesh9.material.side = THREE.FrontSide;
  mutMesh9.material.needsUpdate = true;
  mutMesh9.scale.set(-1, 1, 1);
  mutMesh9.updateMatrixWorld(true);
  await adapter.renderMeshBatch(bridgeHost, [mutMesh9], camera, null, wasmExports, { width, height });
  const cand9d3 = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
  const ref9d3 = await directMeshReference(device, buildIndependentBatchReferenceInput([mutMesh9], camera, width, height, { hasDepth: true, depthFormat: "depth24plus", depthWriteEnabled: true, depthCompare: "less" }));
  if (cand9d3[centerIdx + 1] === 0 || cand9d3[centerIdx] !== 0) {
    throw new Error("Checkpoint 9d-3 failed: Mutated mesh frame 3 (scale.x=-1, side=FrontSide) must be visible Green");
  }
  if (differs(cand9d3, ref9d3)) {
    throw new Error("Checkpoint 9d-3 failed: Mutated mesh frame 3 candidate differs from direct WebGPU reference");
  }

  // Frame 4: Mutate material.side to BackSide with scale.x = -1 -> Reflected BackSide CCW is Culled (Black)
  mutMesh9.material.side = THREE.BackSide;
  mutMesh9.material.needsUpdate = true;
  await adapter.renderMeshBatch(bridgeHost, [mutMesh9], camera, null, wasmExports, { width, height });
  const cand9d4 = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
  const ref9d4 = await directMeshReference(device, buildIndependentBatchReferenceInput([mutMesh9], camera, width, height, { hasDepth: true, depthFormat: "depth24plus", depthWriteEnabled: true, depthCompare: "less" }));
  if (cand9d4[centerIdx + 1] !== 0 || cand9d4[centerIdx] !== 0) {
    throw new Error("Checkpoint 9d-4 failed: Mutated mesh frame 4 (scale.x=-1, side=BackSide) must be culled (black)");
  }
  if (differs(cand9d4, ref9d4)) {
    throw new Error("Checkpoint 9d-4 failed: Mutated mesh frame 4 candidate differs from direct WebGPU reference");
  }

  // ---------------------------------------------------------------------------
  // Checkpoint 10: Mixed per-mesh depth states in a single submission
  // ---------------------------------------------------------------------------
  const leftIdx10 = 32 * bytesPerRow + 12 * 4;
  const rightIdx10 = 32 * bytesPerRow + 52 * 4;

  const geomNearLeft10 = new THREE.BufferGeometry();
  geomNearLeft10.setAttribute("position", new THREE.BufferAttribute(new Float32Array([
    -1.5, -1.5, -2.0,
    -0.3, -1.5, -2.0,
    -0.6,  1.5, -2.0,
  ]), 3));

  const geomNearRight10 = new THREE.BufferGeometry();
  geomNearRight10.setAttribute("position", new THREE.BufferAttribute(new Float32Array([
    0.3, -1.5, -2.0,
    1.5, -1.5, -2.0,
    0.6,  1.5, -2.0,
  ]), 3));

  const geomFarLeft10 = new THREE.BufferGeometry();
  geomFarLeft10.setAttribute("position", new THREE.BufferAttribute(new Float32Array([
    -1.5, -1.5, -2.5,
    -0.3, -1.5, -2.5,
    -0.6,  1.5, -2.5,
  ]), 3));

  const geomFarRight10 = new THREE.BufferGeometry();
  geomFarRight10.setAttribute("position", new THREE.BufferAttribute(new Float32Array([
    0.3, -1.5, -2.5,
    1.5, -1.5, -2.5,
    0.6,  1.5, -2.5,
  ]), 3));

  const geomNearCenter10 = new THREE.BufferGeometry();
  geomNearCenter10.setAttribute("position", new THREE.BufferAttribute(new Float32Array([
    -0.5, -1.0, -2.0,
     0.5, -1.0, -2.0,
     0.0,  1.0, -2.0,
  ]), 3));

  const geomFarCenter10 = new THREE.BufferGeometry();
  geomFarCenter10.setAttribute("position", new THREE.BufferAttribute(new Float32Array([
    -0.5, -1.0, -2.5,
     0.5, -1.0, -2.5,
     0.0,  1.0, -2.5,
  ]), 3));

  const geomMiddleCenter10 = new THREE.BufferGeometry();
  geomMiddleCenter10.setAttribute("position", new THREE.BufferAttribute(new Float32Array([
    -0.5, -1.0, -2.25,
     0.5, -1.0, -2.25,
     0.0,  1.0, -2.25,
  ]), 3));

  // 10a: Mixed depthWrite (true vs false) in a single pass
  // Near meshes at z = -2.0:
  // - meshNearLeft: Blue, depthWrite = false, depthTest = true, LessDepth
  // - meshNearRight: Red, depthWrite = true, depthTest = true, LessDepth
  // Far meshes at z = -2.5:
  // - meshFarLeft: Green, depthWrite = true, depthTest = true, LessDepth (passes against cleared depth -> overwrites Blue with Green)
  // - meshFarRight: Green, depthWrite = true, depthTest = true, LessDepth (farther depth fails Less -> stays Red)
  const meshNearLeft10a = new THREE.Mesh(geomNearLeft10, new THREE.MeshBasicMaterial({ color: 0x0000ff, side: THREE.FrontSide, depthTest: true, depthWrite: false, depthFunc: THREE.LessDepth }));
  const meshNearRight10a = new THREE.Mesh(geomNearRight10, new THREE.MeshBasicMaterial({ color: 0xff0000, side: THREE.FrontSide, depthTest: true, depthWrite: true, depthFunc: THREE.LessDepth }));
  const meshFarLeft10a = new THREE.Mesh(geomFarLeft10, new THREE.MeshBasicMaterial({ color: 0x00ff00, side: THREE.FrontSide, depthTest: true, depthWrite: true, depthFunc: THREE.LessDepth }));
  const meshFarRight10a = new THREE.Mesh(geomFarRight10, new THREE.MeshBasicMaterial({ color: 0x00ff00, side: THREE.FrontSide, depthTest: true, depthWrite: true, depthFunc: THREE.LessDepth }));
  meshNearLeft10a.updateMatrixWorld(true);
  meshNearRight10a.updateMatrixWorld(true);
  meshFarLeft10a.updateMatrixWorld(true);
  meshFarRight10a.updateMatrixWorld(true);

  const batch10a = [meshNearLeft10a, meshNearRight10a, meshFarLeft10a, meshFarRight10a];
  await adapter.renderMeshBatch(bridgeHost, batch10a, camera, null, wasmExports, { width, height });
  const cand10a = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
  const ref10a = await directMeshReference(device, buildIndependentBatchReferenceInput(batch10a, camera, width, height, { hasDepth: true, depthFormat: "depth24plus" }));

  if (cand10a[leftIdx10 + 1] === 0 || cand10a[leftIdx10] !== 0 || cand10a[leftIdx10 + 2] !== 0) {
    throw new Error(`Checkpoint 10a failed: Left must be overwritten to Green by far mesh when depthWrite=false, got [${cand10a[leftIdx10]}, ${cand10a[leftIdx10+1]}, ${cand10a[leftIdx10+2]}]`);
  }
  if (cand10a[rightIdx10] === 0 || cand10a[rightIdx10 + 1] !== 0 || cand10a[rightIdx10 + 2] !== 0) {
    throw new Error(`Checkpoint 10a failed: Right must stay Red (far mesh occluded by near depthWrite=true), got [${cand10a[rightIdx10]}, ${cand10a[rightIdx10+1]}, ${cand10a[rightIdx10+2]}]`);
  }
  if (differs(cand10a, ref10a)) {
    throw new Error("Checkpoint 10a failed: Mixed depthWrite candidate differs from direct WebGPU reference");
  }

  // 10b: Mixed depthFunc (LessDepth vs GreaterDepth) in a single pass
  // Base meshes at z = -2.0:
  // - meshBaseLeft: Red, depthWrite = true, depthTest = true, LessDepth
  // - meshBaseRight: Red, depthWrite = true, depthTest = true, LessDepth
  // Far meshes at z = -2.5:
  // - meshFarLeft: Green, depthWrite = true, depthTest = true, LessDepth (farther depth fails Less -> stays Red)
  // - meshFarRight: Blue, depthWrite = true, depthTest = true, GreaterDepth (farther depth passes Greater -> becomes Blue)
  const meshBaseLeft10b = new THREE.Mesh(geomNearLeft10, new THREE.MeshBasicMaterial({ color: 0xff0000, side: THREE.FrontSide, depthTest: true, depthWrite: true, depthFunc: THREE.LessDepth }));
  const meshBaseRight10b = new THREE.Mesh(geomNearRight10, new THREE.MeshBasicMaterial({ color: 0xff0000, side: THREE.FrontSide, depthTest: true, depthWrite: true, depthFunc: THREE.LessDepth }));
  const meshFarLeft10b = new THREE.Mesh(geomFarLeft10, new THREE.MeshBasicMaterial({ color: 0x00ff00, side: THREE.FrontSide, depthTest: true, depthWrite: true, depthFunc: THREE.LessDepth }));
  const meshFarRight10b = new THREE.Mesh(geomFarRight10, new THREE.MeshBasicMaterial({ color: 0x0000ff, side: THREE.FrontSide, depthTest: true, depthWrite: true, depthFunc: THREE.GreaterDepth }));
  meshBaseLeft10b.updateMatrixWorld(true);
  meshBaseRight10b.updateMatrixWorld(true);
  meshFarLeft10b.updateMatrixWorld(true);
  meshFarRight10b.updateMatrixWorld(true);

  const batch10b = [meshBaseLeft10b, meshBaseRight10b, meshFarLeft10b, meshFarRight10b];
  await adapter.renderMeshBatch(bridgeHost, batch10b, camera, null, wasmExports, { width, height });
  const cand10b = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
  const ref10b = await directMeshReference(device, buildIndependentBatchReferenceInput(batch10b, camera, width, height, { hasDepth: true, depthFormat: "depth24plus" }));

  if (cand10b[leftIdx10] === 0 || cand10b[leftIdx10 + 1] !== 0) {
    throw new Error(`Checkpoint 10b failed: Left must stay Red under LessDepth, got [${cand10b[leftIdx10]}, ${cand10b[leftIdx10+1]}, ${cand10b[leftIdx10+2]}]`);
  }
  if (cand10b[rightIdx10 + 2] === 0 || cand10b[rightIdx10] !== 0) {
    throw new Error(`Checkpoint 10b failed: Right must become Blue under GreaterDepth, got [${cand10b[rightIdx10]}, ${cand10b[rightIdx10+1]}, ${cand10b[rightIdx10+2]}]`);
  }
  if (differs(cand10b, ref10b)) {
    throw new Error("Checkpoint 10b failed: Mixed depthFunc candidate differs from direct WebGPU reference");
  }

  // 10c: Interleaved disabled depth (depthTest = false -> compare: Always)
  // Center:
  // 1. Center base (z = -2.0): Green, depthTest = true, depthWrite = true, LessDepth
  // 2. Center far (z = -2.5): Red, depthTest = false, depthWrite = false (Always passes, overwrites Green with Red)
  // 3. Center middle (z = -2.25): Blue, LessDepth fails against the base at -2.
  // Correct result is Red. An erroneous overlay depth write produces Blue;
  // erroneously depth-testing the overlay leaves Green instead.
  const meshBase10c = new THREE.Mesh(geomNearCenter10, new THREE.MeshBasicMaterial({ color: 0x00ff00, side: THREE.FrontSide, depthTest: true, depthWrite: true, depthFunc: THREE.LessDepth }));
  const meshFar10c = new THREE.Mesh(geomFarCenter10, new THREE.MeshBasicMaterial({ color: 0xff0000, side: THREE.FrontSide, depthTest: false, depthWrite: false }));
  const meshMiddle10c = new THREE.Mesh(geomMiddleCenter10, new THREE.MeshBasicMaterial({ color: 0x0000ff, side: THREE.FrontSide, depthTest: true, depthWrite: true, depthFunc: THREE.LessDepth }));
  meshBase10c.updateMatrixWorld(true);
  meshFar10c.updateMatrixWorld(true);
  meshMiddle10c.updateMatrixWorld(true);

  const batch10c = [meshBase10c, meshFar10c, meshMiddle10c];
  await adapter.renderMeshBatch(bridgeHost, batch10c, camera, null, wasmExports, { width, height });
  const cand10c = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
  const ref10c = await directMeshReference(device, buildIndependentBatchReferenceInput(batch10c, camera, width, height, { hasDepth: true, depthFormat: "depth24plus" }));

  if (cand10c[centerIdx] < 240 || cand10c[centerIdx + 1] > 15 || cand10c[centerIdx + 2] > 15) {
    throw new Error(`Checkpoint 10c failed: Center must stay Red (overlay did not write depth), got [${cand10c[centerIdx]}, ${cand10c[centerIdx+1]}, ${cand10c[centerIdx+2]}]`);
  }
  if (differs(cand10c, ref10c)) {
    throw new Error("Checkpoint 10c failed: Interleaved disabled depth candidate differs from direct WebGPU reference");
  }

  // 10d: Side/Reflection mixed with Depth in a single batch
  // - Left: FrontSide CCW, Blue, z = -2.0, depthWrite = true, LessDepth -> Visible Blue
  // - Center: Reflected FrontSide CCW (scale.x = -1), Green, z = -2.0, depthWrite = false, LessDepth -> Visible Green
  // - Right Foreground: BackSide CCW, Yellow (culled!), z = -1.5, depthWrite = true, LessDepth -> Culled
  // - Right Background: FrontSide CCW, Red, z = -2.5, depthWrite = true, LessDepth -> Visible Red
  const groupRefl10d = new THREE.Group();
  groupRefl10d.scale.set(-1, 1, 1);
  const meshCenter10d = new THREE.Mesh(geomNearCenter10, new THREE.MeshBasicMaterial({ color: 0x00ff00, side: THREE.FrontSide, depthTest: true, depthWrite: false, depthFunc: THREE.LessDepth }));
  groupRefl10d.add(meshCenter10d);
  groupRefl10d.updateMatrixWorld(true);

  const meshLeft10d = new THREE.Mesh(geomNearLeft10, new THREE.MeshBasicMaterial({ color: 0x0000ff, side: THREE.FrontSide, depthTest: true, depthWrite: true, depthFunc: THREE.LessDepth }));
  const meshRightFore10d = new THREE.Mesh(geomNearRight10, new THREE.MeshBasicMaterial({ color: 0xffff00, side: THREE.BackSide, depthTest: true, depthWrite: true, depthFunc: THREE.LessDepth }));
  meshRightFore10d.position.set(0, 0, 0.5); // z = -1.5
  const meshRightBack10d = new THREE.Mesh(geomFarRight10, new THREE.MeshBasicMaterial({ color: 0xff0000, side: THREE.FrontSide, depthTest: true, depthWrite: true, depthFunc: THREE.LessDepth }));
  meshLeft10d.updateMatrixWorld(true);
  meshRightFore10d.updateMatrixWorld(true);
  meshRightBack10d.updateMatrixWorld(true);

  const batch10d = [meshLeft10d, meshCenter10d, meshRightFore10d, meshRightBack10d];
  await adapter.renderMeshBatch(bridgeHost, batch10d, camera, null, wasmExports, { width, height });
  const cand10d = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
  const ref10d = await directMeshReference(device, buildIndependentBatchReferenceInput(batch10d, camera, width, height, { hasDepth: true, depthFormat: "depth24plus" }));

  if (cand10d[leftIdx10 + 2] === 0 || cand10d[leftIdx10] !== 0) {
    throw new Error(`Checkpoint 10d failed: Left must be Blue, got [${cand10d[leftIdx10]}, ${cand10d[leftIdx10+1]}, ${cand10d[leftIdx10+2]}]`);
  }
  if (cand10d[centerIdx + 1] === 0 || cand10d[centerIdx] !== 0) {
    throw new Error(`Checkpoint 10d failed: Center must be Green (reflected FrontSide), got [${cand10d[centerIdx]}, ${cand10d[centerIdx+1]}, ${cand10d[centerIdx+2]}]`);
  }
  if (cand10d[rightIdx10] === 0 || cand10d[rightIdx10 + 1] !== 0) {
    throw new Error(`Checkpoint 10d failed: Right must be Red (foreground Yellow culled), got [${cand10d[rightIdx10]}, ${cand10d[rightIdx10+1]}, ${cand10d[rightIdx10+2]}]`);
  }
  if (differs(cand10d, ref10d)) {
    throw new Error("Checkpoint 10d failed: Side/reflection with depth candidate differs from direct WebGPU reference");
  }

  // 10e: Multi-frame material depth mutation
  const baseMesh10e = new THREE.Mesh(geomNearCenter10, new THREE.MeshBasicMaterial({ color: 0x00ff00, side: THREE.FrontSide, depthTest: true, depthWrite: true, depthFunc: THREE.LessDepth }));
  const mutMesh10e = new THREE.Mesh(geomFarCenter10, new THREE.MeshBasicMaterial({ color: 0xff0000, side: THREE.FrontSide, depthTest: true, depthWrite: true, depthFunc: THREE.LessDepth }));
  baseMesh10e.updateMatrixWorld(true);
  mutMesh10e.updateMatrixWorld(true);

  // Frame 1: mutMesh at z = -2.5 with LessDepth is occluded by baseMesh at z = -2.0 -> Center stays Green
  await adapter.renderMeshBatch(bridgeHost, [baseMesh10e, mutMesh10e], camera, null, wasmExports, { width, height });
  const cand10e1 = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
  const ref10e1 = await directMeshReference(device, buildIndependentBatchReferenceInput([baseMesh10e, mutMesh10e], camera, width, height, { hasDepth: true, depthFormat: "depth24plus" }));
  if (cand10e1[centerIdx + 1] === 0 || cand10e1[centerIdx] !== 0) {
    throw new Error("Checkpoint 10e-1 failed: Frame 1 must stay Green (mutMesh occluded)");
  }
  if (differs(cand10e1, ref10e1)) {
    throw new Error("Checkpoint 10e-1 failed: Frame 1 candidate differs from direct WebGPU reference");
  }

  // Frame 2: Mutate mutMesh to GreaterDepth -> passes test -> Center becomes Red
  mutMesh10e.material.depthFunc = THREE.GreaterDepth;
  mutMesh10e.material.needsUpdate = true;
  await adapter.renderMeshBatch(bridgeHost, [baseMesh10e, mutMesh10e], camera, null, wasmExports, { width, height });
  const cand10e2 = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
  const ref10e2 = await directMeshReference(device, buildIndependentBatchReferenceInput([baseMesh10e, mutMesh10e], camera, width, height, { hasDepth: true, depthFormat: "depth24plus" }));
  if (cand10e2[centerIdx] === 0 || cand10e2[centerIdx + 1] !== 0) {
    throw new Error("Checkpoint 10e-2 failed: Frame 2 (GreaterDepth) must become Red");
  }
  if (differs(cand10e2, ref10e2)) {
    throw new Error("Checkpoint 10e-2 failed: Frame 2 candidate differs from direct WebGPU reference");
  }

  // Frame 3: Mutate mutMesh to depthTest=false -> compare: Always -> Center stays Red
  mutMesh10e.material.depthFunc = THREE.LessDepth;
  mutMesh10e.material.depthTest = false;
  mutMesh10e.material.depthWrite = false;
  mutMesh10e.material.needsUpdate = true;
  await adapter.renderMeshBatch(bridgeHost, [baseMesh10e, mutMesh10e], camera, null, wasmExports, { width, height });
  const cand10e3 = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
  const ref10e3 = await directMeshReference(device, buildIndependentBatchReferenceInput([baseMesh10e, mutMesh10e], camera, width, height, { hasDepth: true, depthFormat: "depth24plus" }));
  if (cand10e3[centerIdx] === 0 || cand10e3[centerIdx + 1] !== 0) {
    throw new Error("Checkpoint 10e-3 failed: Frame 3 (depthTest=false) must become Red");
  }
  if (differs(cand10e3, ref10e3)) {
    throw new Error("Checkpoint 10e-3 failed: Frame 3 candidate differs from direct WebGPU reference");
  }

  // 10f: Scene entry with mixed depth, sampled from the actual canvas texture.
  if (canvasContext) {
    const scene10f = new THREE.Scene();
    batch10a.forEach((mesh, i) => {
      mesh.renderOrder = i;
      scene10f.add(mesh);
    });
    const canvasFormat10f = navigator.gpu.getPreferredCanvasFormat();
    canvasContext.configure({
      device,
      format: canvasFormat10f,
      alphaMode: "opaque",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    await nextFrame();
    const canvasReadback10f = device.createBuffer({
      size: bytesPerRow * height,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const currentTexture10f = canvasContext.getCurrentTexture();
    const render10f = adapter.renderScene(bridgeHost, scene10f, camera, canvasContext, wasmExports, { width, height, sourceBackend: "webgpu" });
    const copy10f = device.createCommandEncoder();
    copy10f.copyTextureToBuffer(
      { texture: currentTexture10f },
      { buffer: canvasReadback10f, bytesPerRow },
      [width, height, 1]
    );
    device.queue.submit([copy10f.finish()]);
    const result10f = await render10f;
    if (result10f.admitted.length !== 4 || result10f.refused.length !== 0) {
      throw new Error(`Checkpoint 10f failed: Mixed-depth scene refused: ${JSON.stringify(result10f)}`);
    }
    await canvasReadback10f.mapAsync(GPUMapMode.READ);
    const cand10f = new Uint8Array(canvasReadback10f.getMappedRange().slice(0));
    canvasReadback10f.unmap();
    canvasReadback10f.destroy();
    const ref10f = await directMeshReference(device, buildIndependentBatchReferenceInput(batch10a, camera, width, height, {
      hasDepth: true, depthFormat: "depth24plus", format: canvasFormat10f, outputSrgb: true,
    }));
    const red10f = canvasFormat10f.startsWith("bgra") ? 2 : 0;
    const blue10f = canvasFormat10f.startsWith("bgra") ? 0 : 2;
    if (cand10f[leftIdx10 + 1] < 240 || cand10f[leftIdx10 + red10f] > 15 || cand10f[leftIdx10 + blue10f] > 15) {
      throw new Error(`Checkpoint 10f failed: Canvas Left must be Green, got [${cand10f.slice(leftIdx10, leftIdx10 + 4)}]`);
    }
    if (cand10f[rightIdx10 + red10f] < 240 || cand10f[rightIdx10 + 1] > 15 || cand10f[rightIdx10 + blue10f] > 15) {
      throw new Error(`Checkpoint 10f failed: Canvas Right must be Red, got [${cand10f.slice(rightIdx10, rightIdx10 + 4)}]`);
    }
    if (differs(cand10f, ref10f)) {
      throw new Error("Checkpoint 10f failed: Canvas mixed depth candidate differs from direct WebGPU reference");
    }
  }

  // ---------------------------------------------------------------------------
  // Checkpoint 11: material.colorWrite=false (invisible depth occluders)
  // ---------------------------------------------------------------------------
  const geomOccluder11 = new THREE.BufferGeometry();
  geomOccluder11.setAttribute("position", new THREE.BufferAttribute(new Float32Array([
    -1.0, -1.0, -1.5,
     1.0, -1.0, -1.5,
     0.0,  1.0, -1.5,
  ]), 3));

  // 11a: Invisible near occluder (colorWrite=false, depthWrite=true)
  // 1. Base mesh at z = -2.0: Red (0xff0000), depthWrite=false, depthTest=true, LessDepth
  // 2. Occluder at z = -1.5: colorWrite=false, depthWrite=true, depthTest=true, LessDepth
  //    (writes depth at -1.5, leaves framebuffer color Red)
  // 3. Far mesh at z = -2.5: Green (0x00ff00), colorWrite=true, depthWrite=true, depthTest=true, LessDepth
  //    (fails depth test against occluder's -1.5 -> rejected)
  // Expected: Center pixel stays Red.
  const meshBase11a = new THREE.Mesh(geomNearCenter10, new THREE.MeshBasicMaterial({ color: 0xff0000, side: THREE.FrontSide, depthTest: true, depthWrite: false, depthFunc: THREE.LessDepth }));
  const meshOccluder11a = new THREE.Mesh(geomOccluder11, new THREE.MeshBasicMaterial({ color: 0x0000ff, colorWrite: false, side: THREE.FrontSide, depthTest: true, depthWrite: true, depthFunc: THREE.LessDepth }));
  const meshFar11a = new THREE.Mesh(geomFarCenter10, new THREE.MeshBasicMaterial({ color: 0x00ff00, colorWrite: true, side: THREE.FrontSide, depthTest: true, depthWrite: true, depthFunc: THREE.LessDepth }));
  meshBase11a.updateMatrixWorld(true);
  meshOccluder11a.updateMatrixWorld(true);
  meshFar11a.updateMatrixWorld(true);

  const batch11a = [meshBase11a, meshOccluder11a, meshFar11a];
  await adapter.renderMeshBatch(bridgeHost, batch11a, camera, null, wasmExports, { width, height });
  const cand11a = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
  const ref11a = await directMeshReference(device, buildIndependentBatchReferenceInput(batch11a, camera, width, height, { hasDepth: true, depthFormat: "depth24plus" }));

  if (cand11a[centerIdx] < 240 || cand11a[centerIdx + 1] > 15 || cand11a[centerIdx + 2] > 15) {
    throw new Error(`Checkpoint 11a failed: Center must stay Red (near occluder with depthWrite=true occluded far Green), got [${cand11a[centerIdx]}, ${cand11a[centerIdx+1]}, ${cand11a[centerIdx+2]}]`);
  }
  if (differs(cand11a, ref11a)) {
    throw new Error("Checkpoint 11a failed: Invisible occluder candidate differs from direct WebGPU reference");
  }

  // 11b: Same occluder with depthWrite=false allows far Green through
  // 1. Base mesh at z = -2.0: Red, depthWrite=false, depthTest=true, LessDepth
  // 2. Occluder at z = -1.5: colorWrite=false, depthWrite=false, depthTest=true, LessDepth
  // 3. Far mesh at z = -2.5: Green, colorWrite=true, depthWrite=true, depthTest=true, LessDepth
  //    (passes against clear depth -> overwrites to Green)
  // Expected: Center pixel becomes Green.
  const meshBase11b = new THREE.Mesh(geomNearCenter10, new THREE.MeshBasicMaterial({ color: 0xff0000, side: THREE.FrontSide, depthTest: true, depthWrite: false, depthFunc: THREE.LessDepth }));
  const meshOccluder11b = new THREE.Mesh(geomOccluder11, new THREE.MeshBasicMaterial({ color: 0x0000ff, colorWrite: false, side: THREE.FrontSide, depthTest: true, depthWrite: false, depthFunc: THREE.LessDepth }));
  const meshFar11b = new THREE.Mesh(geomFarCenter10, new THREE.MeshBasicMaterial({ color: 0x00ff00, colorWrite: true, side: THREE.FrontSide, depthTest: true, depthWrite: true, depthFunc: THREE.LessDepth }));
  meshBase11b.updateMatrixWorld(true);
  meshOccluder11b.updateMatrixWorld(true);
  meshFar11b.updateMatrixWorld(true);

  const batch11b = [meshBase11b, meshOccluder11b, meshFar11b];
  await adapter.renderMeshBatch(bridgeHost, batch11b, camera, null, wasmExports, { width, height });
  const cand11b = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
  const ref11b = await directMeshReference(device, buildIndependentBatchReferenceInput(batch11b, camera, width, height, { hasDepth: true, depthFormat: "depth24plus" }));

  if (cand11b[centerIdx + 1] < 240 || cand11b[centerIdx] > 15 || cand11b[centerIdx + 2] > 15) {
    throw new Error(`Checkpoint 11b failed: Center must become Green (depthWrite=false occluder allowed far Green), got [${cand11b[centerIdx]}, ${cand11b[centerIdx+1]}, ${cand11b[centerIdx+2]}]`);
  }
  if (differs(cand11b, ref11b)) {
    throw new Error("Checkpoint 11b failed: Non-writing occluder candidate differs from direct WebGPU reference");
  }
  if (!differs(cand11a, cand11b)) {
    throw new Error("Checkpoint 11b failed: Changing only occluder depthWrite must change the rendered image");
  }

  // 11c: Interleaved visible / invisible in a single batch (spatial multi-mesh)
  // Left: Base Blue (z = -2.0, depthWrite=false), Occluder (z = -1.5, colorWrite=false, depthWrite=true), Far Yellow (z = -2.5, occluded) -> Left stays Blue
  // Right: Base Red (z = -2.0, depthWrite=false), Occluder (z = -1.5, colorWrite=false, depthWrite=false), Far Green (z = -2.5, depthWrite=true) -> Right becomes Green
  const geomOccluderLeft11 = new THREE.BufferGeometry();
  geomOccluderLeft11.setAttribute("position", new THREE.BufferAttribute(new Float32Array([
    -1.0, -1.0, -1.5,
     0.0, -1.0, -1.5,
    -0.5,  1.0, -1.5,
  ]), 3));
  const geomOccluderRight11 = new THREE.BufferGeometry();
  geomOccluderRight11.setAttribute("position", new THREE.BufferAttribute(new Float32Array([
     0.0, -1.0, -1.5,
     1.0, -1.0, -1.5,
     0.5,  1.0, -1.5,
  ]), 3));

  const meshLeftBase11c = new THREE.Mesh(geomNearLeft10, new THREE.MeshBasicMaterial({ color: 0x0000ff, side: THREE.FrontSide, depthTest: true, depthWrite: false, depthFunc: THREE.LessDepth }));
  const meshLeftOcc11c = new THREE.Mesh(geomOccluderLeft11, new THREE.MeshBasicMaterial({ colorWrite: false, side: THREE.FrontSide, depthTest: true, depthWrite: true, depthFunc: THREE.LessDepth }));
  const meshLeftFar11c = new THREE.Mesh(geomFarLeft10, new THREE.MeshBasicMaterial({ color: 0xffff00, side: THREE.FrontSide, depthTest: true, depthWrite: true, depthFunc: THREE.LessDepth }));

  const meshRightBase11c = new THREE.Mesh(geomNearRight10, new THREE.MeshBasicMaterial({ color: 0xff0000, side: THREE.FrontSide, depthTest: true, depthWrite: false, depthFunc: THREE.LessDepth }));
  const meshRightOcc11c = new THREE.Mesh(geomOccluderRight11, new THREE.MeshBasicMaterial({ colorWrite: false, side: THREE.FrontSide, depthTest: true, depthWrite: false, depthFunc: THREE.LessDepth }));
  const meshRightFar11c = new THREE.Mesh(geomFarRight10, new THREE.MeshBasicMaterial({ color: 0x00ff00, side: THREE.FrontSide, depthTest: true, depthWrite: true, depthFunc: THREE.LessDepth }));

  meshLeftBase11c.updateMatrixWorld(true);
  meshLeftOcc11c.updateMatrixWorld(true);
  meshLeftFar11c.updateMatrixWorld(true);
  meshRightBase11c.updateMatrixWorld(true);
  meshRightOcc11c.updateMatrixWorld(true);
  meshRightFar11c.updateMatrixWorld(true);

  const batch11c = [meshLeftBase11c, meshLeftOcc11c, meshLeftFar11c, meshRightBase11c, meshRightOcc11c, meshRightFar11c];
  await adapter.renderMeshBatch(bridgeHost, batch11c, camera, null, wasmExports, { width, height });
  const cand11c = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
  const ref11c = await directMeshReference(device, buildIndependentBatchReferenceInput(batch11c, camera, width, height, { hasDepth: true, depthFormat: "depth24plus" }));

  if (cand11c[leftIdx10 + 2] < 240 || cand11c[leftIdx10] > 15 || cand11c[leftIdx10 + 1] > 15) {
    throw new Error(`Checkpoint 11c failed: Left must stay Blue (occluder wrote depth), got [${cand11c[leftIdx10]}, ${cand11c[leftIdx10+1]}, ${cand11c[leftIdx10+2]}]`);
  }
  if (cand11c[rightIdx10 + 1] < 240 || cand11c[rightIdx10] > 15 || cand11c[rightIdx10 + 2] > 15) {
    throw new Error(`Checkpoint 11c failed: Right must become Green (non-writing occluder), got [${cand11c[rightIdx10]}, ${cand11c[rightIdx10+1]}, ${cand11c[rightIdx10+2]}]`);
  }
  if (differs(cand11c, ref11c)) {
    throw new Error("Checkpoint 11c failed: Interleaved visible/invisible batch candidate differs from direct WebGPU reference");
  }

  // 11d: Dynamic frame-to-frame colorWrite mutation
  const baseMesh11d = new THREE.Mesh(geomNearCenter10, new THREE.MeshBasicMaterial({ color: 0xff0000, side: THREE.FrontSide, depthTest: true, depthWrite: false, depthFunc: THREE.LessDepth }));
  const mutMesh11d = new THREE.Mesh(geomOccluder11, new THREE.MeshBasicMaterial({ color: 0x0000ff, colorWrite: false, side: THREE.FrontSide, depthTest: true, depthWrite: true, depthFunc: THREE.LessDepth }));
  const farMesh11d = new THREE.Mesh(geomFarCenter10, new THREE.MeshBasicMaterial({ color: 0x00ff00, colorWrite: true, side: THREE.FrontSide, depthTest: true, depthWrite: true, depthFunc: THREE.LessDepth }));
  baseMesh11d.updateMatrixWorld(true);
  mutMesh11d.updateMatrixWorld(true);
  farMesh11d.updateMatrixWorld(true);

  // Frame 1: mutMesh has colorWrite=false, depthWrite=true -> Center stays Red
  await adapter.renderMeshBatch(bridgeHost, [baseMesh11d, mutMesh11d, farMesh11d], camera, null, wasmExports, { width, height });
  const cand11d1 = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
  const ref11d1 = await directMeshReference(device, buildIndependentBatchReferenceInput([baseMesh11d, mutMesh11d, farMesh11d], camera, width, height, { hasDepth: true, depthFormat: "depth24plus" }));
  if (cand11d1[centerIdx] < 240 || cand11d1[centerIdx + 1] > 15 || cand11d1[centerIdx + 2] > 15) {
    throw new Error("Checkpoint 11d-1 failed: Frame 1 must stay Red (colorWrite=false occluder)");
  }
  if (differs(cand11d1, ref11d1)) {
    throw new Error("Checkpoint 11d-1 failed: Frame 1 candidate differs from direct WebGPU reference");
  }

  // Frame 2: Mutate mutMesh to colorWrite=true -> mutMesh writes Blue at -1.5 -> Center becomes Blue
  mutMesh11d.material.colorWrite = true;
  mutMesh11d.material.needsUpdate = true;
  await adapter.renderMeshBatch(bridgeHost, [baseMesh11d, mutMesh11d, farMesh11d], camera, null, wasmExports, { width, height });
  const cand11d2 = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
  const ref11d2 = await directMeshReference(device, buildIndependentBatchReferenceInput([baseMesh11d, mutMesh11d, farMesh11d], camera, width, height, { hasDepth: true, depthFormat: "depth24plus" }));
  if (cand11d2[centerIdx + 2] < 240 || cand11d2[centerIdx] > 15 || cand11d2[centerIdx + 1] > 15) {
    throw new Error(`Checkpoint 11d-2 failed: Frame 2 (mutated colorWrite=true) must become Blue, got [${cand11d2[centerIdx]}, ${cand11d2[centerIdx+1]}, ${cand11d2[centerIdx+2]}]`);
  }
  if (differs(cand11d2, ref11d2)) {
    throw new Error("Checkpoint 11d-2 failed: Frame 2 candidate differs from direct WebGPU reference");
  }

  // Frame 3: Mutate mutMesh back to colorWrite=false -> Center stays Red
  mutMesh11d.material.colorWrite = false;
  mutMesh11d.material.needsUpdate = true;
  await adapter.renderMeshBatch(bridgeHost, [baseMesh11d, mutMesh11d, farMesh11d], camera, null, wasmExports, { width, height });
  const cand11d3 = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
  const ref11d3 = await directMeshReference(device, buildIndependentBatchReferenceInput([baseMesh11d, mutMesh11d, farMesh11d], camera, width, height, { hasDepth: true, depthFormat: "depth24plus" }));
  if (cand11d3[centerIdx] < 240 || cand11d3[centerIdx + 1] > 15 || cand11d3[centerIdx + 2] > 15) {
    throw new Error("Checkpoint 11d-3 failed: Frame 3 (mutated colorWrite=false) must stay Red");
  }
  if (differs(cand11d3, ref11d3)) {
    throw new Error("Checkpoint 11d-3 failed: Frame 3 candidate differs from direct WebGPU reference");
  }

  // 11e: Canvas positive execution of colorWrite=false occluder via renderScene
  if (canvasContext) {
    const scene11e = new THREE.Scene();
    batch11a.forEach((mesh, i) => {
      mesh.renderOrder = i;
      scene11e.add(mesh);
    });
    const canvasFormat11e = navigator.gpu.getPreferredCanvasFormat();
    canvasContext.configure({
      device,
      format: canvasFormat11e,
      alphaMode: "opaque",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    await nextFrame();
    const canvasReadback11e = device.createBuffer({
      size: bytesPerRow * height,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const currentTexture11e = canvasContext.getCurrentTexture();
    const render11e = adapter.renderScene(bridgeHost, scene11e, camera, canvasContext, wasmExports, { width, height, sourceBackend: "webgpu" });
    const copy11e = device.createCommandEncoder();
    copy11e.copyTextureToBuffer(
      { texture: currentTexture11e },
      { buffer: canvasReadback11e, bytesPerRow },
      [width, height, 1]
    );
    device.queue.submit([copy11e.finish()]);
    const result11e = await render11e;
    if (result11e.admitted.length !== 3 || result11e.refused.length !== 0) {
      throw new Error(`Checkpoint 11e failed: Occluder scene refused: ${JSON.stringify(result11e)}`);
    }
    await canvasReadback11e.mapAsync(GPUMapMode.READ);
    const cand11e = new Uint8Array(canvasReadback11e.getMappedRange().slice(0));
    canvasReadback11e.unmap();
    canvasReadback11e.destroy();
    const ref11e = await directMeshReference(device, buildIndependentBatchReferenceInput(batch11a, camera, width, height, {
      hasDepth: true, depthFormat: "depth24plus", format: canvasFormat11e, outputSrgb: true,
    }));
    const red11e = canvasFormat11e.startsWith("bgra") ? 2 : 0;
    const blue11e = canvasFormat11e.startsWith("bgra") ? 0 : 2;
    if (cand11e[centerIdx + red11e] < 240 || cand11e[centerIdx + 1] > 15 || cand11e[centerIdx + blue11e] > 15) {
      throw new Error(`Checkpoint 11e failed: Canvas Center must stay Red, got [${cand11e.slice(centerIdx, centerIdx + 4)}]`);
    }
    if (differs(cand11e, ref11e)) {
      throw new Error("Checkpoint 11e failed: Canvas occluder candidate differs from direct WebGPU reference");
    }
  }

  // 11f: Public single-mesh routing must retain the mask for DoubleSide too.
  // Exercise the actual color export alone when disabled, then mutate to visible
  // with the full module. Black vs white detects a silently used legacy export.
  const single11f = new THREE.Mesh(geomOccluder11, new THREE.MeshBasicMaterial({
    color: 0xffffff, side: THREE.DoubleSide, depthFunc: THREE.LessDepth,
  }));
  single11f.updateMatrixWorld(true);
  const colorOnlyExports11f = {
    f3d_build_mesh_batch_cull_depth_color_packet: wasmExports.f3d_build_mesh_batch_cull_depth_color_packet,
  };
  for (const context11f of (canvasContext ? [null, canvasContext] : [null])) {
    const format11f = context11f ? navigator.gpu.getPreferredCanvasFormat() : "rgba8unorm";
    for (const [depthEnabled11f, colorWrite11f] of [[true, false], [true, true], [false, false], [false, true]]) {
      single11f.material.depthTest = depthEnabled11f;
      single11f.material.depthWrite = depthEnabled11f;
      single11f.material.colorWrite = colorWrite11f;
      const exports11f = colorWrite11f ? wasmExports : colorOnlyExports11f;
      let candidate11f;
      if (context11f) {
        context11f.configure({ device, format: format11f, alphaMode: "opaque",
          usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
        await nextFrame();
        const readback11f = device.createBuffer({ size: bytesPerRow * height,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const texture11f = context11f.getCurrentTexture();
        const render11f = adapter.renderMesh(bridgeHost, single11f, camera, context11f, exports11f, { width, height });
        const copy11f = device.createCommandEncoder();
        copy11f.copyTextureToBuffer({ texture: texture11f },
          { buffer: readback11f, bytesPerRow }, [width, height, 1]);
        device.queue.submit([copy11f.finish()]);
        await render11f;
        await readback11f.mapAsync(GPUMapMode.READ);
        candidate11f = new Uint8Array(readback11f.getMappedRange().slice(0));
        readback11f.unmap();
        readback11f.destroy();
      } else {
        await adapter.renderMesh(bridgeHost, single11f, camera, null, exports11f, { width, height });
        candidate11f = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
      }
      const reference11f = await directMeshReference(device,
        buildIndependentBatchReferenceInput([single11f], camera, width, height, {
          hasDepth: depthEnabled11f, depthFormat: "depth24plus", format: format11f, outputSrgb: !!context11f,
        }));
      const expected11f = colorWrite11f ? 255 : 0;
      if ([0, 1, 2].some(channel => Math.abs(candidate11f[centerIdx + channel] - expected11f) > 15)) {
        throw new Error(`Checkpoint 11f failed: DoubleSide single depth=${depthEnabled11f}, colorWrite=${colorWrite11f}, canvas=${!!context11f}: expected RGB ${expected11f}, got ${candidate11f.slice(centerIdx, centerIdx + 4)}`);
      }
      if (differs(candidate11f, reference11f)) {
        throw new Error(`Checkpoint 11f failed: DoubleSide single depth=${depthEnabled11f}, colorWrite=${colorWrite11f}, canvas=${!!context11f} differs from direct WebGPU reference`);
      }
    }
  }

  // 11g: A no-depth batch still needs mutually compatible pipelines/pass state
  // when one mesh suppresses color. The later Blue draw must leave Red intact.
  for (const mesh11g of [meshBase11a, meshOccluder11a]) {
    mesh11g.material.depthTest = false;
    mesh11g.material.depthWrite = false;
  }
  const batch11g = [meshBase11a, meshOccluder11a];
  await adapter.renderMeshBatch(bridgeHost, batch11g, camera, null, colorOnlyExports11f, { width, height });
  const candidate11g = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
  const reference11g = await directMeshReference(device,
    buildIndependentBatchReferenceInput(batch11g, camera, width, height, { hasDepth: false }));
  if (candidate11g[centerIdx] < 240 || candidate11g[centerIdx + 1] > 15 || candidate11g[centerIdx + 2] > 15) {
    throw new Error(`Checkpoint 11g failed: No-depth mask batch must stay Red, got ${candidate11g.slice(centerIdx, centerIdx + 4)}`);
  }
  if (differs(candidate11g, reference11g)) {
    throw new Error("Checkpoint 11g failed: No-depth mask batch differs from direct WebGPU reference");
  }

  // 12: Persistent uploads must preserve GPU-stale bytes independently of CPU edits.
  // Keep separate source attributes: WebGL consumes updateRanges when it uploads.
  const oracleThree12 = await import("/upstream/three.js/build/three.module.js");
  const initialPositions12 = [-0.8, -0.8, -2, 0.8, -0.8, -2, 0, 0.8, -2];
  const makeUploadMesh12 = (library) => {
    const geometry = new library.BufferGeometry();
    geometry.setAttribute("position", new library.BufferAttribute(new Float32Array(initialPositions12), 3));
    const mesh = new library.Mesh(geometry, new library.MeshBasicMaterial({
      color: 0xff0000, side: library.DoubleSide, depthTest: false, depthWrite: false,
    }));
    mesh.frustumCulled = false;
    mesh.updateMatrixWorld(true);
    const scene = new library.Scene();
    scene.add(mesh);
    return { mesh, geometry, scene, library };
  };
  const reference12 = makeUploadMesh12(oracleThree12);
  const candidate12 = makeUploadMesh12(THREE);
  const pair12 = [reference12, candidate12];
  const camera12 = new oracleThree12.OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
  camera12.updateMatrixWorld(true);
  const renderer12 = new oracleThree12.WebGLRenderer({
    canvas: document.createElement("canvas"), antialias: false, preserveDrawingBuffer: true,
  });
  renderer12.setSize(width, height, false);
  renderer12.setClearColor(0x000000, 1);
  renderer12.toneMapping = oracleThree12.NoToneMapping;
  renderer12.outputColorSpace = oracleThree12.SRGBColorSpace;
  const gl12 = renderer12.getContext();
  const probes12 = [[32, 32], [54, 22]];
  const options12 = { width, height, sourceBackend: "webgl" };
  async function sampleUpload12(phase, expectedRed, route = "mesh") {
    renderer12.render(reference12.scene, camera12);
    const referencePixels = probes12.map(([x, y]) => {
      const pixel = new Uint8Array(4);
      gl12.readPixels(x, height - 1 - y, 1, 1, gl12.RGBA, gl12.UNSIGNED_BYTE, pixel);
      return pixel;
    });
    if (route === "scene") {
      const result = await adapter.renderScene(bridgeHost, candidate12.scene, camera12, null, wasmExports, options12);
      if (result.admitted.length !== 1 || result.refused.length !== 0) {
        throw new Error(`Checkpoint 12 ${phase}: upload scene refused: ${JSON.stringify(result)}`);
      }
    } else if (route === "batch") {
      await adapter.renderMeshBatch(bridgeHost, [candidate12.mesh], camera12, null, wasmExports, options12);
    } else {
      await adapter.renderMesh(bridgeHost, candidate12.mesh, camera12, null, wasmExports, options12);
    }
    const pixels = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
    probes12.forEach(([x, y], probe) => {
      const actual = pixels.slice(y * bytesPerRow + x * 4, y * bytesPerRow + x * 4 + 4);
      const reference = referencePixels[probe];
      const expected = [expectedRed[probe], 0, 0, 255];
      if (expected.some((value, channel) => Math.abs(reference[channel] - value) > 2)) {
        throw new Error(`Checkpoint 12 ${phase}: WebGL control at (${x},${y}) expected ${expected}, got ${reference}`);
      }
      if (expected.some((_, channel) => Math.abs(actual[channel] - reference[channel]) > 2)) {
        throw new Error(`Checkpoint 12 ${phase}: candidate at (${x},${y}) ${actual} differs from WebGL ${reference}`);
      }
    });
  }
  try {
    await sampleUpload12("first upload", [255, 0]);
    for (const { geometry } of pair12) {
      const position = geometry.attributes.position;
      for (let i = 0; i < position.array.length; i += 3) position.array[i] += 5;
    }
    const prepared12 = adapter.prepareMeshPacket(candidate12.mesh, camera12, width, height, wasmExports, options12);
    if (prepared12.snapshot.positions[0] !== candidate12.geometry.attributes.position.array[0]) {
      throw new Error("Checkpoint 12: pure preparation must still capture current CPU positions");
    }
    await sampleUpload12("unrequested CPU edits", [255, 0], "batch");
    for (const { geometry } of pair12) {
      geometry.attributes.position.addUpdateRange(0, 3);
      geometry.attributes.position.needsUpdate = true;
    }
    await sampleUpload12("partial upload preserves other vertices", [0, 255], "scene");
    for (const { geometry } of pair12) {
      if (geometry.attributes.position.updateRanges.length !== 0) {
        throw new Error("Checkpoint 12: submitted partial upload did not consume its update range");
      }
      geometry.attributes.position.needsUpdate = true;
    }
    await sampleUpload12("full upload", [0, 0]);
    for (const { geometry } of pair12) geometry.attributes.position.array.set(initialPositions12);
    await sampleUpload12("CPU restore stays stale before disposal", [0, 0], "batch");
    for (const { geometry } of pair12) geometry.dispose();
    await sampleUpload12("dispose and reuse refreshes unchanged version", [255, 0], "scene");

    // Independent position/index versions must survive switching render entry points.
    const indexedPositions12 = [...initialPositions12, ...initialPositions12.map((value, i) => i % 3 === 0 ? value + 5 : value)];
    for (const { geometry, library } of pair12) {
      geometry.dispose();
      geometry.setAttribute("position", new library.BufferAttribute(new Float32Array(indexedPositions12), 3));
      geometry.setIndex(new library.BufferAttribute(new Uint16Array([0, 1, 2]), 1));
    }
    await sampleUpload12("indexed first upload", [255, 0]);
    for (const { geometry } of pair12) {
      geometry.index.array.set([3, 4, 5]);
      geometry.attributes.position.needsUpdate = true;
    }
    await sampleUpload12("position upload leaves unrequested indices stale", [255, 0], "batch");
    for (const { geometry } of pair12) geometry.index.needsUpdate = true;
    await sampleUpload12("requested index upload", [0, 0], "scene");
    for (const { geometry } of pair12) {
      geometry.attributes.position.array.set(initialPositions12, 9);
      geometry.index.needsUpdate = true;
    }
    await sampleUpload12("index upload leaves unrequested positions stale", [0, 0]);
    for (const { geometry } of pair12) geometry.attributes.position.needsUpdate = true;
    await sampleUpload12("requested indexed position upload", [255, 0], "batch");
  } finally {
    for (const { geometry, mesh } of pair12) {
      geometry.dispose();
      mesh.material.dispose();
    }
    renderer12.dispose();
  }

  // ---------------------------------------------------------------------------
  // Checkpoint 13: Opaque MeshBasicMaterial vertexColors execution
  // Exercises f3d_build_mesh_batch_vertex_color_packet with distinct per-vertex
  // colors, non-white material multiplier, indexed drawRange, GPU-stale attribute
  // mutation without needsUpdate, updated output with needsUpdate, and independent
  // pinned Three.js WebGLRenderer reference oracle parity (offscreen + canvas).
  // ---------------------------------------------------------------------------
  const oracleThree13 = await loadProductionThree();
  const initialPositions13 = new Float32Array([
    -0.7, -0.7, -2.0, // V0 (bottom-left)
     0.7, -0.7, -2.0, // V1 (bottom-right)
     0.0,  0.7, -2.0, // V2 (top-center)
     2.0,  2.0, -2.0, // V3 (spare outside)
  ]);
  const initialIndices13 = new Uint16Array([3, 0, 1, 2]); // drawRange(1, 3) selects [0, 1, 2]
  const initialColors13 = new Float32Array([
    1.0, 0.0, 0.0, 0.0, // V0: Red; opaque material must still write alpha 1
    0.0, 1.0, 0.0, 0.25, // V1: Green
    0.0, 0.0, 1.0, 0.5, // V2: Blue
    1.0, 1.0, 1.0, 1.0, // V3: White
  ]);

  const makeVertexColorMesh13 = (library) => {
    const geometry = new library.BufferGeometry();
    geometry.setAttribute("position", new library.BufferAttribute(new Float32Array(initialPositions13), 3));
    geometry.setAttribute("color", new library.BufferAttribute(new Float32Array(initialColors13), 4));
    geometry.setIndex(new library.BufferAttribute(new Uint16Array(initialIndices13), 1));
    geometry.setDrawRange(1, 3); // Skip index 0 (V3), render triangle (V0, V1, V2)

    // Non-white material multiplier (0.5, 0.8, 0.4) strictly verifies that
    // fragment shader computes: diffuseColor.rgb = vColor.rgb * material.color.rgb
    const material = new library.MeshBasicMaterial({
      vertexColors: true,
      color: 0x80cc66,
      side: library.DoubleSide,
      depthTest: false,
      depthWrite: false,
    });
    material.color.setRGB(0.5, 0.8, 0.4);

    const mesh = new library.Mesh(geometry, material);
    mesh.frustumCulled = false;
    mesh.updateMatrixWorld(true);

    const scene = new library.Scene();
    scene.add(mesh);
    return { mesh, geometry, material, scene };
  };

  const cand13 = makeVertexColorMesh13(THREE);
  const ref13 = makeVertexColorMesh13(oracleThree13);

  const camera13 = new oracleThree13.OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
  camera13.position.set(0, 0, 0);
  camera13.lookAt(0, 0, -1);
  camera13.updateMatrixWorld(true);

  const renderer13 = new oracleThree13.WebGLRenderer({
    canvas: document.createElement("canvas"),
    antialias: false,
    preserveDrawingBuffer: true,
  });
  renderer13.setSize(width, height, false);
  renderer13.setClearColor(0x000000, 1);
  renderer13.toneMapping = oracleThree13.NoToneMapping;
  const gl13 = renderer13.getContext();

  // Selected probe locations inside the triangle:
  // - V0 region (bottom-left): (18, 48) -> Red dominates
  // - V1 region (bottom-right): (46, 48) -> Green dominates
  // - V2 region (top-center): (32, 18) -> Blue dominates
  // - Centroid: (32, 32) -> Mix of Red, Green, Blue
  const probes13 = [
    [18, 48],
    [46, 48],
    [32, 18],
    [32, 32],
  ];

  const sampleOracleProbes13 = () => {
    return probes13.map(([x, y]) => {
      const pixel = new Uint8Array(4);
      gl13.readPixels(x, height - 1 - y, 1, 1, gl13.RGBA, gl13.UNSIGNED_BYTE, pixel);
      return [pixel[0], pixel[1], pixel[2], pixel[3]];
    });
  };

  const sampleCandidateProbes13 = (pixels) => {
    return probes13.map(([x, y]) => {
      const idx = y * bytesPerRow + x * 4;
      return [
        pixels[idx + 0],
        pixels[idx + 1],
        pixels[idx + 2],
        pixels[idx + 3],
      ];
    });
  };

  try {
    // -------------------------------------------------------------------------
    // 13a: Frame 1 - Baseline execution against independent WebGLRenderer oracle
    // -------------------------------------------------------------------------
    renderer13.outputColorSpace = oracleThree13.LinearSRGBColorSpace;
    renderer13.render(ref13.scene, camera13);
    const refProbes1 = sampleOracleProbes13();

    await adapter.renderMeshBatch(bridgeHost, [cand13.mesh], camera13, null, wasmExports, {
      width,
      height,
      sourceBackend: "webgl",
    });
    const candPixels1 = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
    const candProbes1 = sampleCandidateProbes13(candPixels1);

    // Parity assertion against WebGLRenderer oracle (tolerance 2)
    for (let p = 0; p < probes13.length; p++) {
      const [x, y] = probes13[p];
      const cand = candProbes1[p];
      const ref = refProbes1[p];
      for (let c = 0; c < 4; c++) {
        if (Math.abs(cand[c] - ref[c]) > 2) {
          throw new Error(
            `Checkpoint 13a failed: Baseline probe at (${x}, ${y}) differs from WebGL reference: ` +
            `candidate [${cand}], reference [${ref}]`
          );
        }
      }
    }

    // Ensure the probe distinguishes the colored result from a uniform material.
    // If vertexColors were ignored, the color at (18, 48) would be pure material color [128, 204, 102].
    // At (18, 48), vertex is predominantly Red, so Green must strictly diverge from 204.
    const uncoloredGreen13 = 204;
    if (Math.abs(candProbes1[0][1] - uncoloredGreen13) < 50) {
      throw new Error(
        `Checkpoint 13a failed: Probe (18, 48) matched uncolored material color; ` +
        `vertexColors flag was ignored (green=${candProbes1[0][1]})`
      );
    }

    // Material color multiplication must remain observable.
    // If material color was ignored, Red at V0 probe would be ~255 instead of ~128 (0.5 * 255).
    if (candProbes1[0][0] > 200) {
      throw new Error(
        `Checkpoint 13a failed: Probe (18, 48) red channel is ${candProbes1[0][0]}, ` +
        `material color multiplier (0.5) was ignored`
      );
    }

    // -------------------------------------------------------------------------
    // 13b: Frame 2 - Persistent CPU color mutation without needsUpdate (stays GPU-stale)
    // -------------------------------------------------------------------------
    const mutatedColors13 = new Float32Array([
      0.0, 1.0, 1.0, 1.0, // V0 changed to Cyan [0, 1, 1, 1]
      1.0, 0.0, 1.0, 1.0, // V1 changed to Magenta [1, 0, 1, 1]
      1.0, 1.0, 0.0, 1.0, // V2 changed to Yellow [1, 1, 0, 1]
      0.0, 0.0, 0.0, 1.0, // V3: Black
    ]);
    // Mutate both independent arrays WITHOUT requesting either upload.
    cand13.geometry.attributes.color.array.set(mutatedColors13);
    ref13.geometry.attributes.color.array.set(mutatedColors13);
    renderer13.render(ref13.scene, camera13);
    const staleRefProbes13 = sampleOracleProbes13();
    if (JSON.stringify(staleRefProbes13) !== JSON.stringify(refProbes1)) {
      throw new Error("Checkpoint 13b failed: pinned reference did not retain stale color data");
    }

    await adapter.renderMeshBatch(bridgeHost, [cand13.mesh], camera13, null, wasmExports, {
      width,
      height,
      sourceBackend: "webgl",
    });
    const candPixels2 = await bridgeHost.readbackBuffer(20, bytesPerRow * height);

    // Frame 2 output MUST be byte-identical to Frame 1 (GPU memory remained stale)
    if (differs(candPixels2, candPixels1)) {
      throw new Error(
        "Checkpoint 13b failed: Color array mutation without needsUpdate=true altered GPU output; " +
        "persistent residency staleness guarantee violated"
      );
    }

    // -------------------------------------------------------------------------
    // 13c: Frame 3 - Upload only V0. CPU edits to V1/V2 must remain GPU-stale.
    // -------------------------------------------------------------------------
    for (const pair of [cand13, ref13]) {
      pair.geometry.attributes.color.addUpdateRange(0, 4);
      pair.geometry.attributes.color.needsUpdate = true;
    }

    renderer13.render(ref13.scene, camera13);
    const refProbes3 = sampleOracleProbes13();

    await adapter.renderMeshBatch(bridgeHost, [cand13.mesh], camera13, null, wasmExports, {
      width,
      height,
      sourceBackend: "webgl",
    });
    const candPixels3 = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
    const candProbes3 = sampleCandidateProbes13(candPixels3);

    // Assert strictly diverged from Frame 1 (Red at V0 probe went from ~128 to 0)
    if (Math.abs(candProbes3[0][0] - candProbes1[0][0]) < 50) {
      throw new Error(
        `Checkpoint 13c failed: Frame 3 probe (18, 48) red channel did not update after needsUpdate=true ` +
        `(was ${candProbes1[0][0]}, now ${candProbes3[0][0]})`
      );
    }

    // Assert match with updated WebGLRenderer oracle
    for (let p = 0; p < probes13.length; p++) {
      const [x, y] = probes13[p];
      const cand = candProbes3[p];
      const ref = refProbes3[p];
      for (let c = 0; c < 4; c++) {
        if (Math.abs(cand[c] - ref[c]) > 2) {
          throw new Error(
            `Checkpoint 13c failed: Frame 3 probe at (${x}, ${y}) differs from WebGL reference: ` +
            `candidate [${cand}], reference [${ref}]`
          );
        }
      }
    }

    // -------------------------------------------------------------------------
    // 13d: Normalized RGB byte attributes use the same source conversion boundary.
    // Replace the attribute so both renderers perform a fresh full upload.
    const byteColors13 = [128, 32, 240, 16, 192, 64, 224, 96, 48, 255, 255, 255];
    cand13.geometry.setAttribute("color", new THREE.BufferAttribute(new Uint8Array(byteColors13), 3, true));
    ref13.geometry.setAttribute("color", new oracleThree13.BufferAttribute(new Uint8Array(byteColors13), 3, true));
    renderer13.render(ref13.scene, camera13);
    const refByteProbes13 = sampleOracleProbes13();
    await adapter.renderMeshBatch(bridgeHost, [cand13.mesh], camera13, null, wasmExports, {
      width, height, sourceBackend: "webgl",
    });
    const bytePixels13 = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
    const byteProbes13 = sampleCandidateProbes13(bytePixels13);
    for (let p = 0; p < probes13.length; p++) {
      if (byteProbes13[p].some((value, channel) => Math.abs(value - refByteProbes13[p][channel]) > 2)) {
        throw new Error(`Checkpoint 13d failed: normalized RGB probe ${probes13[p]} differs: candidate ${byteProbes13[p]}, reference ${refByteProbes13[p]}`);
      }
    }

    // 13e: Canvas presentation output verification (sRGB transfer)
    // -------------------------------------------------------------------------
    if (canvasContext) {
      const canvasFormat13 = navigator.gpu.getPreferredCanvasFormat();
      canvasContext.configure({
        device,
        format: canvasFormat13,
        alphaMode: "opaque",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      });
      await nextFrame();

      const canvasReadback13 = device.createBuffer({
        size: bytesPerRow * height,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const currentTexture13 = canvasContext.getCurrentTexture();
      const renderPromise13 = adapter.renderScene(
        bridgeHost,
        cand13.scene,
        camera13,
        canvasContext,
        wasmExports,
        { width, height, sourceBackend: "webgl" }
      );
      const copyEncoder13 = device.createCommandEncoder();
      copyEncoder13.copyTextureToBuffer(
        { texture: currentTexture13 },
        { buffer: canvasReadback13, bytesPerRow },
        [width, height, 1]
      );
      device.queue.submit([copyEncoder13.finish()]);

      const sceneResult13 = await renderPromise13;
      if (sceneResult13.admitted.length !== 1 || sceneResult13.refused.length !== 0) {
        throw new Error(`Checkpoint 13e failed: Canvas scene render failed: ${JSON.stringify(sceneResult13)}`);
      }

      await canvasReadback13.mapAsync(GPUMapMode.READ);
      const canvasPixels13 = new Uint8Array(canvasReadback13.getMappedRange().slice(0));
      canvasReadback13.unmap();
      canvasReadback13.destroy();

      // Render WebGLRenderer reference to sRGB color space
      renderer13.outputColorSpace = oracleThree13.SRGBColorSpace;
      renderer13.render(ref13.scene, camera13);
      const refCanvasProbes = sampleOracleProbes13();

      const isBgra13 = canvasFormat13.startsWith("bgra");
      const rCh13 = isBgra13 ? 2 : 0;
      const gCh13 = 1;
      const bCh13 = isBgra13 ? 0 : 2;

      for (let p = 0; p < probes13.length; p++) {
        const [x, y] = probes13[p];
        const idx = y * bytesPerRow + x * 4;
        const candRgba = [
          canvasPixels13[idx + rCh13],
          canvasPixels13[idx + gCh13],
          canvasPixels13[idx + bCh13],
          canvasPixels13[idx + 3],
        ];
        const refRgba = refCanvasProbes[p];
        for (let c = 0; c < 4; c++) {
          if (Math.abs(candRgba[c] - refRgba[c]) > 2) {
            throw new Error(
              `Checkpoint 13e failed: Canvas probe at (${x}, ${y}) differs from sRGB reference: ` +
              `candidate [${candRgba}], reference [${refRgba}] (format=${canvasFormat13})`
            );
          }
        }
      }
    }
  } finally {
    cand13.geometry.dispose();
    cand13.material.dispose();
    ref13.geometry.dispose();
    ref13.material.dispose();
    renderer13.dispose();
  }

  // ---------------------------------------------------------------------------
  // Checkpoint 14: Interleaved buffer attributes execution
  // (1) Nonzero stride/offset, indexed draw, shared Float32 position + color holder
  // (2) Separate normalized Uint8 color holder
  // (3) Visible changes only at holder needsUpdate / updateRanges
  // (4) Sunny 19802 offset mutation retaining old binding, and fresh index/attribute rebinding
  // Compares against independent pinned Three.js WebGLRenderer reference oracle.
  // ---------------------------------------------------------------------------
  const oracleThree14 = await loadProductionThree();

  const initialInterleaved14 = new Float32Array([
    0.0, -0.7, -0.7, -2.0,  1.0, 0.0, 0.0, 1.0, // V0: pos (-0.7, -0.7, -2.0), Red
    0.0,  0.7, -0.7, -2.0,  0.0, 1.0, 0.0, 1.0, // V1: pos ( 0.7, -0.7, -2.0), Green
    0.0,  0.0,  0.7, -2.0,  0.0, 0.0, 1.0, 1.0, // V2: pos ( 0.0,  0.7, -2.0), Blue
    0.0,  2.0,  2.0, -2.0,  1.0, 1.0, 1.0, 1.0, // V3: spare outside
  ]);
  const initialIndices14 = new Uint16Array([3, 0, 1, 2]); // drawRange(1, 3) selects [0, 1, 2]

  const makeInterleavedMesh14 = (library) => {
    const ib = new library.InterleavedBuffer(new Float32Array(initialInterleaved14), 8);
    const posAttr = new library.InterleavedBufferAttribute(ib, 3, 1, false);
    const colAttr = new library.InterleavedBufferAttribute(ib, 4, 4, false);
    const geometry = new library.BufferGeometry();
    geometry.setAttribute("position", posAttr);
    geometry.setAttribute("color", colAttr);
    geometry.setIndex(new library.BufferAttribute(new Uint16Array(initialIndices14), 1));
    geometry.setDrawRange(1, 3);
    const material = new library.MeshBasicMaterial({
      vertexColors: true, side: library.DoubleSide, depthTest: false, depthWrite: false,
    });
    const mesh = new library.Mesh(geometry, material);
    mesh.frustumCulled = false;
    mesh.updateMatrixWorld(true);
    const scene = new library.Scene();
    scene.add(mesh);
    return { mesh, geometry, material, scene, ib, posAttr, colAttr };
  };

  const cand14 = makeInterleavedMesh14(THREE);
  const ref14 = makeInterleavedMesh14(oracleThree14);

  const camera14 = new oracleThree14.OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
  camera14.position.set(0, 0, 0);
  camera14.lookAt(0, 0, -1);
  camera14.updateMatrixWorld(true);

  const renderer14 = new oracleThree14.WebGLRenderer({
    canvas: document.createElement("canvas"),
    antialias: false,
    preserveDrawingBuffer: true,
  });
  renderer14.setSize(width, height, false);
  renderer14.setClearColor(0x000000, 1);
  renderer14.toneMapping = oracleThree14.NoToneMapping;
  renderer14.outputColorSpace = oracleThree14.LinearSRGBColorSpace;
  const gl14 = renderer14.getContext();

  const probes14 = [
    [18, 48], // V0 region (bottom-left): Red
    [46, 48], // V1 region (bottom-right): Green
    [32, 18], // V2 region (top-center): Blue
    [32, 32], // Centroid
  ];

  // Compact local render and probe sampling helper
  const renderAndSample = async (candMesh, refScene, probes) => {
    renderer14.render(refScene, camera14);
    const refPixels = probes.map(([x, y]) => {
      const p = new Uint8Array(4);
      gl14.readPixels(x, height - 1 - y, 1, 1, gl14.RGBA, gl14.UNSIGNED_BYTE, p);
      return [p[0], p[1], p[2], p[3]];
    });
    await adapter.renderMeshBatch(bridgeHost, [candMesh], camera14, null, wasmExports, {
      width, height, sourceBackend: "webgl",
    });
    const candBytes = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
    const candPixels = probes.map(([x, y]) => {
      const idx = y * bytesPerRow + x * 4;
      return [candBytes[idx], candBytes[idx + 1], candBytes[idx + 2], candBytes[idx + 3]];
    });
    return { refPixels, candPixels, candBytes };
  };

  const assertProbesMatch = (candPixels, refPixels, label, expected = null) => {
    const tolerance = expected ? 0 : 2;
    for (let p = 0; p < candPixels.length; p++) {
      for (let c = 0; c < 4; c++) {
        if (Math.abs(candPixels[p][c] - refPixels[p][c]) > tolerance) {
          throw new Error(`${label}: probe ${p} channel ${c} diverged: cand [${candPixels[p]}], ref [${refPixels[p]}]`);
        }
        if (expected && candPixels[p][c] !== expected[c]) {
          throw new Error(`${label}: probe ${p} channel ${c} expected [${expected}], got cand [${candPixels[p]}]`);
        }
        if (expected && refPixels[p][c] !== expected[c]) {
          throw new Error(`${label}: probe ${p} channel ${c} expected [${expected}], got ref [${refPixels[p]}]`);
        }
      }
    }
  };

  let candOffsetGeom, candOffsetMat, refOffsetGeom, refOffsetMat;
  try {
    // -------------------------------------------------------------------------
    // 14a: Nonzero stride/offset, indexed draw, shared Float32 pos+color holder
    // -------------------------------------------------------------------------
    const r14a = await renderAndSample(cand14.mesh, ref14.scene, probes14);
    assertProbesMatch(r14a.candPixels, r14a.refPixels, "Checkpoint 14a");
    if (r14a.candPixels[0][0] < 150 || r14a.candPixels[0][1] > 50) {
      throw new Error(`Checkpoint 14a failed: V0 probe did not show expected Red, got [${r14a.candPixels[0]}]`);
    }

    // -------------------------------------------------------------------------
    // 14b: Separate normalized Uint8 color holder on InterleavedBufferAttribute
    // -------------------------------------------------------------------------
    const u8ColorData = new Uint8Array([
      0, 0, 255,   0,   0, 255, // V0: offset 2 -> [255, 0, 0, 255] Red
      0, 0,   0, 255,   0, 255, // V1: offset 2 -> [0, 255, 0, 255] Green
      0, 0,   0,   0, 255, 255, // V2: offset 2 -> [0, 0, 255, 255] Blue
      0, 0, 255, 255, 255, 255, // V3: offset 2 -> [255, 255, 255, 255] White
    ]);
    const candU8Ib = new THREE.InterleavedBuffer(new Uint8Array(u8ColorData), 6);
    const refU8Ib = new oracleThree14.InterleavedBuffer(new Uint8Array(u8ColorData), 6);
    cand14.geometry.setAttribute("color", new THREE.InterleavedBufferAttribute(candU8Ib, 4, 2, true));
    ref14.geometry.setAttribute("color", new oracleThree14.InterleavedBufferAttribute(refU8Ib, 4, 2, true));

    const r14b = await renderAndSample(cand14.mesh, ref14.scene, probes14);
    assertProbesMatch(r14b.candPixels, r14b.refPixels, "Checkpoint 14b");

    // -------------------------------------------------------------------------
    // 14c: Visible changes only at holder needsUpdate / updateRanges
    // -------------------------------------------------------------------------
    // 14c-1: Mutate CPU array without needsUpdate -> remains GPU-stale
    candU8Ib.array[2] = 0; candU8Ib.array[3] = 255;
    refU8Ib.array[2] = 0; refU8Ib.array[3] = 255;
    const r14c1 = await renderAndSample(cand14.mesh, ref14.scene, probes14);
    assertProbesMatch(r14c1.candPixels, r14c1.refPixels, "Checkpoint 14c-1");
    if (differs(r14c1.candBytes, r14b.candBytes)) {
      throw new Error("Checkpoint 14c-1 failed: Un-bumped InterleavedBuffer edit altered GPU output without needsUpdate");
    }

    // 14c-2: Set updateRange and needsUpdate -> changes become visible (Cyan)
    candU8Ib.addUpdateRange(0, 6); candU8Ib.needsUpdate = true;
    refU8Ib.addUpdateRange(0, 6); refU8Ib.needsUpdate = true;
    const r14c2 = await renderAndSample(cand14.mesh, ref14.scene, probes14);
    assertProbesMatch(r14c2.candPixels, r14c2.refPixels, "Checkpoint 14c-2");
    if (r14c2.candPixels[0][0] > 50 || r14c2.candPixels[0][1] < 150) {
      throw new Error(`Checkpoint 14c-2 failed: V0 did not update to Cyan, got [${r14c2.candPixels[0]}]`);
    }

    // -------------------------------------------------------------------------
    // 14d: Sunny 19802 offset mutation retaining old binding, and fresh-index rebinding
    // -------------------------------------------------------------------------
    const offsetData = new Float32Array([
      -0.7, -0.7, -2.0,   4.3, -0.7, -2.0, // V0: centered @ offset 0, offscreen (x+5) @ offset 3
       0.7, -0.7, -2.0,   5.7, -0.7, -2.0, // V1
       0.0,  0.7, -2.0,   5.0,  0.7, -2.0, // V2
    ]);
    const candOffsetIb = new THREE.InterleavedBuffer(new Float32Array(offsetData), 6);
    const refOffsetIb = new oracleThree14.InterleavedBuffer(new Float32Array(offsetData), 6);
    const candOffsetPos = new THREE.InterleavedBufferAttribute(candOffsetIb, 3, 0, false);
    const refOffsetPos = new oracleThree14.InterleavedBufferAttribute(refOffsetIb, 3, 0, false);

    candOffsetGeom = new THREE.BufferGeometry();
    candOffsetGeom.setAttribute("position", candOffsetPos);
    candOffsetGeom.setIndex(new THREE.BufferAttribute(new Uint16Array([0, 1, 2]), 1));

    refOffsetGeom = new oracleThree14.BufferGeometry();
    refOffsetGeom.setAttribute("position", refOffsetPos);
    refOffsetGeom.setIndex(new oracleThree14.BufferAttribute(new Uint16Array([0, 1, 2]), 1));

    candOffsetMat = new THREE.MeshBasicMaterial({ color: 0xff0000, side: THREE.DoubleSide, depthTest: false, depthWrite: false });
    refOffsetMat = new oracleThree14.MeshBasicMaterial({ color: 0xff0000, side: oracleThree14.DoubleSide, depthTest: false, depthWrite: false });

    const candOffsetMesh = new THREE.Mesh(candOffsetGeom, candOffsetMat);
    candOffsetMesh.frustumCulled = false;
    const refOffsetMesh = new oracleThree14.Mesh(refOffsetGeom, refOffsetMat);
    refOffsetMesh.frustumCulled = false;

    const refOffsetScene = new oracleThree14.Scene();
    refOffsetScene.add(refOffsetMesh);

    const centerProbe = [[32, 32]];
    const redRGBA = [255, 0, 0, 255];
    const blackRGBA = [0, 0, 0, 255];

    // Frame 14d-1: Initial render at offset 0 (start indexed) -> centered Red triangle (all 4 RGBA channels)
    const r14d1 = await renderAndSample(candOffsetMesh, refOffsetScene, centerProbe);
    assertProbesMatch(r14d1.candPixels, r14d1.refPixels, "Checkpoint 14d-1 (baseline)", redRGBA);

    // Frame 14d-2: In-place attribute.offset mutation on the SAME attribute instance without rebinding.
    // Pinned WebGLBindingStates.js:149-192 caches VAO without polling attribute.offset.
    // Both pinned reference and candidate MUST retain stale pointer at offset 0 -> Red (all 4 RGBA channels)
    candOffsetPos.offset = 3;
    refOffsetPos.offset = 3;
    const r14d2 = await renderAndSample(candOffsetMesh, refOffsetScene, centerProbe);
    assertProbesMatch(r14d2.candPixels, r14d2.refPixels, "Checkpoint 14d-2 (stale offset)", redRGBA);

    // Frame 14d-3: Fresh index identity rebinding (replace only index with fresh same values).
    // WebGLBindingStates.js:188 (currentState.index !== index) invalidates VAO cache;
    // setupVertexAttributes re-reads position.offset=3 -> triangle moves offscreen -> Black (all 4 RGBA channels)
    candOffsetGeom.setIndex(new THREE.BufferAttribute(new Uint16Array([0, 1, 2]), 1));
    refOffsetGeom.setIndex(new oracleThree14.BufferAttribute(new Uint16Array([0, 1, 2]), 1));
    const r14d3 = await renderAndSample(candOffsetMesh, refOffsetScene, centerProbe);
    assertProbesMatch(r14d3.candPixels, r14d3.refPixels, "Checkpoint 14d-3 (fresh index rebind)", blackRGBA);

    // Frame 14d-4: A fresh position attribute at offset 0 must move the triangle back into view.
    candOffsetGeom.setAttribute("position", new THREE.InterleavedBufferAttribute(candOffsetIb, 3, 0, false));
    refOffsetGeom.setAttribute("position", new oracleThree14.InterleavedBufferAttribute(refOffsetIb, 3, 0, false));
    const r14d4 = await renderAndSample(candOffsetMesh, refOffsetScene, centerProbe);
    assertProbesMatch(r14d4.candPixels, r14d4.refPixels, "Checkpoint 14d-4 (fresh attribute rebind)", redRGBA);

    // 14e: Switching to a new source program captures the current layout; switching
    // back reuses that program's previous layout, even after material.needsUpdate.
    candOffsetGeom.attributes.position.offset = 3;
    refOffsetGeom.attributes.position.offset = 3;
    candOffsetGeom.setAttribute("color", new THREE.BufferAttribute(new Float32Array(9).fill(1), 3));
    refOffsetGeom.setAttribute("color", new oracleThree14.BufferAttribute(new Float32Array(9).fill(1), 3));
    candOffsetMat.vertexColors = true;
    refOffsetMat.vertexColors = true;
    candOffsetMat.needsUpdate = true;
    refOffsetMat.needsUpdate = true;
    const r14e1 = await renderAndSample(candOffsetMesh, refOffsetScene, centerProbe);
    assertProbesMatch(r14e1.candPixels, r14e1.refPixels, "Checkpoint 14e-1 (new colored program)", blackRGBA);
    candOffsetMat.vertexColors = false;
    refOffsetMat.vertexColors = false;
    candOffsetMat.needsUpdate = true;
    refOffsetMat.needsUpdate = true;
    const r14e2 = await renderAndSample(candOffsetMesh, refOffsetScene, centerProbe);
    assertProbesMatch(r14e2.candPixels, r14e2.refPixels, "Checkpoint 14e-2 (return to uncolored program)", redRGBA);
  } finally {
    candOffsetGeom?.dispose();
    candOffsetMat?.dispose();
    refOffsetGeom?.dispose();
    refOffsetMat?.dispose();
    cand14.geometry.dispose();
    cand14.material.dispose();
    ref14.geometry.dispose();
    ref14.material.dispose();
    renderer14.dispose();
  }

  // ---------------------------------------------------------------------------
  // Checkpoint 15: scene.background Color clear with opaque foreground mesh
  // Compares against independent pinned Three.js WebGLRenderer reference oracle.
  // ---------------------------------------------------------------------------
  const oracleThree15 = await loadProductionThree();
  const makeScene15 = (library) => {
    const geometry = new library.BufferGeometry();
    geometry.setAttribute(
      "position",
      new library.BufferAttribute(new Float32Array([-0.5, -0.5, -2, 0.5, -0.5, -2, 0, 0.5, -2]), 3)
    );
    const material = new library.MeshBasicMaterial({
      color: 0xff0000, side: library.DoubleSide, depthTest: false, depthWrite: false,
    });
    const mesh = new library.Mesh(geometry, material);
    mesh.frustumCulled = false;
    mesh.updateMatrixWorld(true);
    const scene = new library.Scene();
    scene.background = new library.Color(0.25, 0.5, 0.75);
    scene.add(mesh);
    return { scene, mesh, geometry, material };
  };
  const cand15 = makeScene15(THREE);
  const ref15 = makeScene15(oracleThree15);

  const camera15 = new oracleThree15.OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
  camera15.position.set(0, 0, 0);
  camera15.lookAt(0, 0, -1);
  camera15.updateMatrixWorld(true);

  const renderer15 = new oracleThree15.WebGLRenderer({
    canvas: document.createElement("canvas"),
    antialias: false,
    preserveDrawingBuffer: true,
  });
  renderer15.setSize(width, height, false);
  renderer15.toneMapping = oracleThree15.NoToneMapping;
  renderer15.outputColorSpace = oracleThree15.SRGBColorSpace;
  const gl15 = renderer15.getContext();

  const probes15 = [[8, 8], [32, 32]];
  const options15 = { width, height, sourceBackend: "webgl", outputColorSpace: "srgb" };

  const renderAndSample15 = async (candScene, refScene, expectedAdmitted = 1, cam = camera15) => {
    renderer15.render(refScene, cam);
    const refPixels = probes15.map(([x, y]) => {
      const p = new Uint8Array(4);
      gl15.readPixels(x, height - 1 - y, 1, 1, gl15.RGBA, gl15.UNSIGNED_BYTE, p);
      return [p[0], p[1], p[2], p[3]];
    });
    const res = await adapter.renderScene(bridgeHost, candScene, cam, null, wasmExports, options15);
    if (res.admitted.length !== expectedAdmitted || res.refused.length !== 0) {
      throw new Error(`Checkpoint 15 scene refused: expected ${expectedAdmitted} admitted, got ${JSON.stringify(res)}`);
    }
    const candBytes = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
    const candPixels = probes15.map(([x, y]) => {
      const idx = y * bytesPerRow + x * 4;
      return [candBytes[idx], candBytes[idx + 1], candBytes[idx + 2], candBytes[idx + 3]];
    });
    return { refPixels, candPixels, candBytes };
  };

  const makeBlueMesh15 = (library) => {
    const geometry = new library.BufferGeometry();
    geometry.setAttribute(
      "position",
      new library.BufferAttribute(new Float32Array([-0.5, -0.5, -2, 0.5, -0.5, -2, 0, 0.5, -2]), 3)
    );
    geometry.boundingSphere = new library.Sphere(new library.Vector3(10000, 0, 0), 1);
    const material = new library.MeshBasicMaterial({
      color: 0x0000ff, side: library.DoubleSide, depthTest: false, depthWrite: false,
    });
    const mesh = new library.Mesh(geometry, material);
    mesh.renderOrder = 1;
    mesh.updateMatrixWorld(true);
    return { mesh, geometry, material };
  };

  let candBlue, refBlue;
  let candNearGeom, candNearMat, refNearGeom, refNearMat;

  try {
    // Frame 15a: Initial midtone background Color(0.25, 0.5, 0.75) with opaque Red foreground
    const r15a = await renderAndSample15(cand15.scene, ref15.scene);
    assertProbesMatch(r15a.candPixels, r15a.refPixels, "Checkpoint 15a (initial background)");
    if (r15a.candPixels[1][0] < 200 || r15a.candPixels[1][1] > 20) {
      throw new Error(`Checkpoint 15a failed: Foreground mesh not present, got [${r15a.candPixels[1]}]`);
    }

    // Frame 15b: Dynamic background mutation to Color(0.75, 0.25, 0.5)
    cand15.scene.background.setRGB(0.75, 0.25, 0.5);
    ref15.scene.background.setRGB(0.75, 0.25, 0.5);
    const r15b = await renderAndSample15(cand15.scene, ref15.scene);
    if (!differs(r15b.candBytes, r15a.candBytes)) {
      throw new Error("Checkpoint 15b failed: candidate pixels did not change after background mutation");
    }
    assertProbesMatch(r15b.candPixels, r15b.refPixels, "Checkpoint 15b (mutated background)");
    if (Math.abs(r15b.candPixels[0][0] - r15a.candPixels[0][0]) < 30) {
      throw new Error("Checkpoint 15b failed: Background probe did not change between frames");
    }
    if (r15b.candPixels[1][0] < 200 || r15b.candPixels[1][1] > 20) {
      throw new Error(`Checkpoint 15b failed: Foreground mesh not present after mutation, got [${r15b.candPixels[1]}]`);
    }

    // Frame 15c: Add overlapping Blue mesh with distant boundingSphere and default frustumCulled=true
    candBlue = makeBlueMesh15(THREE);
    cand15.scene.add(candBlue.mesh);
    refBlue = makeBlueMesh15(oracleThree15);
    ref15.scene.add(refBlue.mesh);

    const r15c = await renderAndSample15(cand15.scene, ref15.scene, 1);
    assertProbesMatch(r15c.candPixels, r15c.refPixels, "Checkpoint 15c (frustum culled blue mesh)");
    if (r15c.candPixels[1][0] < 200 || r15c.candPixels[1][2] > 20) {
      throw new Error(`Checkpoint 15c failed: Expected red foreground, got [${r15c.candPixels[1]}]`);
    }

    // Frame 15d: Set frustumCulled=false on both -> bypasses culling, admits Blue mesh
    candBlue.mesh.frustumCulled = false;
    refBlue.mesh.frustumCulled = false;
    const r15d = await renderAndSample15(cand15.scene, ref15.scene, 2);
    assertProbesMatch(r15d.candPixels, r15d.refPixels, "Checkpoint 15d (frustumCulled=false admitted)");
    if (r15d.candPixels[1][2] < 200 || r15d.candPixels[1][0] > 20) {
      throw new Error(`Checkpoint 15d failed: Expected blue foreground, got [${r15d.candPixels[1]}]`);
    }

    // Frame 15e: Restore frustumCulled=true and center bounds at (0, 0, -2)
    candBlue.mesh.frustumCulled = true;
    refBlue.mesh.frustumCulled = true;
    candBlue.geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, -2), 1);
    refBlue.geometry.boundingSphere = new oracleThree15.Sphere(new oracleThree15.Vector3(0, 0, -2), 1);
    const r15e = await renderAndSample15(cand15.scene, ref15.scene, 2);
    assertProbesMatch(r15e.candPixels, r15e.refPixels, "Checkpoint 15e (re-centered boundingSphere admitted)");
    if (r15e.candPixels[1][2] < 200 || r15e.candPixels[1][0] > 20) {
      throw new Error(`Checkpoint 15e failed: Expected blue foreground after re-centering bounds, got [${r15e.candPixels[1]}]`);
    }

    // Frame 15f: Remove all foreground meshes -> expected admitted 0, center probe is background Color
    cand15.scene.remove(cand15.mesh);
    cand15.scene.remove(candBlue.mesh);
    ref15.scene.remove(ref15.mesh);
    ref15.scene.remove(refBlue.mesh);

    const r15f = await renderAndSample15(cand15.scene, ref15.scene, 0);
    assertProbesMatch(r15f.candPixels, r15f.refPixels, "Checkpoint 15f (empty scene background clear)");
    for (let c = 0; c < 4; c++) {
      if (Math.abs(r15f.candPixels[1][c] - r15f.candPixels[0][c]) > 2) {
        throw new Error(`Checkpoint 15f failed: Center probe diverged from background probe: center [${r15f.candPixels[1]}], corner [${r15f.candPixels[0]}]`);
      }
    }

    // Frame 15g: Mutate background Color with zero meshes -> candidate changed and matches reference
    cand15.scene.background.setRGB(0.2, 0.8, 0.4);
    ref15.scene.background.setRGB(0.2, 0.8, 0.4);
    const r15g = await renderAndSample15(cand15.scene, ref15.scene, 0);
    if (!differs(r15g.candBytes, r15f.candBytes)) {
      throw new Error("Checkpoint 15g failed: candidate pixels did not change after background mutation in empty scene");
    }
    assertProbesMatch(r15g.candPixels, r15g.refPixels, "Checkpoint 15g (mutated empty scene background)");
    if (Math.abs(r15g.candPixels[1][1] - r15f.candPixels[1][1]) < 30) {
      throw new Error("Checkpoint 15g failed: Center probe did not change after background mutation");
    }
    for (let c = 0; c < 4; c++) {
      if (Math.abs(r15g.candPixels[1][c] - r15g.candPixels[0][c]) > 2) {
        throw new Error(`Checkpoint 15g failed: Center probe diverged from background probe: center [${r15g.candPixels[1]}], corner [${r15g.candPixels[0]}]`);
      }
    }

    // Frame 15h: Canvas target empty scene clear with scene.background Color
    if (canvasContext) {
      const canvasFormat15 = navigator.gpu.getPreferredCanvasFormat();
      canvasContext.configure({
        device,
        format: canvasFormat15,
        alphaMode: "opaque",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      });
      await nextFrame();

      const canvasReadback15 = device.createBuffer({
        size: bytesPerRow * height,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const currentTexture15 = canvasContext.getCurrentTexture();
      const renderPromise15h = adapter.renderScene(
        bridgeHost,
        cand15.scene,
        camera15,
        canvasContext,
        wasmExports,
        options15
      );
      const copyEncoder15 = device.createCommandEncoder();
      copyEncoder15.copyTextureToBuffer(
        { texture: currentTexture15 },
        { buffer: canvasReadback15, bytesPerRow },
        [width, height, 1]
      );
      device.queue.submit([copyEncoder15.finish()]);

      const res15h = await renderPromise15h;
      if (res15h.admitted.length !== 0 || res15h.refused.length !== 0) {
        throw new Error(`Checkpoint 15h failed: Canvas empty scene refused: ${JSON.stringify(res15h)}`);
      }

      await canvasReadback15.mapAsync(GPUMapMode.READ);
      const canvasBytes15 = new Uint8Array(canvasReadback15.getMappedRange().slice(0));
      canvasReadback15.unmap();
      canvasReadback15.destroy();

      const isBgra15 = canvasFormat15.startsWith("bgra");
      const rCh15 = isBgra15 ? 2 : 0;
      const gCh15 = 1;
      const bCh15 = isBgra15 ? 0 : 2;

      const candCanvasPixels = probes15.map(([x, y]) => {
        const idx = y * bytesPerRow + x * 4;
        return [canvasBytes15[idx + rCh15], canvasBytes15[idx + gCh15], canvasBytes15[idx + bCh15], canvasBytes15[idx + 3]];
      });

      renderer15.render(ref15.scene, camera15);
      const refCanvasPixels = probes15.map(([x, y]) => {
        const p = new Uint8Array(4);
        gl15.readPixels(x, height - 1 - y, 1, 1, gl15.RGBA, gl15.UNSIGNED_BYTE, p);
        return [p[0], p[1], p[2], p[3]];
      });

      assertProbesMatch(candCanvasPixels, refCanvasPixels, "Checkpoint 15h (canvas empty scene background clear)");
      for (let c = 0; c < 4; c++) {
        if (Math.abs(candCanvasPixels[1][c] - candCanvasPixels[0][c]) > 2) {
          throw new Error(`Checkpoint 15h failed: Canvas center probe diverged from corner: center [${candCanvasPixels[1]}], corner [${candCanvasPixels[0]}]`);
        }
      }
    }

    // Frame 15i: Explicit WebGL near-plane clip convention with WebGPU-coordinate camera
    const nearCamera15 = new THREE.PerspectiveCamera(60, 1, 0.1, 10);
    nearCamera15.coordinateSystem = THREE.WebGPUCoordinateSystem;
    nearCamera15.updateProjectionMatrix();
    nearCamera15.updateMatrixWorld(true);

    const origElements15 = Array.from(nearCamera15.projectionMatrix.elements);
    const origCoordSystem15 = nearCamera15.coordinateSystem;

    // Tiny red triangle at z = -0.075 with x/y in [-0.003, 0.003]
    const nearPositions = new Float32Array([
      -0.003, -0.003, -0.075,
       0.003, -0.003, -0.075,
       0.000,  0.003, -0.075,
    ]);

    candNearGeom = new THREE.BufferGeometry();
    candNearGeom.setAttribute("position", new THREE.BufferAttribute(nearPositions, 3));
    candNearGeom.computeBoundingSphere();
    candNearMat = new THREE.MeshBasicMaterial({
      color: 0xff0000, side: THREE.DoubleSide, depthTest: false, depthWrite: false,
    });
    const candNearMesh = new THREE.Mesh(candNearGeom, candNearMat);
    candNearMesh.frustumCulled = true;
    candNearMesh.updateMatrixWorld(true);
    cand15.scene.add(candNearMesh);

    refNearGeom = new oracleThree15.BufferGeometry();
    refNearGeom.setAttribute("position", new oracleThree15.BufferAttribute(new Float32Array(nearPositions), 3));
    refNearGeom.computeBoundingSphere();
    refNearMat = new oracleThree15.MeshBasicMaterial({
      color: 0xff0000, side: oracleThree15.DoubleSide, depthTest: false, depthWrite: false,
    });
    const refNearMesh = new oracleThree15.Mesh(refNearGeom, refNearMat);
    refNearMesh.frustumCulled = true;
    refNearMesh.updateMatrixWorld(true);
    ref15.scene.add(refNearMesh);

    const r15i = await renderAndSample15(cand15.scene, ref15.scene, 1, nearCamera15);
    assertProbesMatch(r15i.candPixels, r15i.refPixels, "Checkpoint 15i (explicit webgl near-plane triangle)");
    if (r15i.candPixels[1][0] < 200 || r15i.candPixels[1][1] > 30 || r15i.candPixels[1][2] > 30) {
      throw new Error(`Checkpoint 15i failed: Expected red center pixel, got [${r15i.candPixels[1]}]`);
    }
    if (!differs(r15i.candBytes, r15g.candBytes)) {
      throw new Error("Checkpoint 15i failed: candidate pixels did not change when near triangle was added");
    }

    // Verify camera projection matrix and coordinateSystem metadata were not mutated
    for (let i = 0; i < 16; i++) {
      if (nearCamera15.projectionMatrix.elements[i] !== origElements15[i]) {
        throw new Error(`Checkpoint 15i failed: Camera projectionMatrix element ${i} mutated from ${origElements15[i]} to ${nearCamera15.projectionMatrix.elements[i]}`);
      }
    }
    if (nearCamera15.coordinateSystem !== origCoordSystem15) {
      throw new Error(`Checkpoint 15i failed: Camera coordinateSystem mutated from ${origCoordSystem15} to ${nearCamera15.coordinateSystem}`);
    }
  } finally {
    candNearGeom?.dispose();
    candNearMat?.dispose();
    refNearGeom?.dispose();
    refNearMat?.dispose();
    candBlue?.geometry.dispose();
    candBlue?.material.dispose();
    refBlue?.geometry.dispose();
    refBlue?.material.dispose();
    cand15.geometry.dispose();
    cand15.material.dispose();
    ref15.geometry.dispose();
    ref15.material.dispose();
    renderer15.dispose();
  }

  return "Variable-length multi-mesh batch verified (depth24plus): near-first/far-second with depthWrite=true produces near mesh (Green) matching independent direct WebGPU reference; near-first/far-second with depthWrite=false produces far mesh (Red) matching independent reference and strictly diverging from depthWrite=true; far-first/near-second with depthWrite=true produces near mesh (Green); immutable snapshots verified with distinct dynamic transforms and colors; negative controls strictly refuse empty batch (EMPTY_MESH_BATCH), mixed depth settings on legacy exports (INCOMPATIBLE_BATCH_DEPTH), and invisible meshes; retained WebGLRenderer multi-mesh oracle matches candidate within tolerance" + (canvasContext ? "; visible canvas batch verified against direct reference" : "") + "; scene hierarchy renderScene verified with translated Group, legitimate culls ignored, positive canvas execution (3 admitted, projected center sRGB colors, depth24plus occlusion, planted negative), and separate visible-unsupported whole-scene refusal; material side culling and reflected winding verified (FrontSide/BackSide/DoubleSide, CCW/CW, reflected det<0 parity, mixed-side multi-mesh batch, dynamic mutation between frames matching direct WebGPU reference); mixed per-mesh depth states verified (depthWrite on/off, Less/Greater depthFunc, interleaved disabled depthTest, reflection with depth, multi-frame depth mutation, canvas renderScene mixed depth); colorWrite=false invisible depth occluders verified (depthWrite on/off occlusion, mixed visible/invisible batch, dynamic mutation between frames, direct WebGPU reference parity with writeMask, canvas execution, DoubleSide single-mesh routing with new export alone); opaque MeshBasicMaterial vertexColors verified (distinct per-vertex colors, non-white material multiplier, indexed drawRange, WebGLRenderer reference parity, GPU-stale attribute mutation without needsUpdate, updated output with needsUpdate, offscreen and canvas sRGB); interleaved vertex attributes verified (shared Float32 position/color holder with nonzero stride and offset, indexed drawRange, normalized Uint8 color holder, GPU-stale CPU edits without needsUpdate, partial updateRanges, and persistent attribute offset mutation WebGLBindingStates parity); scene.background Color clear verified with opaque foreground mesh and dynamic background mutation against WebGLRenderer oracle; single-camera frustum culling verified with distant boundingSphere cull, frustumCulled=false opt-out, and re-centered boundingSphere admission matching WebGLRenderer oracle; empty scene with scene.background Color clear verified with zero admitted meshes, dynamic background mutation, and visible canvas clear matching WebGLRenderer oracle; explicit-WebGL near-plane clip convention verified with WebGPU-coordinate PerspectiveCamera and near triangle matching WebGLRenderer oracle";
}
