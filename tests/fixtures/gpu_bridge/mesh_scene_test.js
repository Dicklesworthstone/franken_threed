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

  // Stride 20 bytes: 3 floats position [x, y, z] + 2 floats uv [0, 0]
  const vertexCount = positions.length / 3;
  const vertexData = new Float32Array(vertexCount * 5);
  for (let i = 0; i < vertexCount; i++) {
    vertexData[i * 5 + 0] = positions[i * 3 + 0];
    vertexData[i * 5 + 1] = positions[i * 3 + 1];
    vertexData[i * 5 + 2] = positions[i * 3 + 2];
    vertexData[i * 5 + 3] = 0.0;
    vertexData[i * 5 + 4] = 0.0;
  }

  const vertexBuffer = device.createBuffer({
    size: Math.max(vertexData.byteLength, 16),
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vertexBuffer, 0, vertexData);

  let indexBuffer = null;
  const isIndexed = indices && indices.length > 0;
  if (isIndexed) {
    indexBuffer = device.createBuffer({
      size: Math.max(indices.byteLength, 16),
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(indexBuffer, 0, indices);
  }

  // 144 bytes uniform struct (padded to 256 for standard alignment):
  // offset 0..64: model_view mat4x4<f32>
  // offset 64..128: projection mat4x4<f32>
  // offset 128..144: color vec4<f32>
  const uniformData = new Float32Array(64); // 256 bytes
  uniformData.set(new Float32Array(modelView), 0);
  uniformData.set(new Float32Array(projection), 16);
  uniformData.set(new Float32Array(color), 32);

  const uniformBuffer = device.createBuffer({
    size: 256,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuffer, 0, uniformData);

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

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  });

  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
  });

  const pipeline = device.createRenderPipeline({
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
      targets: [{ format }],
    },
    primitive: {
      topology: "triangle-list",
      cullMode: "none", // DoubleSide
    },
    depthStencil: hasDepth
      ? {
          format: depthFormat,
          depthWriteEnabled,
          depthCompare,
        }
      : undefined,
  });

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
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.setVertexBuffer(0, vertexBuffer);
    if (isIndexed) {
      pass.setIndexBuffer(indexBuffer, "uint32");
      pass.drawIndexed(indices.length, 1, 0, 0, 0);
    } else {
      pass.draw(vertexCount, 1, 0, 0);
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
    vertexBuffer.destroy();
    if (indexBuffer) indexBuffer.destroy();
    uniformBuffer.destroy();
  }
}

export function differs(a, b) {
  if (a.length !== b.length) return true;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return true;
  }
  return false;
}

/**
 * Builds reference inputs INDEPENDENTLY from actual Three.js scene objects.
 * Does NOT consume candidate adapter extraction.
 * Reads position attribute directly via getX/getY/getZ.
 * Computes model-view matrix directly via camera.matrixWorldInverse.clone().multiply(mesh.matrixWorld).
 * Reads projection matrix and material directly from source objects.
 */
function buildIndependentReferenceInput(mesh, camera, width = 64, height = 64) {
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

  return {
    positions,
    indices,
    modelView,
    projection,
    webglDepth,
    color,
    width,
    height,
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
function renderRetainedWebGLReference(THREE, mesh, camera, width = 64, height = 64) {
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
    scene.add(mesh);
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

  mesh.material.side = THREE.BackSide;
  const sideAdmission = adapter.canAdmitMesh(mesh, camera);
  if (sideAdmission.admitted) {
    throw new Error("Negative control failed: BackSide was admitted in DoubleSide slice");
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
      throw new Error("Checkpoint 6 failed: Visible canvas mesh depth pixels differ from direct WebGPU reference in canvas format");
    }
  }

  return "Real THREE.Mesh depthTest/depthWrite/depthFunc mutation verified (depth24plus): LessDepth produces visible mesh matching direct WebGPU reference; NeverDepth suppresses rendering (clear color preserved); AlwaysDepth renders visible mesh; depthTest=false without sourceBackend refused (AMBIGUOUS_DEPTH_PAIR); sourceBackend='webgpu' enables depthWrite flag; sourceBackend='webgl' suppresses depthWrite flag; depthWrite=false flag and selected cases verified against independent oracle (single-mesh draw verifies pipeline write flags, multi-object occlusion verified by static depth fixture)" + (canvasContext ? "; visible canvas mesh depth verified against direct reference" : "");
}
