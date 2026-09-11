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
  } = options;

  const bytesPerRow = Math.ceil((width * 4) / 256) * 256;
  const target = device.createTexture({
    size: [width, height, 1],
    format,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
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

      @fragment
      fn fs_main() -> @location(0) vec4<f32> {
        return uniforms.color;
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
  });

  device.pushErrorScope("validation");
  let scopeOpen = true;
  try {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: target.createView(),
          loadOp: "clear",
          storeOp: "store",
          clearValue: [0, 0, 0, 1],
        },
      ],
    });
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
  mesh.material.depthTest = true;
  const depthAdmission = adapter.canAdmitMesh(mesh, camera);
  if (depthAdmission.admitted) {
    throw new Error("Negative control failed: depthTest=true was admitted without depth buffer");
  }
  mesh.material.depthTest = false;

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

    return "Visible canvas Three.js Mesh rendering matches independent direct WebGPU reference (preferredCanvasFormat) across rAF-separated presentation frames; translation and color mutations observed on swapchain; stale packet and absent canvasContext correctly rejected; Gray's prepareCanvasMeshPacket and renderMesh (with canvas pixel verification) verified directly";
  } finally {
    canvasReadback.destroy();
  }
}
