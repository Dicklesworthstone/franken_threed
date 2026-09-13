/**
 * bridge_runtime.js - Production Static WebGPU Host Bridge Runtime
 * 
 * F3D Bulk WebGPU Bridge implementation:
 * - Pre-device feature and limit negotiation
 * - Synchronous error-scope stack discipline: push, encode, finish, submit, pop synchronously; then await
 * - Bounds-checked binary coarse packet decoder (zero eval/new Function)
 * - Fresh canvas swapchain texture acquisition per render interval
 * - Per-use versioned slices for queue-ordering snapshot isolation (Red-A / Blue-B)
 */

export const PACKET_MAGIC = 0x50443346; // 'F3DP' in little-endian
export const PACKET_VERSION = 1;

export const OPCODE_CREATE_BUFFER = 1;
export const OPCODE_WRITE_BUFFER = 2;
export const OPCODE_CREATE_PIPELINE = 3;
export const OPCODE_RENDER_PASS = 4;
export const OPCODE_COPY_TEXTURE_TO_BUFFER = 5;
export const OPCODE_CREATE_TEXTURE = 6;
export const OPCODE_RECORD_BUNDLE = 7;
export const OPCODE_EXECUTE_BUNDLES = 8;
export const OPCODE_SET_VIEWPORT = 9;
export const OPCODE_SET_SCISSOR_RECT = 10;
export const OPCODE_SET_DRAW_PARAMETERS = 11;
export const OPCODE_CREATE_PIPELINE_DEPTH = 12;
export const OPCODE_RENDER_PASS_DEPTH = 13;
export const OPCODE_CREATE_PIPELINE_CULL = 14;
export const OPCODE_CREATE_PIPELINE_DEPTH_CULL = 15;
export const OPCODE_CREATE_PIPELINE_DEPTH_CULL_COLOR = 16;
export const OPCODE_WRITE_TEXTURE = 17;
export const OPCODE_CREATE_PIPELINE_TEXTURED = 18;
export const OPCODE_RECORD_BUNDLE_BATCH = 19;
export const OPCODE_COPY_BUFFER_TO_BUFFER = 20;
export const OPCODE_CREATE_COMPUTE_PIPELINE = 21;
export const OPCODE_DISPATCH_COMPUTE = 22;

export const BINDING_TYPE_UNIFORM = 0;
export const BINDING_TYPE_STORAGE_READ = 1;
export const BINDING_TYPE_STORAGE_READ_WRITE = 2;

export const SAMPLER_FILTER_NEAREST = 0;
export const SAMPLER_FILTER_LINEAR = 1;

export const SAMPLER_FILTER_NAMES = {
  0: "nearest",
  1: "linear",
};

export const ADDRESS_MODE_CLAMP_TO_EDGE = 0;
export const ADDRESS_MODE_REPEAT = 1;

export const ADDRESS_MODE_NAMES = {
  0: "clamp-to-edge",
  1: "repeat",
};

export const CULL_MODE_NAMES = {
  0: "none",
  1: "front",
  2: "back",
};

export const FRONT_FACE_NAMES = {
  0: "ccw",
  1: "cw",
};

export const TARGET_OFFSCREEN = 0;
export const TARGET_CANVAS = 1;

export const TARGET_FORMAT_PREFERRED_CANVAS = 0;
export const TARGET_FORMAT_BGRA8UNORM = 1;
export const TARGET_FORMAT_RGBA8UNORM = 2;

export const TEXTURE_FORMAT_BGRA8UNORM = 1;
export const TEXTURE_FORMAT_RGBA8UNORM = 2;
export const TEXTURE_FORMAT_DEPTH24PLUS = 3;
export const TEXTURE_FORMAT_DEPTH32FLOAT = 4;

export const DEPTH_COMPARE_NEVER = 1;
export const DEPTH_COMPARE_LESS = 2;
export const DEPTH_COMPARE_EQUAL = 3;
export const DEPTH_COMPARE_LESS_EQUAL = 4;
export const DEPTH_COMPARE_GREATER = 5;
export const DEPTH_COMPARE_NOT_EQUAL = 6;
export const DEPTH_COMPARE_GREATER_EQUAL = 7;
export const DEPTH_COMPARE_ALWAYS = 8;

export const DEPTH_COMPARE_NAMES = [
  null,
  "never",
  "less",
  "equal",
  "less-equal",
  "greater",
  "not-equal",
  "greater-equal",
  "always",
];

export const TEXTURE_USAGE_COPY_SRC = 1;
export const TEXTURE_USAGE_COPY_DST = 2;
export const TEXTURE_USAGE_TEXTURE_BINDING = 4;
export const TEXTURE_USAGE_STORAGE_BINDING = 8;
export const TEXTURE_USAGE_RENDER_ATTACHMENT = 16;

import { isDetached, ensureSafePacketBytes } from "./memory_transport.js";

export class WebGpuBridgeHost {
  /**
   * @param {Object} [options={}]
   * @param {boolean} [options.wrongImplSkipBundleStateReset=false] - vqa.7 wrong-implementation toggle for the bundle-then-direct counterexample; default off; never set by production paths.
   * @param {boolean} [options.wrongImplSkipReadbackDeviceCheck=false] - vqa.7 wrong-implementation toggle; default off.
   */
  constructor(options = {}) {
    this.adapter = null;
    this.device = null;
    this.capabilityRecord = null;
    this.deviceGeneration = 0;
    this.deviceRequest = 0;
    this.buffers = new Map();
    this.textures = new Map();
    this.pipelines = new Map();
    this.bindGroups = new Map();
    this.bundles = new Map();
    this.computePipelines = new Map();
    this.bufferEpochs = new Map();
    this.errorScopeActive = false;
    this.lastRenderTargetId = null;
    this.wrongImplSkipBundleStateReset = options.wrongImplSkipBundleStateReset === true;
    this.wrongImplSkipReadbackDeviceCheck = options.wrongImplSkipReadbackDeviceCheck === true;
  }

  clearDeviceResources() {
    this.buffers.clear();
    this.textures.clear();
    this.pipelines.clear();
    this.bindGroups.clear();
    this.bundles.clear();
    this.computePipelines.clear();
    this.bufferEpochs.clear();
    this.lastRenderTargetId = null;
  }

  /** Stop owned work before requesting destruction; device.lost is asynchronous. */
  destroyDevice() {
    // Also cancel a pending request that has not published a device yet.
    this.deviceRequest++;
    const device = this.device;
    this.clearDeviceResources();
    this.device = null;
    this.adapter = null;
    this.capabilityRecord = null;
    this.deviceGeneration++;
    device?.destroy();
  }

  /**
   * Negotiate features and limits before requesting a device.
   * If a required feature is missing or limits cannot be satisfied,
   * throws a structured error and does not create a device.
   */
  async negotiateAndCreateDevice(requiredProfile = {}) {
    const request = ++this.deviceRequest;
    if (!navigator.gpu) {
      throw new Error("WebGPU is not supported in this browser environment");
    }

    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: requiredProfile.powerPreference || "high-performance",
    });

    if (!adapter) {
      throw new Error("Failed to acquire a WebGPU adapter");
    }
    if (request !== this.deviceRequest) {
      throw new Error("WebGPU device request superseded by a newer request");
    }

    const adapterFeatures = new Set(adapter.features);
    const requiredFeatures = requiredProfile.requiredFeatures || [];

    // Verify all required features are supported
    for (const feat of requiredFeatures) {
      if (!adapterFeatures.has(feat)) {
        throw new Error(`Negotiation failed: required WebGPU feature '${feat}' is not supported by adapter`);
      }
    }

    // Verify and forward required limits
    const reqLimits = {
      ...(requiredProfile.minLimits || {}),
      ...(requiredProfile.requiredLimits || {}),
    };
    const adapterLimits = adapter.limits;
    const requiredLimits = {};

    for (const [key, requestedVal] of Object.entries(reqLimits)) {
      if (requestedVal === undefined || requestedVal === null) continue;
      const availableVal = adapterLimits[key];
      if (availableVal === undefined) {
        throw new Error(`Negotiation failed: unknown or unsupported WebGPU limit '${key}'`);
      }
      if (key.startsWith("min")) {
        if (availableVal > requestedVal) {
          throw new Error(`Negotiation failed: ${key} alignment mismatch (requested ${requestedVal}, adapter requires ${availableVal})`);
        }
      } else {
        if (availableVal < requestedVal) {
          throw new Error(`Negotiation failed: ${key} insufficient (requested ${requestedVal}, available ${availableVal})`);
        }
      }
      requiredLimits[key] = requestedVal;
    }

    // Request the device with only explicitly negotiated features and limits
    const deviceDescriptor = {
      requiredFeatures: requiredFeatures,
      requiredLimits: requiredLimits,
    };

    const device = await adapter.requestDevice(deviceDescriptor);
    if (request !== this.deviceRequest) {
      device.destroy();
      throw new Error("WebGPU device request superseded by a newer request");
    }

    // Assert the resulting device actually meets the requested profile
    try {
      const deviceLimits = device.limits;
      for (const [key, requestedVal] of Object.entries(requiredLimits)) {
        const actualVal = deviceLimits[key];
        if (key.startsWith("min")) {
          if (actualVal > requestedVal) {
            throw new Error(`Device limit verification failed: ${key} does not meet requested profile (requested ${requestedVal}, device has ${actualVal})`);
          }
        } else {
          if (actualVal < requestedVal) {
            throw new Error(`Device limit verification failed: ${key} does not meet requested profile (requested ${requestedVal}, device has ${actualVal})`);
          }
        }
      }
      for (const feat of requiredFeatures) {
        if (!device.features.has(feat)) {
          throw new Error(`Device feature verification failed: required WebGPU feature '${feat}' is not enabled on device`);
        }
      }
    } catch (error) {
      device.destroy();
      throw error;
    }

    // Attach uncapturederror listener
    device.addEventListener("uncapturederror", (event) => {
      console.error("[f3d-bridge] Uncaptured WebGPU error:", event.error);
    });

    // Record verified device capabilities
    let adapterInfo = { vendor: "unknown", architecture: "unknown", description: "unknown" };
    if (adapter.info) {
      adapterInfo = {
        vendor: adapter.info.vendor || "unknown",
        architecture: adapter.info.architecture || "unknown",
        description: adapter.info.description || "unknown",
      };
    }

    const capabilityRecord = {
      hostEnvironment: navigator.userAgent,
      webgpuSupported: true,
      adapter: adapterInfo,
      isFallbackAdapter: adapter.isFallbackAdapter || false,
      enabledFeatures: Array.from(device.features),
      limits: {
        maxTextureDimension2D: device.limits.maxTextureDimension2D,
        maxBufferSize: device.limits.maxBufferSize,
        maxBindGroups: device.limits.maxBindGroups,
        minUniformBufferOffsetAlignment: device.limits.minUniformBufferOffsetAlignment,
        minStorageBufferOffsetAlignment: device.limits.minStorageBufferOffsetAlignment,
      },
      adapterLimits: {
        maxTextureDimension2D: adapter.limits.maxTextureDimension2D,
        maxBufferSize: adapter.limits.maxBufferSize,
        maxBindGroups: adapter.limits.maxBindGroups,
        minUniformBufferOffsetAlignment: adapter.limits.minUniformBufferOffsetAlignment,
        minStorageBufferOffsetAlignment: adapter.limits.minStorageBufferOffsetAlignment,
      },
      requiredLimits: requiredLimits,
      preferredCanvasFormat: navigator.gpu.getPreferredCanvasFormat(),
    };

    // Publish only a fully verified device. Numeric IDs never retain residency
    // from the previous device, even when the new device reuses those IDs.
    this.publishDevice(adapter, device, capabilityRecord);

    return capabilityRecord;
  }

  publishDevice(adapter, device, capabilityRecord) {
    this.clearDeviceResources();
    this.adapter = adapter;
    this.device = device;
    this.capabilityRecord = capabilityRecord;
    const generation = ++this.deviceGeneration;
    device.lost.then(() => {
      // A late loss notification from a retired device must not clear its successor.
      if (this.device !== device || this.deviceGeneration !== generation) return;
      this.clearDeviceResources();
      this.device = null;
      this.adapter = null;
      this.capabilityRecord = null;
      this.deviceGeneration++;
    });
  }

  /**
   * vqa.7 test seam that deterministically replaces a live device; never called by production paths.
   *
   * @param {GPUAdapter} adapter
   * @param {GPUDevice} device
   */
  installDeviceForTest(adapter, device) {
    this.publishDevice(adapter, device, null);
  }

  /**
   * Enforces synchronous push -> execute -> pop error scope discipline.
   * Interleaving or awaiting inside syncAction is strictly detected and rejected.
   */
  async withErrorScopes(kinds, syncAction) {
    if (this.errorScopeActive) {
      throw new Error("Error-scope serialization violation: concurrent task attempted to interleave error scopes");
    }

    this.errorScopeActive = true;
    const device = this.device;
    if (!device) {
      this.errorScopeActive = false;
      throw new Error("Cannot execute error scopes: device not initialized");
    }

    // Push all requested scopes synchronously
    for (const kind of kinds) {
      device.pushErrorScope(kind);
    }

    let syncResult;
    let syncError = null;
    try {
      syncResult = syncAction();
      if (syncResult && typeof syncResult.then === "function") {
        throw new Error("Error-scope violation: syncAction returned a Promise. Awaits inside error-scope body are strictly forbidden");
      }
    } catch (err) {
      syncError = err;
    } finally {
      // Pop all scopes synchronously in reverse order
      const popPromises = [];
      for (let i = kinds.length - 1; i >= 0; i--) {
        popPromises.push(device.popErrorScope());
      }
      this.errorScopeActive = false;

      // If syncAction threw, await settled pops and re-throw
      if (syncError) {
        await Promise.allSettled(popPromises);
        throw syncError;
      }

      // Await error scope results after synchronous execution
      const errors = await Promise.all(popPromises);
      for (const err of errors) {
        if (err) {
          throw new Error(`WebGPU error scope reported error: ${err.message || err}`);
        }
      }
      if (this.device !== device) {
        throw new Error("WebGPU device changed before scoped work completed");
      }
    }

    return syncResult;
  }

  /**
   * Decodes and executes a coarse checked packet binary buffer.
   *
   * Invariant: Command recording, canvas texture view acquisition, commandEncoder.finish(),
   * and queue.submit() all execute SYNCHRONOUSLY inside the pushed error scopes.
   * Error scopes are popped immediately after submit() and awaited afterwards.
   * onSubmitted runs synchronously after those pops. It observes issued queue
   * effects, not GPU completion or successful asynchronous validation.
   * Zero eval / new Function.
   */
  async executePacket(packetBytes, canvasContext = null, onSubmitted = null) {
    if (onSubmitted !== null && typeof onSubmitted !== "function") {
      throw new TypeError("onSubmitted must be a function or null");
    }
    if (!this.device) {
      throw new Error("Device not initialized");
    }

    ensureSafePacketBytes(packetBytes);

    const headerLen = 16;
    if (packetBytes.byteLength < headerLen) {
      throw new Error("Packet buffer too small for header");
    }

    const dataView = new DataView(packetBytes.buffer, packetBytes.byteOffset, packetBytes.byteLength);

    const magic = dataView.getUint32(0, true);
    if (magic !== PACKET_MAGIC) {
      throw new Error(`Invalid packet magic: 0x${magic.toString(16)} (expected 0x${PACKET_MAGIC.toString(16)})`);
    }

    const version = dataView.getUint16(4, true);
    if (version !== PACKET_VERSION) {
      throw new Error(`Unsupported packet version: ${version}`);
    }

    const commandCount = dataView.getUint32(8, true);
    const dataLen = dataView.getUint32(12, true);

    const dataBlockStart = packetBytes.byteLength - dataLen;
    if (dataBlockStart < headerLen) {
      throw new Error("Malformed packet: data payload overlaps header");
    }

    const dataPayload = packetBytes.subarray(dataBlockStart);

    // Synchronous execution block inside error scopes
    let submitted = false;
    const completion = this.withErrorScopes(["validation", "out-of-memory"], () => {
      const commandEncoder = this.device.createCommandEncoder();
      const stagingBuffers = [];
      try {
      let cursor = headerLen;
      let currentPassEncoder = null;
      let currentPassTargetKey = null;
      let currentPassHasDepth = false;
      let frameCanvasView = null;
      let pendingDrawParameters = null;
      let passState = {
        pipelineId: null,
        uniformBufferId: null,
        dynamicOffset: null,
        vertexBufferId: null,
        boundPipelineId: null,
      };
      // Per-executePacket bind-group cache keyed by pipelineRecord -> (GPUBuffer | null) -> GPUBindGroup.
      // Descriptors use buffer offset 0 with dynamicOffset supplied at setBindGroup time.
      const packetBindGroups = new Map();

      const getCanvasView = () => {
        if (!frameCanvasView && canvasContext) {
          frameCanvasView = canvasContext.getCurrentTexture().createView();
        }
        return frameCanvasView;
      };

      const closeActivePass = () => {
        if (currentPassEncoder) {
          currentPassEncoder.end();
          currentPassEncoder = null;
          currentPassTargetKey = null;
          currentPassHasDepth = false;
          passState = {
            pipelineId: null,
            uniformBufferId: null,
            dynamicOffset: null,
            vertexBufferId: null,
            boundPipelineId: null,
          };
        }
      };

      for (let i = 0; i < commandCount; i++) {
        if (cursor + 2 > dataBlockStart) {
          throw new Error(`Truncated command at index ${i}: cursor exceeded command block`);
        }

        const opcode = dataView.getUint16(cursor, true);
        cursor += 2;
        if (pendingDrawParameters && opcode !== OPCODE_RENDER_PASS && opcode !== OPCODE_RENDER_PASS_DEPTH) {
          throw new Error("SetDrawParameters must immediately precede RenderPass or RenderPassDepth");
        }

        switch (opcode) {
          case OPCODE_CREATE_BUFFER: {
            closeActivePass();
            if (cursor + 12 > dataBlockStart) {
              throw new Error(`Truncated CREATE_BUFFER fields at command ${i}`);
            }
            const bufferId = dataView.getUint32(cursor, true);
            const size = dataView.getUint32(cursor + 4, true);
            const usage = dataView.getUint32(cursor + 8, true);
            cursor += 12;

            const alignedSize = Math.ceil(Math.max(size, 16) / 4) * 4;
            const buffer = this.device.createBuffer({
              size: alignedSize,
              usage: usage,
            });
            this.buffers.set(bufferId, buffer);
            break;
          }

          case OPCODE_WRITE_BUFFER: {
            closeActivePass();
            if (cursor + 16 > dataBlockStart) {
              throw new Error(`Truncated WRITE_BUFFER fields at command ${i}`);
            }
            const bufferId = dataView.getUint32(cursor, true);
            const offset = dataView.getUint32(cursor + 4, true);
            const dataOffset = dataView.getUint32(cursor + 8, true);
            const dataLength = dataView.getUint32(cursor + 12, true);
            cursor += 16;

            const buffer = this.buffers.get(bufferId);
            if (!buffer) {
              throw new Error(`WriteBuffer: unknown bufferId ${bufferId}`);
            }
            if (dataOffset + dataLength > dataPayload.byteLength) {
              throw new Error(`WriteBuffer: data slice out of bounds (offset ${dataOffset} + len ${dataLength} > payload ${dataPayload.byteLength})`);
            }

            const chunk = dataPayload.subarray(dataOffset, dataOffset + dataLength);
            if (isDetached(chunk)) {
              throw new Error("WriteBuffer: data slice is detached");
            }
            this.device.queue.writeBuffer(buffer, offset, chunk);
            break;
          }

          case OPCODE_CREATE_TEXTURE: {
            closeActivePass();
            if (cursor + 20 > dataBlockStart) {
              throw new Error(`Truncated CREATE_TEXTURE fields at command ${i}`);
            }
            const textureId = dataView.getUint32(cursor, true);
            const width = dataView.getUint32(cursor + 4, true);
            const height = dataView.getUint32(cursor + 8, true);
            const formatCode = dataView.getUint32(cursor + 12, true);
            const usage = dataView.getUint32(cursor + 16, true);
            cursor += 20;

            let format;
            let isDepth = false;
            if (formatCode === 1) {
              format = "bgra8unorm";
            } else if (formatCode === 2) {
              format = "rgba8unorm";
            } else if (formatCode === 3) {
              format = "depth24plus";
              isDepth = true;
            } else if (formatCode === 4) {
              format = "depth32float";
              isDepth = true;
            } else {
              throw new Error(`Invalid texture formatCode: ${formatCode}`);
            }

            const texture = this.device.createTexture({
              size: [width, height, 1],
              format: format,
              usage: usage,
            });
            this.textures.set(textureId, texture);
            if (!isDepth) {
              this.lastRenderTargetId = textureId;
            }
            break;
          }

          case OPCODE_WRITE_TEXTURE: {
            closeActivePass();
            if (cursor + 24 > dataBlockStart) {
              throw new Error(`Truncated WRITE_TEXTURE fields at command ${i}`);
            }
            const textureId = dataView.getUint32(cursor, true);
            const width = dataView.getUint32(cursor + 4, true);
            const height = dataView.getUint32(cursor + 8, true);
            const bytesPerRow = dataView.getUint32(cursor + 12, true);
            const dataOffset = dataView.getUint32(cursor + 16, true);
            const dataLength = dataView.getUint32(cursor + 20, true);
            cursor += 24;

            const texture = this.textures.get(textureId);
            if (!texture) {
              throw new Error(`WriteTexture: unknown textureId ${textureId}`);
            }

            const format = texture.format;
            if (format !== "rgba8unorm" && format !== "bgra8unorm") {
              throw new Error(`WriteTexture: unsupported texture format ${format}`);
            }

            if (width === 0 || height === 0) {
              throw new Error(`WriteTexture: invalid texture dimensions ${width}x${height}`);
            }
            const bytesPerTexel = 4;
            const minRowBytes = width * bytesPerTexel;
            if (bytesPerRow < minRowBytes) {
              throw new Error(`WriteTexture: bytesPerRow ${bytesPerRow} less than row width ${minRowBytes}`);
            }
            const minDataLength = height > 1 ? (height - 1) * bytesPerRow + minRowBytes : minRowBytes;
            if (dataLength < minDataLength) {
              throw new Error(`WriteTexture: dataLength ${dataLength} insufficient for dimensions ${width}x${height} and bytesPerRow ${bytesPerRow}`);
            }

            if (dataOffset + dataLength > dataPayload.byteLength) {
              throw new Error(`WriteTexture: data slice out of bounds (offset ${dataOffset} + len ${dataLength} > payload ${dataPayload.byteLength})`);
            }

            const chunk = dataPayload.subarray(dataOffset, dataOffset + dataLength);
            if (isDetached(chunk)) {
              throw new Error("WriteTexture: data slice is detached");
            }

            const alignedBytesPerRow = Math.ceil(minRowBytes / 256) * 256;
            const stagingBufferSize = Math.max(alignedBytesPerRow * height, 16);

            const stagingBuffer = this.device.createBuffer({
              size: stagingBufferSize,
              usage: GPUBufferUsage.COPY_SRC,
              mappedAtCreation: true,
            });
            stagingBuffers.push(stagingBuffer);
            const mapped = new Uint8Array(stagingBuffer.getMappedRange());
            for (let r = 0; r < height; r++) {
              const srcRowStart = r * bytesPerRow;
              const dstRowStart = r * alignedBytesPerRow;
              mapped.set(chunk.subarray(srcRowStart, srcRowStart + minRowBytes), dstRowStart);
            }
            stagingBuffer.unmap();

            commandEncoder.copyBufferToTexture(
              {
                buffer: stagingBuffer,
                offset: 0,
                bytesPerRow: alignedBytesPerRow,
                rowsPerImage: height,
              },
              { texture: texture },
              [width, height, 1]
            );
            break;
          }

          case OPCODE_CREATE_PIPELINE:
          case OPCODE_CREATE_PIPELINE_CULL: {
            closeActivePass();
            const hasCullFields = (opcode === OPCODE_CREATE_PIPELINE_CULL);
            const expectedHeaderSize = hasCullFields ? 40 : 32;
            if (cursor + expectedHeaderSize > dataBlockStart) {
              throw new Error(`Truncated CREATE_PIPELINE fields at command ${i}`);
            }
            const pipelineId = dataView.getUint32(cursor, true);
            const codeOffset = dataView.getUint32(cursor + 4, true);
            const codeLen = dataView.getUint32(cursor + 8, true);
            const formatCode = dataView.getUint32(cursor + 12, true);
            const hasVertexBuffer = dataView.getUint32(cursor + 16, true) === 1;
            const hasUniformBuffer = dataView.getUint32(cursor + 20, true) === 1;
            const explicitUniformSize = dataView.getUint32(cursor + 24, true);
            const explicitVertexStride = dataView.getUint32(cursor + 28, true);
            let cullModeCode = 0;
            let frontFaceCode = 0;
            if (hasCullFields) {
              cullModeCode = dataView.getUint32(cursor + 32, true);
              frontFaceCode = dataView.getUint32(cursor + 36, true);
            }
            cursor += expectedHeaderSize;

            if (cullModeCode > 2 || !(cullModeCode in CULL_MODE_NAMES)) {
              throw new Error(`CreatePipeline: invalid cull_mode ${cullModeCode}`);
            }
            const cullMode = CULL_MODE_NAMES[cullModeCode];

            if (frontFaceCode > 1 || !(frontFaceCode in FRONT_FACE_NAMES)) {
              throw new Error(`CreatePipeline: invalid front_face ${frontFaceCode}`);
            }
            const frontFace = FRONT_FACE_NAMES[frontFaceCode];

            if (codeOffset + codeLen > dataPayload.byteLength) {
              throw new Error(`CreatePipeline: shader code slice out of bounds (offset ${codeOffset} + len ${codeLen} > payload ${dataPayload.byteLength})`);
            }

            let format;
            if (formatCode === 0) {
              format = this.capabilityRecord?.preferredCanvasFormat || "bgra8unorm";
            } else if (formatCode === 1) {
              format = "bgra8unorm";
            } else if (formatCode === 2) {
              format = "rgba8unorm";
            } else {
              throw new Error(`Invalid pipeline target formatCode: ${formatCode}`);
            }

            const codeBytes = dataPayload.subarray(codeOffset, codeOffset + codeLen);
            const shaderCode = new TextDecoder().decode(codeBytes);

            const shaderModule = this.device.createShaderModule({ code: shaderCode });

            const uniformSize = explicitUniformSize > 0 ? explicitUniformSize : (hasUniformBuffer ? 48 : 0);

            let bindGroupLayout = null;
            if (hasUniformBuffer) {
              bindGroupLayout = this.device.createBindGroupLayout({
                entries: [
                  {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    buffer: {
                      type: "uniform",
                      hasDynamicOffset: true,
                      minBindingSize: uniformSize,
                    },
                  },
                ],
              });
            }

            const pipelineLayout = this.device.createPipelineLayout({
              bindGroupLayouts: bindGroupLayout ? [bindGroupLayout] : [],
            });

            // Default layout matches f3d_core::layout::VERTEX_POS_UV_STRIDE (20 bytes: pos vec3<f32> at 0 + uv vec2<f32> at 12)
            const vertexStride = explicitVertexStride > 0 ? explicitVertexStride : 20;
            const vertexBuffers = hasVertexBuffer
              ? [
                  {
                    arrayStride: vertexStride,
                    attributes: [
                      { shaderLocation: 0, offset: 0, format: "float32x3" },
                      { shaderLocation: 1, offset: 12, format: vertexStride === 28 ? "float32x4" : "float32x2" },
                    ],
                  },
                ]
              : [];

            const pipeline = this.device.createRenderPipeline({
              layout: pipelineLayout,
              vertex: {
                module: shaderModule,
                entryPoint: "vs_main",
                buffers: vertexBuffers,
              },
              fragment: {
                module: shaderModule,
                entryPoint: "fs_main",
                targets: [{ format: format }],
              },
              primitive: {
                topology: "triangle-list",
                cullMode,
                frontFace,
              },
            });

            this.pipelines.set(pipelineId, { pipeline, bindGroupLayout, hasUniformBuffer, uniformSize, hasDepth: false });
            break;
          }

          case OPCODE_CREATE_PIPELINE_DEPTH:
          case OPCODE_CREATE_PIPELINE_DEPTH_CULL:
          case OPCODE_CREATE_PIPELINE_DEPTH_CULL_COLOR: {
            closeActivePass();
            const hasColorFields = (opcode === OPCODE_CREATE_PIPELINE_DEPTH_CULL_COLOR);
            const hasCullFields = (opcode === OPCODE_CREATE_PIPELINE_DEPTH_CULL || hasColorFields);
            const expectedHeaderSize = hasColorFields ? 56 : (hasCullFields ? 52 : 44);
            if (cursor + expectedHeaderSize > dataBlockStart) {
              throw new Error(`Truncated CREATE_PIPELINE_DEPTH fields at command ${i}`);
            }
            const pipelineId = dataView.getUint32(cursor, true);
            const codeOffset = dataView.getUint32(cursor + 4, true);
            const codeLen = dataView.getUint32(cursor + 8, true);
            const formatCode = dataView.getUint32(cursor + 12, true);
            const hasVertexBuffer = dataView.getUint32(cursor + 16, true) === 1;
            const hasUniformBuffer = dataView.getUint32(cursor + 20, true) === 1;
            const explicitUniformSize = dataView.getUint32(cursor + 24, true);
            const explicitVertexStride = dataView.getUint32(cursor + 28, true);
            const depthFormatCode = dataView.getUint32(cursor + 32, true);
            const depthWriteEnabled = dataView.getUint32(cursor + 36, true) === 1;
            const depthCompareCode = dataView.getUint32(cursor + 40, true);
            let cullModeCode = 0;
            let frontFaceCode = 0;
            let writeMask = 0xF;
            if (hasCullFields) {
              cullModeCode = dataView.getUint32(cursor + 44, true);
              frontFaceCode = dataView.getUint32(cursor + 48, true);
            }
            if (hasColorFields) {
              writeMask = dataView.getUint32(cursor + 52, true);
            }
            cursor += expectedHeaderSize;

            if (cullModeCode > 2 || !(cullModeCode in CULL_MODE_NAMES)) {
              throw new Error(`CreatePipelineDepth: invalid cull_mode ${cullModeCode}`);
            }
            const cullMode = CULL_MODE_NAMES[cullModeCode];

            if (frontFaceCode > 1 || !(frontFaceCode in FRONT_FACE_NAMES)) {
              throw new Error(`CreatePipelineDepth: invalid front_face ${frontFaceCode}`);
            }
            const frontFace = FRONT_FACE_NAMES[frontFaceCode];

            if (codeOffset + codeLen > dataPayload.byteLength) {
              throw new Error(`CreatePipelineDepth: shader code slice out of bounds (offset ${codeOffset} + len ${codeLen} > payload ${dataPayload.byteLength})`);
            }

            let format;
            if (formatCode === 0) {
              format = this.capabilityRecord?.preferredCanvasFormat || "bgra8unorm";
            } else if (formatCode === 1) {
              format = "bgra8unorm";
            } else if (formatCode === 2) {
              format = "rgba8unorm";
            } else {
              throw new Error(`Invalid pipeline target formatCode: ${formatCode}`);
            }

            let depthFormat;
            if (depthFormatCode === 3) {
              depthFormat = "depth24plus";
            } else if (depthFormatCode === 4) {
              depthFormat = "depth32float";
            } else {
              throw new Error(`CreatePipelineDepth: invalid depth formatCode ${depthFormatCode}`);
            }

            if (depthCompareCode < 1 || depthCompareCode > 8) {
              throw new Error(`CreatePipelineDepth: invalid depth_compare code ${depthCompareCode}`);
            }
            const depthCompare = DEPTH_COMPARE_NAMES[depthCompareCode];

            const codeBytes = dataPayload.subarray(codeOffset, codeOffset + codeLen);
            const shaderCode = new TextDecoder().decode(codeBytes);

            const shaderModule = this.device.createShaderModule({ code: shaderCode });

            const uniformSize = explicitUniformSize > 0 ? explicitUniformSize : (hasUniformBuffer ? 48 : 0);

            let bindGroupLayout = null;
            if (hasUniformBuffer) {
              bindGroupLayout = this.device.createBindGroupLayout({
                entries: [
                  {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    buffer: {
                      type: "uniform",
                      hasDynamicOffset: true,
                      minBindingSize: uniformSize,
                    },
                  },
                ],
              });
            }

            const pipelineLayout = this.device.createPipelineLayout({
              bindGroupLayouts: bindGroupLayout ? [bindGroupLayout] : [],
            });

            const vertexStride = explicitVertexStride > 0 ? explicitVertexStride : 20;
            const vertexBuffers = hasVertexBuffer
              ? [
                  {
                    arrayStride: vertexStride,
                    attributes: [
                      { shaderLocation: 0, offset: 0, format: "float32x3" },
                      { shaderLocation: 1, offset: 12, format: vertexStride === 28 ? "float32x4" : "float32x2" },
                    ],
                  },
                ]
              : [];

            const pipeline = this.device.createRenderPipeline({
              layout: pipelineLayout,
              vertex: {
                module: shaderModule,
                entryPoint: "vs_main",
                buffers: vertexBuffers,
              },
              fragment: {
                module: shaderModule,
                entryPoint: "fs_main",
                targets: [{ format: format, writeMask: writeMask }],
              },
              primitive: {
                topology: "triangle-list",
                cullMode,
                frontFace,
              },
              depthStencil: {
                format: depthFormat,
                depthWriteEnabled: depthWriteEnabled,
                depthCompare: depthCompare,
              },
            });

            this.pipelines.set(pipelineId, {
              pipeline,
              bindGroupLayout,
              hasUniformBuffer,
              uniformSize,
              hasDepth: true,
              depthFormat,
              depthWriteEnabled,
              depthCompare,
              cullMode,
              frontFace,
              writeMask,
            });
            break;
          }

          case OPCODE_CREATE_PIPELINE_TEXTURED: {
            closeActivePass();
            if (cursor + 44 > dataBlockStart) {
              throw new Error(`Truncated CREATE_PIPELINE_TEXTURED fields at command ${i}`);
            }
            const pipelineId = dataView.getUint32(cursor, true);
            const codeOffset = dataView.getUint32(cursor + 4, true);
            const codeLen = dataView.getUint32(cursor + 8, true);
            const formatCode = dataView.getUint32(cursor + 12, true);
            const hasVertexBuffer = dataView.getUint32(cursor + 16, true) === 1;
            const hasUniformBuffer = dataView.getUint32(cursor + 20, true) === 1;
            const explicitUniformSize = dataView.getUint32(cursor + 24, true);
            const explicitVertexStride = dataView.getUint32(cursor + 28, true);
            const textureId = dataView.getUint32(cursor + 32, true);
            const samplerFilterCode = dataView.getUint32(cursor + 36, true);
            const addressModeCode = dataView.getUint32(cursor + 40, true);
            cursor += 44;

            if (samplerFilterCode > 1 || !(samplerFilterCode in SAMPLER_FILTER_NAMES)) {
              throw new Error(`CreatePipelineTextured: invalid sampler_filter ${samplerFilterCode}`);
            }
            const samplerFilter = SAMPLER_FILTER_NAMES[samplerFilterCode];

            if (addressModeCode > 1 || !(addressModeCode in ADDRESS_MODE_NAMES)) {
              throw new Error(`CreatePipelineTextured: invalid address_mode ${addressModeCode}`);
            }
            const addressMode = ADDRESS_MODE_NAMES[addressModeCode];

            const texture = this.textures.get(textureId);
            if (!texture) {
              throw new Error(`CreatePipelineTextured: unknown textureId ${textureId}`);
            }

            if (codeOffset + codeLen > dataPayload.byteLength) {
              throw new Error(`CreatePipelineTextured: shader code slice out of bounds (offset ${codeOffset} + len ${codeLen} > payload ${dataPayload.byteLength})`);
            }

            let format;
            if (formatCode === 0) {
              format = this.capabilityRecord?.preferredCanvasFormat || "bgra8unorm";
            } else if (formatCode === 1) {
              format = "bgra8unorm";
            } else if (formatCode === 2) {
              format = "rgba8unorm";
            } else {
              throw new Error(`Invalid pipeline target formatCode: ${formatCode}`);
            }

            const codeBytes = dataPayload.subarray(codeOffset, codeOffset + codeLen);
            const shaderCode = new TextDecoder().decode(codeBytes);

            const shaderModule = this.device.createShaderModule({ code: shaderCode });

            const uniformSize = explicitUniformSize > 0 ? explicitUniformSize : (hasUniformBuffer ? 48 : 0);

            const sampler = this.device.createSampler({
              magFilter: samplerFilter,
              minFilter: samplerFilter,
              addressModeU: addressMode,
              addressModeV: addressMode,
            });

            const bindGroupLayoutEntries = [];
            if (hasUniformBuffer) {
              bindGroupLayoutEntries.push({
                binding: 0,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                buffer: {
                  type: "uniform",
                  hasDynamicOffset: true,
                  minBindingSize: uniformSize,
                },
              });
            }
            bindGroupLayoutEntries.push(
              {
                binding: 1,
                visibility: GPUShaderStage.FRAGMENT,
                texture: {
                  sampleType: "float",
                  viewDimension: "2d",
                  multisampled: false,
                },
              },
              {
                binding: 2,
                visibility: GPUShaderStage.FRAGMENT,
                sampler: {
                  type: "filtering",
                },
              }
            );

            const bindGroupLayout = this.device.createBindGroupLayout({
              entries: bindGroupLayoutEntries,
            });

            const pipelineLayout = this.device.createPipelineLayout({
              bindGroupLayouts: [bindGroupLayout],
            });

            const vertexStride = explicitVertexStride > 0 ? explicitVertexStride : 20;
            const vertexBuffers = hasVertexBuffer
              ? [
                  {
                    arrayStride: vertexStride,
                    attributes: [
                      { shaderLocation: 0, offset: 0, format: "float32x3" },
                      { shaderLocation: 1, offset: 12, format: vertexStride === 28 ? "float32x4" : "float32x2" },
                    ],
                  },
                ]
              : [];

            const pipeline = this.device.createRenderPipeline({
              layout: pipelineLayout,
              vertex: {
                module: shaderModule,
                entryPoint: "vs_main",
                buffers: vertexBuffers,
              },
              fragment: {
                module: shaderModule,
                entryPoint: "fs_main",
                targets: [{ format: format }],
              },
              primitive: {
                topology: "triangle-list",
                cullMode: "none",
                frontFace: "ccw",
              },
            });

            this.pipelines.set(pipelineId, {
              pipeline,
              bindGroupLayout,
              hasUniformBuffer,
              uniformSize,
              hasDepth: false,
              isTextured: true,
              texture,
              sampler,
              textureView: texture.createView(),
            });
            break;
          }

          case OPCODE_RENDER_PASS: {
            const hasDrawParameters = pendingDrawParameters !== null;
            const drawParameters = pendingDrawParameters || [1, 0, 0];
            pendingDrawParameters = null;
            if (cursor + 44 > dataBlockStart) {
              throw new Error(`Truncated RENDER_PASS fields at command ${i}`);
            }
            const rawTargetType = dataView.getUint32(cursor, true);
            const targetKind = rawTargetType & 0xFF;
            if (targetKind !== TARGET_OFFSCREEN && targetKind !== TARGET_CANVAS) {
              throw new Error(`RenderPass: invalid target kind ${targetKind} in 0x${rawTargetType.toString(16)}`);
            }
            const loadOpCode = (rawTargetType >> 8) & 0xFF;
            if (loadOpCode > 2) {
              throw new Error(`RenderPass: invalid load_op code ${loadOpCode}`);
            }
            const storeOpCode = (rawTargetType >> 16) & 0xFF;
            if (storeOpCode > 1) {
              throw new Error(`RenderPass: invalid store_op code ${storeOpCode}`);
            }
            const passFlags = (rawTargetType >> 24) & 0xFF;
            if ((passFlags & ~1) !== 0) {
              throw new Error(`RenderPass: unknown pass flags 0x${passFlags.toString(16)}`);
            }

            const targetId = dataView.getUint32(cursor + 4, true);
            const cr = dataView.getFloat32(cursor + 8, true);
            const cg = dataView.getFloat32(cursor + 12, true);
            const cb = dataView.getFloat32(cursor + 16, true);
            const ca = dataView.getFloat32(cursor + 20, true);
            const pipelineId = dataView.getUint32(cursor + 24, true);
            const vertexBufferId = dataView.getUint32(cursor + 28, true);
            const vertexCount = dataView.getUint32(cursor + 32, true);
            const dynamicOffset = dataView.getUint32(cursor + 36, true);
            const uniformBufferId = dataView.getUint32(cursor + 40, true) || 1;
            cursor += 44;

            if (targetKind === TARGET_OFFSCREEN) {
              this.lastRenderTargetId = targetId;
            }

            const isNewPass = (passFlags & 1) !== 0;
            const targetKey = `${targetKind}:${targetId}:none`;
            if (hasDrawParameters && (isNewPass || currentPassTargetKey !== targetKey)) {
              throw new Error("SetDrawParameters cannot cross a render-pass boundary");
            }
            if (!currentPassEncoder || currentPassTargetKey !== targetKey || isNewPass) {
              closeActivePass();

              let targetView;
              if (targetKind === TARGET_CANVAS) {
                if (!canvasContext) {
                  throw new Error("RenderPass: canvas target requires a canvas context");
                }
                // Canvas swapchain view is acquired at most once per packet execution (§8.5, [S48])
                targetView = getCanvasView();
              } else if (targetKind === TARGET_OFFSCREEN) {
                const texture = this.textures.get(targetId);
                if (!texture) {
                  throw new Error(`RenderPass: unknown offscreen targetId ${targetId}`);
                }
                targetView = texture.createView();
              } else {
                throw new Error(`Invalid render pass targetKind: ${targetKind}`);
              }

              const loadOp = loadOpCode === 1 ? "load" : "clear";
              const storeOp = storeOpCode === 1 ? "discard" : "store";
              const colorAttachmentDesc = {
                view: targetView,
                loadOp,
                storeOp,
              };
              if (loadOp === "clear") {
                colorAttachmentDesc.clearValue = { r: cr, g: cg, b: cb, a: ca };
              }

              currentPassEncoder = commandEncoder.beginRenderPass({
                colorAttachments: [colorAttachmentDesc],
              });
              currentPassTargetKey = targetKey;
              currentPassHasDepth = false;
              passState = {
                pipelineId: null,
                uniformBufferId: null,
                dynamicOffset: null,
                vertexBufferId: null,
                boundPipelineId: null,
              };
            }

            if (vertexCount > 0) {
              const pipelineRecord = this.pipelines.get(pipelineId);
              if (!pipelineRecord) {
                throw new Error(`RenderPass: unknown pipelineId ${pipelineId}`);
              }

              if (passState.pipelineId !== pipelineId) {
                currentPassEncoder.setPipeline(pipelineRecord.pipeline);
                passState.pipelineId = pipelineId;
                // The new pipeline may require a different layout or binding size
                // even when the underlying buffer and dynamic offset are unchanged.
                passState.uniformBufferId = null;
                passState.dynamicOffset = null;
                passState.boundPipelineId = null;
              }

              const hasUniform = pipelineRecord.hasUniformBuffer;
              const isTextured = Boolean(pipelineRecord.isTextured || (pipelineRecord.texture && pipelineRecord.sampler));
              if (hasUniform || isTextured) {
                const needsRebind = hasUniform
                  ? (passState.uniformBufferId !== uniformBufferId || passState.dynamicOffset !== dynamicOffset || passState.boundPipelineId !== pipelineId)
                  : (passState.boundPipelineId !== pipelineId);
                if (needsRebind) {
                  let uniformBuf = null;
                  if (hasUniform) {
                    uniformBuf = this.buffers.get(uniformBufferId);
                    if (!uniformBuf) {
                      throw new Error(`RenderPass: uniform buffer ${uniformBufferId} missing for pipeline`);
                    }
                  }

                  let pipelineBindGroups = packetBindGroups.get(pipelineRecord);
                  if (!pipelineBindGroups) {
                    pipelineBindGroups = new Map();
                    packetBindGroups.set(pipelineRecord, pipelineBindGroups);
                  }

                  let bindGroup = pipelineBindGroups.get(uniformBuf);
                  if (!bindGroup) {
                    const bindGroupEntries = [];
                    if (hasUniform) {
                      bindGroupEntries.push({
                        binding: 0,
                        resource: {
                          buffer: uniformBuf,
                          offset: 0,
                          size: pipelineRecord.uniformSize || 48,
                        },
                      });
                    }
                    if (isTextured) {
                      bindGroupEntries.push(
                        {
                          binding: 1,
                          resource: pipelineRecord.textureView || pipelineRecord.texture.createView(),
                        },
                        {
                          binding: 2,
                          resource: pipelineRecord.sampler,
                        }
                      );
                    }
                    bindGroup = this.device.createBindGroup({
                      layout: pipelineRecord.bindGroupLayout,
                      entries: bindGroupEntries,
                    });
                    pipelineBindGroups.set(uniformBuf, bindGroup);
                  }

                  if (hasUniform) {
                    currentPassEncoder.setBindGroup(0, bindGroup, [dynamicOffset]);
                    passState.uniformBufferId = uniformBufferId;
                    passState.dynamicOffset = dynamicOffset;
                  } else {
                    currentPassEncoder.setBindGroup(0, bindGroup);
                  }
                  passState.boundPipelineId = pipelineId;
                }
              }

              if (vertexBufferId > 0) {
                if (passState.vertexBufferId !== vertexBufferId) {
                  const vb = this.buffers.get(vertexBufferId);
                  if (!vb) {
                    throw new Error(`RenderPass: unknown vertexBufferId ${vertexBufferId}`);
                  }
                  currentPassEncoder.setVertexBuffer(0, vb);
                  passState.vertexBufferId = vertexBufferId;
                }
              }

              currentPassEncoder.draw(vertexCount, ...drawParameters);
            }
            break;
          }

          case OPCODE_RENDER_PASS_DEPTH: {
            const hasDrawParameters = pendingDrawParameters !== null;
            const drawParameters = pendingDrawParameters || [1, 0, 0];
            pendingDrawParameters = null;
            if (cursor + 56 > dataBlockStart) {
              throw new Error(`Truncated RENDER_PASS_DEPTH fields at command ${i}`);
            }
            const rawTargetType = dataView.getUint32(cursor, true);
            const targetKind = rawTargetType & 0xFF;
            if (targetKind !== TARGET_OFFSCREEN && targetKind !== TARGET_CANVAS) {
              throw new Error(`RenderPassDepth: invalid target kind ${targetKind} in 0x${rawTargetType.toString(16)}`);
            }
            const loadOpCode = (rawTargetType >> 8) & 0xFF;
            if (loadOpCode > 2) {
              throw new Error(`RenderPassDepth: invalid load_op code ${loadOpCode}`);
            }
            const storeOpCode = (rawTargetType >> 16) & 0xFF;
            if (storeOpCode > 1) {
              throw new Error(`RenderPassDepth: invalid store_op code ${storeOpCode}`);
            }
            const passFlags = (rawTargetType >> 24) & 0xFF;
            if ((passFlags & ~1) !== 0) {
              throw new Error(`RenderPassDepth: unknown pass flags 0x${passFlags.toString(16)}`);
            }

            const targetId = dataView.getUint32(cursor + 4, true);
            const cr = dataView.getFloat32(cursor + 8, true);
            const cg = dataView.getFloat32(cursor + 12, true);
            const cb = dataView.getFloat32(cursor + 16, true);
            const ca = dataView.getFloat32(cursor + 20, true);
            const pipelineId = dataView.getUint32(cursor + 24, true);
            const vertexBufferId = dataView.getUint32(cursor + 28, true);
            const vertexCount = dataView.getUint32(cursor + 32, true);
            const dynamicOffset = dataView.getUint32(cursor + 36, true);
            const uniformBufferId = dataView.getUint32(cursor + 40, true) || 1;
            const depthTargetId = dataView.getUint32(cursor + 44, true);
            const packedDepthOps = dataView.getUint32(cursor + 48, true);
            const depthClearValue = dataView.getFloat32(cursor + 52, true);
            cursor += 56;

            const depthLoadOpCode = packedDepthOps & 0xFF;
            const depthStoreOpCode = (packedDepthOps >> 8) & 0xFF;
            const depthReadOnlyCode = (packedDepthOps >> 16) & 0xFF;
            const reservedDepthFlags = (packedDepthOps >> 24) & 0xFF;

            if (depthLoadOpCode > 2) {
              throw new Error(`RenderPassDepth: invalid depth_load_op code ${depthLoadOpCode}`);
            }
            if (depthStoreOpCode > 1) {
              throw new Error(`RenderPassDepth: invalid depth_store_op code ${depthStoreOpCode}`);
            }
            if (depthReadOnlyCode > 1) {
              throw new Error(`RenderPassDepth: invalid depth_read_only code ${depthReadOnlyCode}`);
            }
            if (reservedDepthFlags !== 0) {
              throw new Error(`RenderPassDepth: reserved depth flags must be 0, received ${reservedDepthFlags}`);
            }

            if (targetKind === TARGET_OFFSCREEN) {
              this.lastRenderTargetId = targetId;
            }

            const isNewPass = (passFlags & 1) !== 0;
            const targetKey = `${targetKind}:${targetId}:depth:${depthTargetId}`;
            if (hasDrawParameters && (isNewPass || currentPassTargetKey !== targetKey)) {
              throw new Error("SetDrawParameters cannot cross a render-pass boundary");
            }

            if (!currentPassEncoder || currentPassTargetKey !== targetKey || isNewPass) {
              closeActivePass();

              let targetView;
              if (targetKind === TARGET_CANVAS) {
                if (!canvasContext) {
                  throw new Error("RenderPassDepth: canvas target requires a canvas context");
                }
                // Canvas swapchain view is acquired at most once per packet execution (§8.5, [S48])
                targetView = getCanvasView();
              } else if (targetKind === TARGET_OFFSCREEN) {
                const texture = this.textures.get(targetId);
                if (!texture) {
                  throw new Error(`RenderPassDepth: unknown offscreen targetId ${targetId}`);
                }
                targetView = texture.createView();
              } else {
                throw new Error(`Invalid render pass targetKind: ${targetKind}`);
              }

              const depthTexture = this.textures.get(depthTargetId);
              if (!depthTexture) {
                throw new Error(`RenderPassDepth: unknown depthTargetId ${depthTargetId}`);
              }
              const depthView = depthTexture.createView();

              const loadOp = loadOpCode === 1 ? "load" : "clear";
              const storeOp = storeOpCode === 1 ? "discard" : "store";
              const colorAttachmentDesc = {
                view: targetView,
                loadOp,
                storeOp,
              };
              if (loadOp === "clear") {
                colorAttachmentDesc.clearValue = { r: cr, g: cg, b: cb, a: ca };
              }

              const depthReadOnly = depthReadOnlyCode === 1;

              const depthAttachmentDesc = {
                view: depthView,
                depthReadOnly,
              };
              if (!depthReadOnly) {
                const depthLoadOp = depthLoadOpCode === 1 ? "load" : "clear";
                const depthStoreOp = depthStoreOpCode === 1 ? "discard" : "store";
                depthAttachmentDesc.depthLoadOp = depthLoadOp;
                depthAttachmentDesc.depthStoreOp = depthStoreOp;
                if (depthLoadOp === "clear") {
                  depthAttachmentDesc.depthClearValue = depthClearValue;
                }
              }

              currentPassEncoder = commandEncoder.beginRenderPass({
                colorAttachments: [colorAttachmentDesc],
                depthStencilAttachment: depthAttachmentDesc,
              });
              currentPassTargetKey = targetKey;
              currentPassHasDepth = true;
              passState = {
                pipelineId: null,
                uniformBufferId: null,
                dynamicOffset: null,
                vertexBufferId: null,
                boundPipelineId: null,
              };
            }

            if (vertexCount > 0) {
              const pipelineRecord = this.pipelines.get(pipelineId);
              if (!pipelineRecord) {
                throw new Error(`RenderPassDepth: unknown pipelineId ${pipelineId}`);
              }

              if (passState.pipelineId !== pipelineId) {
                currentPassEncoder.setPipeline(pipelineRecord.pipeline);
                passState.pipelineId = pipelineId;
                passState.uniformBufferId = null;
                passState.dynamicOffset = null;
                passState.boundPipelineId = null;
              }

              const hasUniform = pipelineRecord.hasUniformBuffer;
              const isTextured = Boolean(pipelineRecord.isTextured || (pipelineRecord.texture && pipelineRecord.sampler));
              if (hasUniform || isTextured) {
                const needsRebind = hasUniform
                  ? (passState.uniformBufferId !== uniformBufferId || passState.dynamicOffset !== dynamicOffset || passState.boundPipelineId !== pipelineId)
                  : (passState.boundPipelineId !== pipelineId);
                if (needsRebind) {
                  const bindGroupEntries = [];
                  if (hasUniform) {
                    const uniformBuf = this.buffers.get(uniformBufferId);
                    if (!uniformBuf) {
                      throw new Error(`RenderPassDepth: uniform buffer ${uniformBufferId} missing for pipeline`);
                    }
                    bindGroupEntries.push({
                      binding: 0,
                      resource: {
                        buffer: uniformBuf,
                        offset: 0,
                        size: pipelineRecord.uniformSize || 48,
                      },
                    });
                  }
                  if (isTextured) {
                    bindGroupEntries.push(
                      {
                        binding: 1,
                        resource: pipelineRecord.textureView || pipelineRecord.texture.createView(),
                      },
                      {
                        binding: 2,
                        resource: pipelineRecord.sampler,
                      }
                    );
                  }
                  const bindGroup = this.device.createBindGroup({
                    layout: pipelineRecord.bindGroupLayout,
                    entries: bindGroupEntries,
                  });
                  if (hasUniform) {
                    currentPassEncoder.setBindGroup(0, bindGroup, [dynamicOffset]);
                    passState.uniformBufferId = uniformBufferId;
                    passState.dynamicOffset = dynamicOffset;
                  } else {
                    currentPassEncoder.setBindGroup(0, bindGroup);
                  }
                  passState.boundPipelineId = pipelineId;
                }
              }

              if (vertexBufferId > 0) {
                if (passState.vertexBufferId !== vertexBufferId) {
                  const vb = this.buffers.get(vertexBufferId);
                  if (!vb) {
                    throw new Error(`RenderPassDepth: unknown vertexBufferId ${vertexBufferId}`);
                  }
                  currentPassEncoder.setVertexBuffer(0, vb);
                  passState.vertexBufferId = vertexBufferId;
                }
              }

              currentPassEncoder.draw(vertexCount, ...drawParameters);
            }
            break;
          }

          case OPCODE_COPY_TEXTURE_TO_BUFFER: {
            closeActivePass();
            if (cursor + 24 > dataBlockStart) {
              throw new Error(`Truncated COPY_TEXTURE_TO_BUFFER fields at command ${i}`);
            }
            const textureId = dataView.getUint32(cursor, true);
            const bufferId = dataView.getUint32(cursor + 4, true);
            const width = dataView.getUint32(cursor + 8, true);
            const height = dataView.getUint32(cursor + 12, true);
            const epochHi = dataView.getUint32(cursor + 16, true);
            const epochLo = dataView.getUint32(cursor + 20, true);
            cursor += 24;

            this.bufferEpochs.set(bufferId, { epochHi, epochLo });

            const texture = this.textures.get(textureId);
            const buffer = this.buffers.get(bufferId);
            if (!texture || !buffer) {
              throw new Error("CopyTextureToBuffer: invalid texture or buffer id");
            }

            const unalignedBytesPerRow = width * 4;
            const bytesPerRow = Math.ceil(unalignedBytesPerRow / 256) * 256;

            commandEncoder.copyTextureToBuffer(
              { texture: texture },
              { buffer: buffer, bytesPerRow: bytesPerRow, rowsPerImage: height },
              [width, height, 1]
            );
            break;
          }

          case OPCODE_COPY_BUFFER_TO_BUFFER: {
            closeActivePass();
            if (cursor + 40 > dataBlockStart) {
              throw new Error(`Truncated COPY_BUFFER_TO_BUFFER fields at command ${i}`);
            }

            const MAX_SAFE_INTEGER_BIGINT = BigInt(Number.MAX_SAFE_INTEGER);

            const sourceBufferId = dataView.getUint32(cursor, true);
            const srcOffsetBig = dataView.getBigUint64(cursor + 4, true);
            const destinationBufferId = dataView.getUint32(cursor + 12, true);
            const dstOffsetBig = dataView.getBigUint64(cursor + 16, true);
            const sizeBig = dataView.getBigUint64(cursor + 24, true);
            const epochLo = dataView.getUint32(cursor + 32, true);
            const epochHi = dataView.getUint32(cursor + 36, true);
            cursor += 40;

            if (srcOffsetBig > MAX_SAFE_INTEGER_BIGINT) {
              throw new Error(`CopyBufferToBuffer: source_offset ${srcOffsetBig} exceeds Number.MAX_SAFE_INTEGER`);
            }
            if (dstOffsetBig > MAX_SAFE_INTEGER_BIGINT) {
              throw new Error(`CopyBufferToBuffer: destination_offset ${dstOffsetBig} exceeds Number.MAX_SAFE_INTEGER`);
            }
            if (sizeBig > MAX_SAFE_INTEGER_BIGINT) {
              throw new Error(`CopyBufferToBuffer: size ${sizeBig} exceeds Number.MAX_SAFE_INTEGER`);
            }

            const sourceOffset = Number(srcOffsetBig);
            const destinationOffset = Number(dstOffsetBig);
            const size = Number(sizeBig);

            if (sourceOffset % 4 !== 0) {
              throw new Error(`CopyBufferToBuffer: source_offset (${sourceOffset}) must be a multiple of 4`);
            }
            if (destinationOffset % 4 !== 0) {
              throw new Error(`CopyBufferToBuffer: destination_offset (${destinationOffset}) must be a multiple of 4`);
            }
            if (size % 4 !== 0) {
              throw new Error(`CopyBufferToBuffer: size (${size}) must be a multiple of 4`);
            }

            if (srcOffsetBig + sizeBig > MAX_SAFE_INTEGER_BIGINT || !Number.isSafeInteger(sourceOffset + size)) {
              throw new Error(`CopyBufferToBuffer: source range calculation overflows safe integer bounds`);
            }
            if (dstOffsetBig + sizeBig > MAX_SAFE_INTEGER_BIGINT || !Number.isSafeInteger(destinationOffset + size)) {
              throw new Error(`CopyBufferToBuffer: destination range calculation overflows safe integer bounds`);
            }

            const srcBuffer = this.buffers.get(sourceBufferId);
            if (!srcBuffer) {
              throw new Error(`CopyBufferToBuffer: unknown source bufferId ${sourceBufferId}`);
            }
            const dstBuffer = this.buffers.get(destinationBufferId);
            if (!dstBuffer) {
              throw new Error(`CopyBufferToBuffer: unknown destination bufferId ${destinationBufferId}`);
            }

            const COPY_SRC = (typeof GPUBufferUsage !== "undefined" && GPUBufferUsage.COPY_SRC) ? GPUBufferUsage.COPY_SRC : 0x0004;
            const COPY_DST = (typeof GPUBufferUsage !== "undefined" && GPUBufferUsage.COPY_DST) ? GPUBufferUsage.COPY_DST : 0x0008;

            if ((srcBuffer.usage & COPY_SRC) === 0) {
              throw new Error(`CopyBufferToBuffer: source buffer ${sourceBufferId} lacks COPY_SRC usage (usage: 0x${srcBuffer.usage.toString(16)})`);
            }
            if ((dstBuffer.usage & COPY_DST) === 0) {
              throw new Error(`CopyBufferToBuffer: destination buffer ${destinationBufferId} lacks COPY_DST usage (usage: 0x${dstBuffer.usage.toString(16)})`);
            }

            if (sourceOffset + size > srcBuffer.size) {
              throw new Error(`CopyBufferToBuffer: source range out of bounds (offset ${sourceOffset} + size ${size} = ${sourceOffset + size} > buffer size ${srcBuffer.size})`);
            }
            if (destinationOffset + size > dstBuffer.size) {
              throw new Error(`CopyBufferToBuffer: destination range out of bounds (offset ${destinationOffset} + size ${size} = ${destinationOffset + size} > buffer size ${dstBuffer.size})`);
            }

            if (srcBuffer === dstBuffer || sourceBufferId === destinationBufferId) {
              throw new Error(
                `CopyBufferToBuffer: source and destination buffers must be distinct objects (source: ${sourceBufferId}, destination: ${destinationBufferId})`
              );
            }

            this.bufferEpochs.set(destinationBufferId, { epochHi, epochLo });

            commandEncoder.copyBufferToBuffer(
              srcBuffer,
              sourceOffset,
              dstBuffer,
              destinationOffset,
              size
            );
            break;
          }

          case OPCODE_RECORD_BUNDLE:
          case OPCODE_RECORD_BUNDLE_BATCH: {
            closeActivePass();
            const isBatch = opcode === OPCODE_RECORD_BUNDLE_BATCH;
            const fieldBytes = isBatch ? 36 : 28;
            if (cursor + fieldBytes > dataBlockStart) {
              throw new Error(`Truncated RECORD_BUNDLE fields at command ${i}`);
            }
            const bundleId = dataView.getUint32(cursor, true);
            const pipelineId = dataView.getUint32(cursor + 4, true);
            const vertexBufferId = dataView.getUint32(cursor + 8, true);
            const vertexCount = dataView.getUint32(cursor + 12, true);
            const dynamicOffset = dataView.getUint32(cursor + 16, true);
            const uniformBufferId = dataView.getUint32(cursor + 20, true) || 1;
            const targetFormatCode = dataView.getUint32(cursor + 24, true);
            const drawCount = isBatch ? dataView.getUint32(cursor + 28, true) : 1;
            const offsetStride = isBatch ? dataView.getUint32(cursor + 32, true) : 0;
            cursor += fieldBytes;
            const lastOffset = drawCount === 0 ? dynamicOffset : dynamicOffset + (drawCount - 1) * offsetStride;
            if (drawCount > 0 && (!Number.isSafeInteger(lastOffset) || lastOffset > 0xffffffff)) {
              throw new Error("RecordBundleBatch: dynamic offset overflow");
            }

            let format;
            if (targetFormatCode === 0) {
              format = this.capabilityRecord?.preferredCanvasFormat || "bgra8unorm";
            } else if (targetFormatCode === 1) {
              format = "bgra8unorm";
            } else if (targetFormatCode === 2) {
              format = "rgba8unorm";
            } else {
              throw new Error(`Invalid bundle target formatCode: ${targetFormatCode}`);
            }

            const pipelineRecord = this.pipelines.get(pipelineId);
            if (!pipelineRecord) {
              throw new Error(`RecordBundle: unknown pipelineId ${pipelineId}`);
            }
            if (pipelineRecord.hasDepth) {
              throw new Error("RecordBundle: bundles with depth attachments are not yet implemented");
            }
            const uniformBuf = pipelineRecord.hasUniformBuffer ? this.buffers.get(uniformBufferId) : null;
            if (pipelineRecord.hasUniformBuffer && !uniformBuf) {
              throw new Error(`RecordBundle: uniform buffer ${uniformBufferId} missing for pipeline`);
            }
            if (isBatch && pipelineRecord.hasUniformBuffer && drawCount > 0) {
              const alignment = this.device.limits.minUniformBufferOffsetAlignment;
              if (dynamicOffset % alignment !== 0 || (drawCount > 1 && offsetStride % alignment !== 0) ||
                  lastOffset + (pipelineRecord.uniformSize || 48) > uniformBuf.size) {
                throw new Error("RecordBundleBatch: uniform offsets exceed buffer bounds or device alignment");
              }
            }

            const bundleEncoder = this.device.createRenderBundleEncoder({
              colorFormats: [format],
            });

            bundleEncoder.setPipeline(pipelineRecord.pipeline);

            const hasUniform = pipelineRecord.hasUniformBuffer;
            const isTextured = Boolean(pipelineRecord.isTextured || (pipelineRecord.texture && pipelineRecord.sampler));
            let bindGroup = null;
            if (hasUniform || isTextured) {
              const bindGroupEntries = [];
              if (hasUniform) {
                bindGroupEntries.push({
                  binding: 0,
                  resource: {
                    buffer: uniformBuf,
                    offset: 0,
                    size: pipelineRecord.uniformSize || 48,
                  },
                });
              }
              if (isTextured) {
                bindGroupEntries.push(
                  {
                    binding: 1,
                    resource: pipelineRecord.textureView || pipelineRecord.texture.createView(),
                  },
                  {
                    binding: 2,
                    resource: pipelineRecord.sampler,
                  }
                );
              }
              bindGroup = this.device.createBindGroup({
                layout: pipelineRecord.bindGroupLayout,
                entries: bindGroupEntries,
              });
              if (hasUniform) {
                if (drawCount > 0) bundleEncoder.setBindGroup(0, bindGroup, [dynamicOffset]);
              } else {
                bundleEncoder.setBindGroup(0, bindGroup);
              }
            }

            if (vertexBufferId > 0) {
              const vb = this.buffers.get(vertexBufferId);
              if (!vb) {
                throw new Error(`RecordBundle: unknown vertexBufferId ${vertexBufferId}`);
              }
              bundleEncoder.setVertexBuffer(0, vb);
            }

            for (let draw = 0; draw < drawCount; draw++) {
              if (draw > 0 && hasUniform) {
                bundleEncoder.setBindGroup(0, bindGroup, [dynamicOffset + draw * offsetStride]);
              }
              if (vertexCount > 0) bundleEncoder.draw(vertexCount, 1, 0, 0);
            }

            const bundle = bundleEncoder.finish();
            this.bundles.set(bundleId, bundle);
            break;
          }

          case OPCODE_EXECUTE_BUNDLES: {
            if (currentPassHasDepth) {
              throw new Error("ExecuteBundles: bundles in passes with depth attachments are not yet implemented");
            }
            if (cursor + 4 > dataBlockStart) {
              throw new Error(`Truncated EXECUTE_BUNDLES fields at command ${i}`);
            }
            const bundleCount = dataView.getUint32(cursor, true);
            cursor += 4;
            if (cursor + bundleCount * 4 > dataBlockStart) {
              throw new Error(`Truncated EXECUTE_BUNDLES list at command ${i}`);
            }
            const bundleIds = [];
            for (let b = 0; b < bundleCount; b++) {
              bundleIds.push(dataView.getUint32(cursor + b * 4, true));
            }
            cursor += bundleCount * 4;

            if (!currentPassEncoder) {
              // Open pass on enclosing render pass target if not already open
              let targetType = TARGET_OFFSCREEN;
              let targetId = 0;
              let clearColor = [0, 0, 0, 1];
              let scanCursor = cursor;
              let foundTarget = false;

              while (scanCursor + 2 <= dataBlockStart) {
                const nextOp = dataView.getUint16(scanCursor, true);
                scanCursor += 2;
                if (nextOp === OPCODE_RENDER_PASS) {
                  if (scanCursor + 44 <= dataBlockStart) {
                    targetType = dataView.getUint32(scanCursor, true) & 0xFF;
                    targetId = dataView.getUint32(scanCursor + 4, true);
                    clearColor = [
                      dataView.getFloat32(scanCursor + 8, true),
                      dataView.getFloat32(scanCursor + 12, true),
                      dataView.getFloat32(scanCursor + 16, true),
                      dataView.getFloat32(scanCursor + 20, true),
                    ];
                    foundTarget = true;
                  }
                  break;
                } else if (nextOp === OPCODE_RENDER_PASS_DEPTH) {
                  throw new Error("ExecuteBundles: bundles in passes with depth attachments are not yet implemented");
                } else if (nextOp === OPCODE_CREATE_BUFFER) {
                  scanCursor += 12;
                } else if (nextOp === OPCODE_WRITE_BUFFER) {
                  scanCursor += 16;
                } else if (nextOp === OPCODE_CREATE_TEXTURE) {
                  scanCursor += 20;
                } else if (nextOp === OPCODE_WRITE_TEXTURE) {
                  scanCursor += 24;
                } else if (nextOp === OPCODE_CREATE_PIPELINE) {
                  scanCursor += 32;
                } else if (nextOp === OPCODE_CREATE_PIPELINE_CULL) {
                  scanCursor += 40;
                } else if (nextOp === OPCODE_CREATE_PIPELINE_DEPTH) {
                  scanCursor += 44;
                } else if (nextOp === OPCODE_CREATE_PIPELINE_DEPTH_CULL) {
                  scanCursor += 52;
                } else if (nextOp === OPCODE_CREATE_PIPELINE_DEPTH_CULL_COLOR) {
                  scanCursor += 56;
                } else if (nextOp === OPCODE_CREATE_PIPELINE_TEXTURED) {
                  scanCursor += 44;
                } else if (nextOp === OPCODE_COPY_TEXTURE_TO_BUFFER) {
                  scanCursor += 24;
                } else if (nextOp === OPCODE_RECORD_BUNDLE) {
                  scanCursor += 28;
                } else if (nextOp === OPCODE_RECORD_BUNDLE_BATCH) {
                  scanCursor += 36;
                } else if (nextOp === OPCODE_COPY_BUFFER_TO_BUFFER) {
                  scanCursor += 40;
                } else if (nextOp === OPCODE_CREATE_COMPUTE_PIPELINE) {
                  if (scanCursor + 30 <= dataBlockStart) {
                    const bindingCount = dataView.getUint32(scanCursor + 22, true);
                    scanCursor += 30 + bindingCount * 16;
                  } else {
                    break;
                  }
                } else if (nextOp === OPCODE_DISPATCH_COMPUTE) {
                  if (scanCursor + 22 <= dataBlockStart) {
                    const bindingCount = dataView.getUint32(scanCursor + 18, true);
                    scanCursor += 22 + bindingCount * 48;
                  } else {
                    break;
                  }
                } else if (nextOp === OPCODE_SET_VIEWPORT) {
                  scanCursor += 24;
                } else if (nextOp === OPCODE_SET_SCISSOR_RECT) {
                  scanCursor += 16;
                } else if (nextOp === OPCODE_SET_DRAW_PARAMETERS) {
                  scanCursor += 12;
                } else if (nextOp === OPCODE_EXECUTE_BUNDLES) {
                  if (scanCursor + 4 <= dataBlockStart) {
                    const cnt = dataView.getUint32(scanCursor, true);
                    scanCursor += 4 + cnt * 4;
                  } else {
                    break;
                  }
                } else {
                  break;
                }
              }

              if (!foundTarget && this.lastRenderTargetId) {
                targetType = TARGET_OFFSCREEN;
                targetId = this.lastRenderTargetId;
                foundTarget = true;
              }

              if (!foundTarget) {
                throw new Error("ExecuteBundles: no active render pass and no target found");
              }

              let targetView;
              if (targetType === TARGET_CANVAS) {
                if (!canvasContext) {
                  throw new Error("ExecuteBundles: canvas target requires a canvas context");
                }
                targetView = getCanvasView();
              } else {
                const texture = this.textures.get(targetId);
                if (!texture) {
                  throw new Error(`ExecuteBundles: unknown offscreen targetId ${targetId}`);
                }
                targetView = texture.createView();
              }

              currentPassEncoder = commandEncoder.beginRenderPass({
                colorAttachments: [
                  {
                    view: targetView,
                    clearValue: { r: clearColor[0], g: clearColor[1], b: clearColor[2], a: clearColor[3] },
                    loadOp: "clear",
                    storeOp: "store",
                  },
                ],
              });
              currentPassTargetKey = `${targetType}:${targetId}:none`;
              currentPassHasDepth = false;
              passState = {
                pipelineId: null,
                uniformBufferId: null,
                dynamicOffset: null,
                vertexBufferId: null,
                boundPipelineId: null,
              };
            }

            const bundleList = [];
            for (const bId of bundleIds) {
              const b = this.bundles.get(bId);
              if (!b) {
                throw new Error(`ExecuteBundles: unknown bundleId ${bId}`);
              }
              bundleList.push(b);
            }

            currentPassEncoder.executeBundles(bundleList);

            // WebGPU Spec Invariant: executeBundles clears pass state!
            // Any following direct draw MUST explicitly rebind pipeline, bind groups, and vertex buffers.
            if (!this.wrongImplSkipBundleStateReset) {
              passState = {
                pipelineId: null,
                uniformBufferId: null,
                dynamicOffset: null,
                vertexBufferId: null,
                boundPipelineId: null,
              };
            }
            break;
          }

          case OPCODE_SET_VIEWPORT: {
            if (!currentPassEncoder) {
              throw new Error("SetViewport: no active render pass");
            }
            if (cursor + 24 > dataBlockStart) {
              throw new Error(`Truncated SET_VIEWPORT fields at command ${i}`);
            }
            const x = dataView.getFloat32(cursor, true);
            const y = dataView.getFloat32(cursor + 4, true);
            const width = dataView.getFloat32(cursor + 8, true);
            const height = dataView.getFloat32(cursor + 12, true);
            const minDepth = dataView.getFloat32(cursor + 16, true);
            const maxDepth = dataView.getFloat32(cursor + 20, true);
            cursor += 24;

            currentPassEncoder.setViewport(x, y, width, height, minDepth, maxDepth);
            break;
          }

          case OPCODE_SET_DRAW_PARAMETERS: {
            if (!currentPassEncoder) {
              throw new Error("SetDrawParameters: no active render pass");
            }
            if (cursor + 12 > dataBlockStart) {
              throw new Error(`Truncated SET_DRAW_PARAMETERS fields at command ${i}`);
            }
            pendingDrawParameters = [
              dataView.getUint32(cursor, true),
              dataView.getUint32(cursor + 4, true),
              dataView.getUint32(cursor + 8, true),
            ];
            cursor += 12;
            break;
          }

          case OPCODE_SET_SCISSOR_RECT: {
            if (!currentPassEncoder) {
              throw new Error("SetScissorRect: no active render pass");
            }
            if (cursor + 16 > dataBlockStart) {
              throw new Error(`Truncated SET_SCISSOR_RECT fields at command ${i}`);
            }
            const x = dataView.getUint32(cursor, true);
            const y = dataView.getUint32(cursor + 4, true);
            const width = dataView.getUint32(cursor + 8, true);
            const height = dataView.getUint32(cursor + 12, true);
            cursor += 16;

            currentPassEncoder.setScissorRect(x, y, width, height);
            break;
          }

          case OPCODE_CREATE_COMPUTE_PIPELINE: {
            closeActivePass();
            if (cursor + 30 > dataBlockStart) {
              throw new RangeError(`Truncated CREATE_COMPUTE_PIPELINE header at command ${i}`);
            }
            const _pad0 = dataView.getUint16(cursor, true);
            const pipelineId = dataView.getUint32(cursor + 2, true);
            const codeOffset = dataView.getUint32(cursor + 6, true);
            const codeLen = dataView.getUint32(cursor + 10, true);
            const entryPointOffset = dataView.getUint32(cursor + 14, true);
            const entryPointLen = dataView.getUint32(cursor + 18, true);
            const bindingCount = dataView.getUint32(cursor + 22, true);
            const _pad1 = dataView.getUint32(cursor + 26, true);
            cursor += 30;

            if (cursor + bindingCount * 16 > dataBlockStart) {
              throw new RangeError(`Truncated CREATE_COMPUTE_PIPELINE binding list at command ${i} (needs ${bindingCount * 16} bytes)`);
            }

            if (codeOffset + codeLen > dataPayload.byteLength) {
              throw new RangeError(`CreateComputePipeline: WGSL code out of bounds (offset ${codeOffset} + len ${codeLen} > payload ${dataPayload.byteLength})`);
            }
            const codeBytes = dataPayload.subarray(codeOffset, codeOffset + codeLen);
            const wgslCode = new TextDecoder("utf-8").decode(codeBytes);

            let entryPoint = "main";
            if (entryPointLen > 0) {
              if (entryPointOffset + entryPointLen > dataPayload.byteLength) {
                throw new RangeError(`CreateComputePipeline: entry_point out of bounds (offset ${entryPointOffset} + len ${entryPointLen} > payload ${dataPayload.byteLength})`);
              }
              const epBytes = dataPayload.subarray(entryPointOffset, entryPointOffset + entryPointLen);
              entryPoint = new TextDecoder("utf-8").decode(epBytes);
            }

            const layoutEntries = [];
            const bindingSpecs = new Map();
            let storageBufferCount = 0;

            for (let b = 0; b < bindingCount; b++) {
              const bindingIndex = dataView.getUint32(cursor, true);
              const bindingType = dataView.getUint32(cursor + 4, true);
              const minBindingSize = dataView.getUint32(cursor + 8, true);
              const _pad = dataView.getUint32(cursor + 12, true);
              cursor += 16;

              if (bindingSpecs.has(bindingIndex)) {
                throw new Error(`CreateComputePipeline: duplicate binding index ${bindingIndex}`);
              }

              let bufferType;
              if (bindingType === BINDING_TYPE_UNIFORM) {
                bufferType = "uniform";
              } else if (bindingType === BINDING_TYPE_STORAGE_READ) {
                bufferType = "read-only-storage";
                storageBufferCount++;
              } else if (bindingType === BINDING_TYPE_STORAGE_READ_WRITE) {
                bufferType = "storage";
                storageBufferCount++;
              } else {
                throw new TypeError(`CreateComputePipeline: invalid binding_type ${bindingType} at index ${b}`);
              }

              const entry = {
                binding: bindingIndex,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                  type: bufferType,
                },
              };
              if (minBindingSize > 0) {
                entry.buffer.minBindingSize = minBindingSize;
              }
              layoutEntries.push(entry);
              bindingSpecs.set(bindingIndex, { bindingIndex, bindingType, minBindingSize });
            }

            if (this.device.limits && this.device.limits.maxStorageBuffersPerShaderStage !== undefined) {
              if (storageBufferCount > this.device.limits.maxStorageBuffersPerShaderStage) {
                throw new RangeError(
                  `CreateComputePipeline: storage buffer count ${storageBufferCount} exceeds device limit maxStorageBuffersPerShaderStage (${this.device.limits.maxStorageBuffersPerShaderStage})`
                );
              }
            }

            const shaderModule = this.device.createShaderModule({ code: wgslCode });
            const bindGroupLayout = this.device.createBindGroupLayout({ entries: layoutEntries });
            const pipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
            const computePipeline = this.device.createComputePipeline({
              layout: pipelineLayout,
              compute: {
                module: shaderModule,
                entryPoint: entryPoint,
              },
            });

            this.computePipelines.set(pipelineId, {
              pipeline: computePipeline,
              bindGroupLayout: bindGroupLayout,
              bindingSpecs: bindingSpecs,
            });
            break;
          }

          case OPCODE_DISPATCH_COMPUTE: {
            closeActivePass();
            if (cursor + 22 > dataBlockStart) {
              throw new RangeError(`Truncated DISPATCH_COMPUTE header at command ${i}`);
            }
            const _pad0 = dataView.getUint16(cursor, true);
            const pipelineId = dataView.getUint32(cursor + 2, true);
            const workgroupCountX = dataView.getUint32(cursor + 6, true);
            const workgroupCountY = dataView.getUint32(cursor + 10, true);
            const workgroupCountZ = dataView.getUint32(cursor + 14, true);
            const bindingCount = dataView.getUint32(cursor + 18, true);
            cursor += 22;

            if (cursor + bindingCount * 48 > dataBlockStart) {
              throw new RangeError(`Truncated DISPATCH_COMPUTE binding records at command ${i} (needs ${bindingCount * 48} bytes)`);
            }

            const pipelineRecord = this.computePipelines.get(pipelineId);
            if (!pipelineRecord) {
              throw new Error(`DispatchCompute: unknown compute pipelineId ${pipelineId}`);
            }

            if (this.device.limits && this.device.limits.maxComputeWorkgroupsPerDimension !== undefined) {
              const maxDim = this.device.limits.maxComputeWorkgroupsPerDimension;
              if (workgroupCountX > maxDim || workgroupCountY > maxDim || workgroupCountZ > maxDim) {
                throw new RangeError(
                  `DispatchCompute: workgroup count (${workgroupCountX}, ${workgroupCountY}, ${workgroupCountZ}) exceeds device limit maxComputeWorkgroupsPerDimension (${maxDim})`
                );
              }
            }

            const MAX_SAFE_INTEGER_BIGINT = BigInt(Number.MAX_SAFE_INTEGER);
            const parsedBindings = [];
            const seenBindingIndices = new Set();

            for (let b = 0; b < bindingCount; b++) {
              const bindingIndex = dataView.getUint32(cursor, true);
              const bufferId = dataView.getUint32(cursor + 4, true);
              const offsetBig = dataView.getBigUint64(cursor + 8, true);
              const sizeBig = dataView.getBigUint64(cursor + 16, true);
              const bindingType = dataView.getUint32(cursor + 24, true);
              const _pad = dataView.getUint32(cursor + 28, true);
              const epochLo = dataView.getUint32(cursor + 32, true);
              const epochHi = dataView.getUint32(cursor + 36, true);
              const dataVersionBig = dataView.getBigUint64(cursor + 40, true);
              cursor += 48;

              if (seenBindingIndices.has(bindingIndex)) {
                throw new Error(`DispatchCompute: duplicate binding index ${bindingIndex}`);
              }
              seenBindingIndices.add(bindingIndex);

              const spec = pipelineRecord.bindingSpecs.get(bindingIndex);
              if (!spec) {
                throw new Error(`DispatchCompute: pipeline ${pipelineId} has no binding specification for index ${bindingIndex}`);
              }

              if (bindingType !== spec.bindingType) {
                throw new TypeError(`DispatchCompute: binding index ${bindingIndex} type mismatch: dispatch has ${bindingType}, pipeline expects ${spec.bindingType}`);
              }

              if (offsetBig > MAX_SAFE_INTEGER_BIGINT) {
                throw new RangeError(`DispatchCompute: binding ${bindingIndex} offset exceeds Number.MAX_SAFE_INTEGER`);
              }
              if (sizeBig > MAX_SAFE_INTEGER_BIGINT) {
                throw new RangeError(`DispatchCompute: binding ${bindingIndex} size exceeds Number.MAX_SAFE_INTEGER`);
              }
              const offset = Number(offsetBig);
              const size = Number(sizeBig);

              if (size === 0) {
                throw new RangeError(`DispatchCompute: binding ${bindingIndex} size must be greater than 0`);
              }
              if ((bindingType === BINDING_TYPE_STORAGE_READ || bindingType === BINDING_TYPE_STORAGE_READ_WRITE) && size % 4 !== 0) {
                throw new RangeError(`DispatchCompute: storage buffer binding ${bindingIndex} size ${size} must be a multiple of 4 bytes`);
              }
              if (spec.minBindingSize > 0 && size < spec.minBindingSize) {
                throw new RangeError(`DispatchCompute: binding ${bindingIndex} size ${size} is smaller than pipeline minBindingSize (${spec.minBindingSize})`);
              }

              const buffer = this.buffers.get(bufferId);
              if (!buffer) {
                throw new Error(`DispatchCompute: unknown bufferId ${bufferId} for binding ${bindingIndex}`);
              }

              if (bindingType === BINDING_TYPE_UNIFORM) {
                const align = (this.device.limits && this.device.limits.minUniformBufferOffsetAlignment) || 256;
                if (offset % align !== 0) {
                  throw new RangeError(`DispatchCompute: uniform buffer ${bufferId} offset ${offset} must be aligned to ${align}`);
                }
                if (this.device.limits && this.device.limits.maxUniformBufferBindingSize !== undefined && size > this.device.limits.maxUniformBufferBindingSize) {
                  throw new RangeError(`DispatchCompute: uniform buffer ${bufferId} binding size ${size} exceeds maxUniformBufferBindingSize (${this.device.limits.maxUniformBufferBindingSize})`);
                }
              } else if (bindingType === BINDING_TYPE_STORAGE_READ || bindingType === BINDING_TYPE_STORAGE_READ_WRITE) {
                const align = (this.device.limits && this.device.limits.minStorageBufferOffsetAlignment) || 256;
                if (offset % align !== 0) {
                  throw new RangeError(`DispatchCompute: storage buffer ${bufferId} offset ${offset} must be aligned to ${align}`);
                }
                if (this.device.limits && this.device.limits.maxStorageBufferBindingSize !== undefined && size > this.device.limits.maxStorageBufferBindingSize) {
                  throw new RangeError(`DispatchCompute: storage buffer ${bufferId} binding size ${size} exceeds maxStorageBufferBindingSize (${this.device.limits.maxStorageBufferBindingSize})`);
                }
              } else {
                throw new TypeError(`DispatchCompute: invalid binding_type ${bindingType} at binding index ${b}`);
              }

              if (offset + size > buffer.size) {
                throw new RangeError(`DispatchCompute: buffer ${bufferId} binding range out of bounds (offset ${offset} + size ${size} = ${offset + size} > buffer size ${buffer.size})`);
              }

              if (bindingType === BINDING_TYPE_UNIFORM) {
                const UNIFORM = (typeof GPUBufferUsage !== "undefined" && GPUBufferUsage.UNIFORM) ? GPUBufferUsage.UNIFORM : 0x0040;
                if ((buffer.usage & UNIFORM) === 0) {
                  throw new Error(`DispatchCompute: buffer ${bufferId} lacks UNIFORM usage`);
                }
              } else {
                const STORAGE = (typeof GPUBufferUsage !== "undefined" && GPUBufferUsage.STORAGE) ? GPUBufferUsage.STORAGE : 0x0080;
                if ((buffer.usage & STORAGE) === 0) {
                  throw new Error(`DispatchCompute: buffer ${bufferId} lacks STORAGE usage`);
                }
              }

              parsedBindings.push({
                bindingIndex,
                bufferId,
                buffer,
                offset,
                size,
                bindingType,
                epochHi,
                epochLo,
              });
            }

            if (seenBindingIndices.size !== pipelineRecord.bindingSpecs.size) {
              for (const reqIndex of pipelineRecord.bindingSpecs.keys()) {
                if (!seenBindingIndices.has(reqIndex)) {
                  throw new Error(`DispatchCompute: missing binding index ${reqIndex} required by pipeline ${pipelineId}`);
                }
              }
            }

            for (let b = 0; b < parsedBindings.length; b++) {
              const pb = parsedBindings[b];
              if (pb.bindingType === BINDING_TYPE_STORAGE_READ_WRITE) {
                this.bufferEpochs.set(pb.bufferId, { epochHi: pb.epochHi, epochLo: pb.epochLo });
              }
            }

            const bindGroupEntries = parsedBindings.map((pb) => ({
              binding: pb.bindingIndex,
              resource: {
                buffer: pb.buffer,
                offset: pb.offset,
                size: pb.size,
              },
            }));

            const bindGroup = this.device.createBindGroup({
              layout: pipelineRecord.bindGroupLayout,
              entries: bindGroupEntries,
            });

            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(pipelineRecord.pipeline);
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(workgroupCountX, workgroupCountY, workgroupCountZ);
            pass.end();
            break;
          }

          default:
            throw new Error(`Unknown opcode: ${opcode}`);
        }
      }

      if (pendingDrawParameters) {
        throw new Error("SetDrawParameters has no following RenderPass or RenderPassDepth");
      }
      closeActivePass();
      frameCanvasView = null;
      // Finish and submit synchronously inside the error scope
      const commandBuffer = commandEncoder.finish();
      this.device.queue.submit([commandBuffer]);
      submitted = true;
      } finally {
        for (const sb of stagingBuffers) {
          sb.destroy();
        }
      }
    });
    // Keep bookkeeping in queue-effect order even when scope promises settle
    // in another order. Never invoke the observer while device scopes are open.
    try {
      if (submitted && onSubmitted !== null) {
        const observation = onSubmitted();
        if (observation && typeof observation.then === "function") {
          // Reject async observers without leaving their rejection unowned.
          void Promise.resolve(observation).catch(() => {});
          throw new TypeError("onSubmitted must complete synchronously");
        }
      }
    } catch (error) {
      // The already-issued work still owns its asynchronous scope results.
      await completion.catch(() => {});
      throw error;
    }
    await completion;
  }

  /**
   * Reads back data from a map-readable buffer.
   */
  async readbackBuffer(bufferId, byteLength) {
    const device = this.device;
    const deviceGeneration = this.deviceGeneration;
    if (!device) throw new Error("readbackBuffer: device not initialized");
    const buffer = this.buffers.get(bufferId);
    if (!buffer) {
      throw new Error(`readbackBuffer: unknown bufferId ${bufferId}`);
    }

    // Capture metadata with this buffer: a later packet can reuse its numeric ID
    // while mapAsync is pending, but must not relabel these older GPU bytes.
    const { epochHi, epochLo } = this.bufferEpochs.get(bufferId) || { epochHi: 0, epochLo: 0 };
    await buffer.mapAsync(GPUMapMode.READ, 0, byteLength);
    let copy;
    try {
      if (!this.wrongImplSkipReadbackDeviceCheck && (this.device !== device || this.deviceGeneration !== deviceGeneration)) {
        throw new Error("readbackBuffer: device changed before mapping completed");
      }
      copy = new Uint8Array(buffer.getMappedRange(0, byteLength).slice(0));
    } finally {
      buffer.unmap();
    }

    copy.epochHi = epochHi;
    copy.epochLo = epochLo;
    copy.bufferId = bufferId;
    copy.deviceGeneration = deviceGeneration;
    copy.data = copy;
    return copy;
  }
}
