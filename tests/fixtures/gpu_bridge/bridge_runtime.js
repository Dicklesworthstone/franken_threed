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

export const TARGET_OFFSCREEN = 0;
export const TARGET_CANVAS = 1;

export const TARGET_FORMAT_PREFERRED_CANVAS = 0;
export const TARGET_FORMAT_BGRA8UNORM = 1;
export const TARGET_FORMAT_RGBA8UNORM = 2;

export const TEXTURE_USAGE_COPY_SRC = 1;
export const TEXTURE_USAGE_COPY_DST = 2;
export const TEXTURE_USAGE_TEXTURE_BINDING = 4;
export const TEXTURE_USAGE_STORAGE_BINDING = 8;
export const TEXTURE_USAGE_RENDER_ATTACHMENT = 16;

import { isDetached, ensureSafePacketBytes } from "./memory_transport.js";

export class WebGpuBridgeHost {
  constructor() {
    this.adapter = null;
    this.device = null;
    this.capabilityRecord = null;
    this.buffers = new Map();
    this.textures = new Map();
    this.pipelines = new Map();
    this.bindGroups = new Map();
    this.bundles = new Map();
    this.bufferEpochs = new Map();
    this.errorScopeActive = false;
    this.lastRenderTargetId = null;
  }

  /**
   * Negotiate features and limits before requesting a device.
   * If a required feature is missing or limits cannot be satisfied,
   * throws a structured error and does not create a device.
   */
  async negotiateAndCreateDevice(requiredProfile = {}) {
    if (!navigator.gpu) {
      throw new Error("WebGPU is not supported in this browser environment");
    }

    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: requiredProfile.powerPreference || "high-performance",
    });

    if (!adapter) {
      throw new Error("Failed to acquire a WebGPU adapter");
    }
    this.adapter = adapter;

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
    this.device = device;

    // Assert the resulting device actually meets the requested profile
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

    this.capabilityRecord = {
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

    return this.capabilityRecord;
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
    }

    return syncResult;
  }

  /**
   * Decodes and executes a coarse checked packet binary buffer.
   *
   * Invariant: Command recording, canvas texture view acquisition, commandEncoder.finish(),
   * and queue.submit() all execute SYNCHRONOUSLY inside the pushed error scopes.
   * Error scopes are popped immediately after submit() and awaited afterwards.
   * Zero eval / new Function.
   */
  async executePacket(packetBytes, canvasContext = null) {
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
    await this.withErrorScopes(["validation", "out-of-memory"], () => {
      const commandEncoder = this.device.createCommandEncoder();
      let cursor = headerLen;
      let currentPassEncoder = null;
      let currentPassTargetKey = null;
      let frameCanvasView = null;
      let passState = {
        pipelineId: null,
        uniformBufferId: null,
        dynamicOffset: null,
        vertexBufferId: null,
      };

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
          passState = {
            pipelineId: null,
            uniformBufferId: null,
            dynamicOffset: null,
            vertexBufferId: null,
          };
        }
      };

      for (let i = 0; i < commandCount; i++) {
        if (cursor + 2 > dataBlockStart) {
          throw new Error(`Truncated command at index ${i}: cursor exceeded command block`);
        }

        const opcode = dataView.getUint16(cursor, true);
        cursor += 2;

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
            if (formatCode === 1) {
              format = "bgra8unorm";
            } else if (formatCode === 2) {
              format = "rgba8unorm";
            } else {
              throw new Error(`Invalid texture formatCode: ${formatCode}`);
            }

            const texture = this.device.createTexture({
              size: [width, height, 1],
              format: format,
              usage: usage,
            });
            this.textures.set(textureId, texture);
            this.lastRenderTargetId = textureId;
            break;
          }

          case OPCODE_CREATE_PIPELINE: {
            closeActivePass();
            if (cursor + 32 > dataBlockStart) {
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
            cursor += 32;

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
                      { shaderLocation: 1, offset: 12, format: "float32x2" },
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
              },
            });

            this.pipelines.set(pipelineId, { pipeline, bindGroupLayout, hasUniformBuffer, uniformSize });
            break;
          }

          case OPCODE_RENDER_PASS: {
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
            const targetKey = `${targetKind}:${targetId}`;
            if (!currentPassEncoder || currentPassTargetKey !== targetKey || isNewPass) {
              closeActivePass();

              let targetView;
              if (targetKind === TARGET_CANVAS) {
                if (!canvasContext) {
                  // Gracefully skip canvas swapchain presentation pass in headless or pure-offscreen execution
                  continue;
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
              passState = {
                pipelineId: null,
                uniformBufferId: null,
                dynamicOffset: null,
                vertexBufferId: null,
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
              }

              if (pipelineRecord.hasUniformBuffer) {
                if (passState.uniformBufferId !== uniformBufferId || passState.dynamicOffset !== dynamicOffset) {
                  const uniformBuf = this.buffers.get(uniformBufferId);
                  if (!uniformBuf) {
                    throw new Error(`RenderPass: uniform buffer ${uniformBufferId} missing for pipeline`);
                  }
                  const bindGroup = this.device.createBindGroup({
                    layout: pipelineRecord.bindGroupLayout,
                    entries: [
                      {
                        binding: 0,
                        resource: {
                          buffer: uniformBuf,
                          offset: 0,
                          size: pipelineRecord.uniformSize || 48,
                        },
                      },
                    ],
                  });
                  currentPassEncoder.setBindGroup(0, bindGroup, [dynamicOffset]);
                  passState.uniformBufferId = uniformBufferId;
                  passState.dynamicOffset = dynamicOffset;
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

              currentPassEncoder.draw(vertexCount, 1, 0, 0);
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

          case OPCODE_RECORD_BUNDLE: {
            closeActivePass();
            if (cursor + 28 > dataBlockStart) {
              throw new Error(`Truncated RECORD_BUNDLE fields at command ${i}`);
            }
            const bundleId = dataView.getUint32(cursor, true);
            const pipelineId = dataView.getUint32(cursor + 4, true);
            const vertexBufferId = dataView.getUint32(cursor + 8, true);
            const vertexCount = dataView.getUint32(cursor + 12, true);
            const dynamicOffset = dataView.getUint32(cursor + 16, true);
            const uniformBufferId = dataView.getUint32(cursor + 20, true) || 1;
            const targetFormatCode = dataView.getUint32(cursor + 24, true);
            cursor += 28;

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

            const bundleEncoder = this.device.createRenderBundleEncoder({
              colorFormats: [format],
            });

            bundleEncoder.setPipeline(pipelineRecord.pipeline);

            if (pipelineRecord.hasUniformBuffer) {
              const uniformBuf = this.buffers.get(uniformBufferId);
              if (!uniformBuf) {
                throw new Error(`RecordBundle: uniform buffer ${uniformBufferId} missing for pipeline`);
              }
              const bindGroup = this.device.createBindGroup({
                layout: pipelineRecord.bindGroupLayout,
                entries: [
                  {
                    binding: 0,
                    resource: {
                      buffer: uniformBuf,
                      offset: 0,
                      size: pipelineRecord.uniformSize || 48,
                    },
                  },
                ],
              });
              bundleEncoder.setBindGroup(0, bindGroup, [dynamicOffset]);
            }

            if (vertexBufferId > 0) {
              const vb = this.buffers.get(vertexBufferId);
              if (!vb) {
                throw new Error(`RecordBundle: unknown vertexBufferId ${vertexBufferId}`);
              }
              bundleEncoder.setVertexBuffer(0, vb);
            }

            if (vertexCount > 0) {
              bundleEncoder.draw(vertexCount, 1, 0, 0);
            }

            const bundle = bundleEncoder.finish();
            this.bundles.set(bundleId, bundle);
            break;
          }

          case OPCODE_EXECUTE_BUNDLES: {
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
                } else if (nextOp === OPCODE_CREATE_BUFFER) {
                  scanCursor += 12;
                } else if (nextOp === OPCODE_WRITE_BUFFER) {
                  scanCursor += 16;
                } else if (nextOp === OPCODE_CREATE_TEXTURE) {
                  scanCursor += 20;
                } else if (nextOp === OPCODE_CREATE_PIPELINE) {
                  scanCursor += 32;
                } else if (nextOp === OPCODE_COPY_TEXTURE_TO_BUFFER) {
                  scanCursor += 24;
                } else if (nextOp === OPCODE_RECORD_BUNDLE) {
                  scanCursor += 28;
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
                  continue;
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
              currentPassTargetKey = `${targetType}:${targetId}`;
              passState = {
                pipelineId: null,
                uniformBufferId: null,
                dynamicOffset: null,
                vertexBufferId: null,
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
            passState = {
              pipelineId: null,
              uniformBufferId: null,
              dynamicOffset: null,
              vertexBufferId: null,
            };
            break;
          }

          default:
            throw new Error(`Unknown opcode: ${opcode}`);
        }
      }

      closeActivePass();
      frameCanvasView = null;
      // Finish and submit synchronously inside the error scope
      const commandBuffer = commandEncoder.finish();
      this.device.queue.submit([commandBuffer]);
    });
  }

  /**
   * Reads back data from a map-readable buffer.
   */
  async readbackBuffer(bufferId, byteLength) {
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
      copy = new Uint8Array(buffer.getMappedRange(0, byteLength).slice(0));
    } finally {
      buffer.unmap();
    }

    copy.epochHi = epochHi;
    copy.epochLo = epochLo;
    copy.bufferId = bufferId;
    copy.data = copy;
    return copy;
  }
}
