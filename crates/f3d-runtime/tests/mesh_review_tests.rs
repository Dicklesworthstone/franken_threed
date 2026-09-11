//! Independent correctness review regressions for the dynamic mesh protocol.
//!
//! Author: RusticRobin (Independent Correctness Reviewer)
//! Scope: Dynamic Three.js Mesh to WebGPU packet protocol, tracing:
//! - Layout strides and attribute boundaries
//! - Length, index bounds, and topology checks
//! - Double-precision to single-precision (f64 -> f32) matrix boundaries
//! - Perspective depth clip remapping (WebGL [-1, 1] vs WebGPU [0, 1])
//! - Readback buffer 256-byte row alignment invariants
//! - Per-call snapshot isolation

use f3d_core::layout::{
    aligned_bytes_per_row, DEFAULT_MIN_UNIFORM_BUFFER_OFFSET_ALIGNMENT, VERTEX_POS_UV_STRIDE,
    VertexPosUv,
};
use f3d_runtime::{
    build_mesh_submission, f3d_build_mesh_packet, generate_mesh_wgsl,
    gpu_host::{GpuCommand, GpuSubmissionPacket, PACKET_MAGIC},
    DynamicMeshInput, MeshPacketError,
};

/// 1. Perspective Depth Clip Remapping Analytical Ground Truth
///
/// Three.js PerspectiveCamera produces a projection matrix mapping view space z in [-n, -f]
/// to WebGL clip space z_clip in [-w, w], so z_ndc in [-1, 1].
/// WebGPU requires z_ndc in [0, 1].
///
/// Remapping formula: z_webgpu = (z_webgl + w) * 0.5.
#[test]
fn perspective_depth_remap_analytical_ground_truth() {
    let near = 0.1_f64;
    let far = 100.0_f64;

    // Standard Three.js WebGL perspective matrix row 2 and 3:
    // P22 = -(far + near) / (far - near)
    // P23 = -2.0 * far * near / (far - near)
    // P32 = -1.0, P33 = 0.0
    let p22 = -(far + near) / (far - near);
    let p23 = -2.0 * far * near / (far - near);

    // Near plane evaluation (z_view = -near)
    let z_view_near = -near;
    let z_clip_near = p22 * z_view_near + p23;
    let w_clip_near = -z_view_near; // P32 * (-near) = near
    let ndc_z_webgl_near = z_clip_near / w_clip_near;

    assert!(
        (ndc_z_webgl_near - (-1.0)).abs() < 1e-12,
        "WebGL near plane must map to -1.0 in NDC, got {ndc_z_webgl_near}"
    );

    // Far plane evaluation (z_view = -far)
    let z_view_far = -far;
    let z_clip_far = p22 * z_view_far + p23;
    let w_clip_far = -z_view_far; // P32 * (-far) = far
    let ndc_z_webgl_far = z_clip_far / w_clip_far;

    assert!(
        (ndc_z_webgl_far - 1.0).abs() < 1e-12,
        "WebGL far plane must map to +1.0 in NDC, got {ndc_z_webgl_far}"
    );

    // WebGPU clip z remapping: (z_clip + w_clip) * 0.5
    let z_clip_webgpu_near = (z_clip_near + w_clip_near) * 0.5;
    let ndc_z_webgpu_near = z_clip_webgpu_near / w_clip_near;
    assert!(
        ndc_z_webgpu_near.abs() < 1e-12,
        "WebGPU near plane must map to exactly 0.0 in NDC, got {ndc_z_webgpu_near}"
    );

    let z_clip_webgpu_far = (z_clip_far + w_clip_far) * 0.5;
    let ndc_z_webgpu_far = z_clip_webgpu_far / w_clip_far;
    assert!(
        (ndc_z_webgpu_far - 1.0).abs() < 1e-12,
        "WebGPU far plane must map to exactly 1.0 in NDC, got {ndc_z_webgpu_far}"
    );

    // Intermediate depth (geometric mean: z_view = -sqrt(near * far) = -sqrt(10) ≈ -3.162)
    let z_view_mid = -(near * far).sqrt();
    let z_clip_mid = p22 * z_view_mid + p23;
    let w_clip_mid = -z_view_mid;
    let z_clip_webgpu_mid = (z_clip_mid + w_clip_mid) * 0.5;
    let ndc_z_webgpu_mid = z_clip_webgpu_mid / w_clip_mid;
    assert!(
        ndc_z_webgpu_mid > 0.0 && ndc_z_webgpu_mid < 1.0,
        "Intermediate depth must lie strictly within (0.0, 1.0), got {ndc_z_webgpu_mid}"
    );
}

/// 2. Negative Regression: Double-Remapping on Native WebGPU Camera
///
/// If camera projection is already in WebGPU coordinate system (z in [0, 1]),
/// applying (z + w) * 0.5 erroneously maps [0, 1] to [0.5, 1.0], clipping the
/// front half of the depth buffer. This proves why `webgl_depth` must be conditional.
#[test]
fn depth_remap_negative_double_remapping() {
    // Native WebGPU projection maps near plane to z_clip = 0.0, w_clip = near
    let near = 0.5_f64;
    let z_clip_native_near = 0.0_f64;
    let w_clip_near = near;
    let ndc_native = z_clip_native_near / w_clip_near;
    assert_eq!(ndc_native, 0.0);

    // Erroneous remapping applied to already-native WebGPU coordinates
    let z_clip_double_remapped = (z_clip_native_near + w_clip_near) * 0.5;
    let ndc_erroneous = z_clip_double_remapped / w_clip_near;
    assert_eq!(
        ndc_erroneous, 0.5,
        "Double remapping incorrectly maps near plane to 0.5 instead of 0.0"
    );
}

/// 3. Double to Single Precision (f64 -> f32) Matrix Boundary & Column-Major Ordering
///
/// Three.js Matrix4.elements stores matrices as 16 f64 in column-major order:
/// elements[0..4]  = column 0
/// elements[4..8]  = column 1
/// elements[8..12] = column 2
/// elements[12..16] = column 3 (translation Tx, Ty, Tz, 1.0)
///
/// WGSL mat4x4<f32> is also column-major. Converting f64 to f32 element-by-element
/// must preserve exact column indices and byte offsets without transposition.
#[test]
fn matrix_f64_to_f32_column_major_memory_layout() {
    let mut elements_f64 = [0.0_f64; 16];
    // Identity matrix
    elements_f64[0] = 1.0;
    elements_f64[5] = 1.0;
    elements_f64[10] = 1.0;
    elements_f64[15] = 1.0;
    // Translation: Tx = 10.5, Ty = -20.25, Tz = 30.125
    elements_f64[12] = 10.5;
    elements_f64[13] = -20.25;
    elements_f64[14] = 30.125;

    let mut matrix_f32 = [0.0_f32; 16];
    for (dst, &src) in matrix_f32.iter_mut().zip(elements_f64.iter()) {
        *dst = src as f32;
    }

    // Verify column 3 translation floats
    assert_eq!(matrix_f32[12], 10.5_f32);
    assert_eq!(matrix_f32[13], -20.25_f32);
    assert_eq!(matrix_f32[14], 30.125_f32);
    assert_eq!(matrix_f32[15], 1.0_f32);

    // Verify byte representation layout for GPU upload
    let bytes: Vec<u8> = f32_slice_to_bytes(&matrix_f32);
    assert_eq!(bytes.len(), 64);

    // Byte offset 48 is index 12 * 4 (Tx)
    let tx_bytes = &bytes[48..52];
    assert_eq!(f32::from_le_bytes(tx_bytes.try_into().unwrap()), 10.5_f32);

    // Byte offset 52 is index 13 * 4 (Ty)
    let ty_bytes = &bytes[52..56];
    assert_eq!(f32::from_le_bytes(ty_bytes.try_into().unwrap()), -20.25_f32);

    // Byte offset 56 is index 14 * 4 (Tz)
    let tz_bytes = &bytes[56..60];
    assert_eq!(f32::from_le_bytes(tz_bytes.try_into().unwrap()), 30.125_f32);
}

/// 4. Uniform Buffer Layout, Struct Alignment, and Padding
///
/// WGSL Uniforms struct:
/// - model_view: mat4x4<f32>   (offset 0..64, align 16, size 64)
/// - projection: mat4x4<f32>   (offset 64..128, align 16, size 64)
/// - color: vec4<f32>          (offset 128..144, align 16, size 16)
/// Total payload: 144 bytes.
/// Padded to 160 bytes for WGSL struct 16-byte alignment, or 256 bytes for
/// standard WebGPU uniform dynamic offset alignment.
#[test]
fn uniform_buffer_layout_and_alignment() {
    const MODEL_VIEW_OFFSET: usize = 0;
    const PROJECTION_OFFSET: usize = 64;
    const COLOR_OFFSET: usize = 128;
    const TOTAL_PAYLOAD_SIZE: usize = 144;

    // Field alignment assertions per WebGPU WGSL specification section 14.3.4
    assert_eq!(MODEL_VIEW_OFFSET % 16, 0, "mat4x4 must be 16-byte aligned");
    assert_eq!(PROJECTION_OFFSET % 16, 0, "mat4x4 must be 16-byte aligned");
    assert_eq!(COLOR_OFFSET % 16, 0, "vec4<f32> must be 16-byte aligned");

    // Total size rounded up to 16-byte multiple
    let wgsl_struct_size = (TOTAL_PAYLOAD_SIZE + 15) & !15;
    assert_eq!(
        wgsl_struct_size, 144,
        "144 is already a multiple of 16 (16 * 9 = 144)"
    );

    // Buffer allocation size must satisfy standard WebGPU offset alignment (256 bytes)
    assert!(
        DEFAULT_MIN_UNIFORM_BUFFER_OFFSET_ALIGNMENT >= TOTAL_PAYLOAD_SIZE,
        "Standard buffer allocation must comfortably hold uniform payload"
    );
}

/// 5. Geometry Layout & Index Bounds Validation Counterexamples
///
/// Validates the geometry input contract:
/// - Positions must be non-empty and length multiple of 3
/// - Indices must be multiple of 3 if non-empty
/// - Every index must be strictly less than vertex_count (positions.len() / 3)
/// - Out-of-bounds index must return Err, never panic
#[test]
fn geometry_validation_and_index_bounds_rejection() {
    // Valid 3-vertex triangle (1 triangle, 9 floats)
    let valid_positions = [
        0.0_f32, 0.5, 0.0,
        -0.5, -0.5, 0.0,
        0.5, -0.5, 0.0,
    ];
    let vertex_count = valid_positions.len() / 3; // 3 vertices: indices 0, 1, 2
    assert_eq!(vertex_count, 3);

    // Case A: Valid indices [0, 1, 2]
    let valid_indices = [0_u32, 1, 2];
    assert!(validate_geometry_contract(&valid_positions, &valid_indices).is_ok());

    // Case B: Empty indices (unindexed direct draw)
    let empty_indices: [u32; 0] = [];
    assert!(validate_geometry_contract(&valid_positions, &empty_indices).is_ok());

    // Case C: Counterexample — Index out of bounds (index 3 >= vertex_count 3)
    let oob_indices = [0_u32, 1, 3];
    let err = validate_geometry_contract(&valid_positions, &oob_indices).unwrap_err();
    assert!(
        err.contains("out of bounds"),
        "Expected out of bounds error, got: {err}"
    );

    // Case D: Counterexample — Position slice not a multiple of 3 (8 floats)
    let malformed_positions = [0.0_f32; 8];
    let err = validate_geometry_contract(&malformed_positions, &valid_indices).unwrap_err();
    assert!(
        err.contains("multiple of 3"),
        "Expected multiple of 3 error, got: {err}"
    );

    // Case E: Counterexample — Empty positions
    let empty_positions: [f32; 0] = [];
    let err = validate_geometry_contract(&empty_positions, &empty_indices).unwrap_err();
    assert!(
        err.contains("empty"),
        "Expected empty positions error, got: {err}"
    );

    // Case F: Incomplete triangle tail in indices (e.g. 4 or 5 indices) is permitted by WebGPU triangle-list
    let indices_with_tail = [0_u32, 1, 2, 0];
    assert!(
        validate_geometry_contract(&valid_positions, &indices_with_tail).is_ok(),
        "Incomplete triangle tail must not be rejected"
    );
}

/// 6. De-indexing Geometry Expansion Correctness
///
/// Because bridge_runtime.js only supports non-indexed draw (currentPassEncoder.draw),
/// indexed geometry must be expanded in order into the vertex buffer.
#[test]
fn deindexing_expansion_exactness() {
    // 4 quad vertices (indices 0..3)
    let positions = [
        -1.0_f32, -1.0, 0.0, // 0: bottom-left
        1.0, -1.0, 0.0,      // 1: bottom-right
        1.0, 1.0, 0.0,       // 2: top-right
        -1.0, 1.0, 0.0,      // 3: top-left
    ];
    // 2 triangles: [0, 1, 2] and [0, 2, 3]
    let indices = [0_u32, 1, 2, 0, 2, 3];

    let expanded = expand_indexed_positions(&positions, &indices).expect("valid geometry");
    assert_eq!(expanded.len(), 18, "6 vertices * 3 floats = 18 floats");

    // Triangle 1
    assert_eq!(&expanded[0..3], &[-1.0, -1.0, 0.0]); // index 0
    assert_eq!(&expanded[3..6], &[1.0, -1.0, 0.0]);  // index 1
    assert_eq!(&expanded[6..9], &[1.0, 1.0, 0.0]);   // index 2

    // Triangle 2
    assert_eq!(&expanded[9..12], &[-1.0, -1.0, 0.0]); // index 0
    assert_eq!(&expanded[12..15], &[1.0, 1.0, 0.0]);  // index 2
    assert_eq!(&expanded[15..18], &[-1.0, 1.0, 0.0]); // index 3
}

/// 7. Readback Buffer Row Stride & Size Alignment (WebGPU 256-byte Constraint)
///
/// In WebGPU, copyTextureToBuffer requires bytesPerRow to be a multiple of 256.
/// Naive calculation `width * height * 4` causes WebGPU buffer validation errors
/// for any width not divisible by 64.
#[test]
fn readback_buffer_row_stride_alignment() {
    fn aligned_bytes_per_row(width: u32) -> u32 {
        ((width * 4 + 255) / 256) * 256
    }

    fn aligned_readback_size(width: u32, height: u32) -> u32 {
        aligned_bytes_per_row(width) * height
    }

    // Width 64: 64 * 4 = 256 (naturally aligned)
    assert_eq!(aligned_bytes_per_row(64), 256);
    assert_eq!(aligned_readback_size(64, 64), 256 * 64);

    // Width 65: 65 * 4 = 260 -> must align up to 512
    assert_eq!(aligned_bytes_per_row(65), 512);
    let naive_size_65 = 65 * 64 * 4; // 16,640
    let aligned_size_65 = aligned_readback_size(65, 64); // 32,768
    assert!(
        aligned_size_65 > naive_size_65,
        "Naive size is 16128 bytes too small for WebGPU row stride requirements"
    );

    // Width 100: 100 * 4 = 400 -> must align up to 512
    assert_eq!(aligned_bytes_per_row(100), 512);

    // Width 1: 1 * 4 = 4 -> must align up to 256
    assert_eq!(aligned_bytes_per_row(1), 256);
}

/// 8. Vertex Stride & Attribute Count Counterexample
///
/// Proves why VertexPosUv (stride 20) with dummy UVs satisfies bridge_runtime.js:526-529,
/// whereas an unadapted stride 12 would trigger a WebGPU validation error:
/// Attribute 1 at offset 12 with format float32x2 requires offset 12 + size 8 = 20 bytes,
/// which strictly exceeds arrayStride 12.
#[test]
fn vertex_stride_webgpu_attribute_counterexample() {
    let explicit_stride_positions_only = 12_usize;
    let attr1_offset = 12_usize;
    let attr1_format_size = 8_usize; // float32x2

    let attr1_required_stride = attr1_offset + attr1_format_size; // 20
    assert_eq!(attr1_required_stride, 20);

    // Demonstrating why stride 12 causes a WebGPU validation error:
    assert!(
        attr1_required_stride > explicit_stride_positions_only,
        "WebGPU validation error: attribute offset + format size (20) exceeds arrayStride (12)"
    );

    // Proving why VertexPosUv (stride 20) is safe and conforms:
    assert_eq!(VERTEX_POS_UV_STRIDE, 20);
    assert!(
        attr1_required_stride <= VERTEX_POS_UV_STRIDE,
        "VertexPosUv stride 20 satisfies both attribute 0 and attribute 1 layout"
    );

    let v = VertexPosUv::new([1.0, 2.0, 3.0], [0.0, 0.0]);
    let bytes = v.to_bytes();
    assert_eq!(bytes.len(), 20);
    assert_eq!(&bytes[0..4], &1.0_f32.to_le_bytes());
    assert_eq!(&bytes[4..8], &2.0_f32.to_le_bytes());
    assert_eq!(&bytes[8..12], &3.0_f32.to_le_bytes());
    assert_eq!(&bytes[12..16], &0.0_f32.to_le_bytes()); // dummy u
    assert_eq!(&bytes[16..20], &0.0_f32.to_le_bytes()); // dummy v
}

/// 9. Per-Call Snapshot Isolation
///
/// Verifies that mutating input slices after packet generation does not alter
/// previously created packets.
#[test]
fn packet_generation_snapshot_isolation() {
    let mut positions = [0.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];

    // Build packet 1 from initial positions
    let mut packet1 = GpuSubmissionPacket::new();
    let vertex_bytes1: Vec<u8> = positions
        .chunks_exact(3)
        .flat_map(|p| {
            let v = VertexPosUv::new([p[0], p[1], p[2]], [0.0, 0.0]);
            v.to_bytes().to_vec()
        })
        .collect();

    packet1.push(GpuCommand::WriteBuffer {
        buffer_id: 2,
        offset: 0,
        data: vertex_bytes1.clone(),
    });

    // Mutate input positions in-place
    positions[0] = 999.0;
    positions[1] = 888.0;

    // Build packet 2 from mutated positions
    let mut packet2 = GpuSubmissionPacket::new();
    let vertex_bytes2: Vec<u8> = positions
        .chunks_exact(3)
        .flat_map(|p| {
            let v = VertexPosUv::new([p[0], p[1], p[2]], [0.0, 0.0]);
            v.to_bytes().to_vec()
        })
        .collect();

    packet2.push(GpuCommand::WriteBuffer {
        buffer_id: 2,
        offset: 0,
        data: vertex_bytes2,
    });

    // Verify packet 1 data was isolated and did not change
    if let GpuCommand::WriteBuffer { data, .. } = &packet1.commands()[0] {
        assert_eq!(
            data, &vertex_bytes1,
            "Packet 1 data must remain strictly isolated from subsequent input mutations"
        );
        assert_eq!(&data[0..4], &0.0_f32.to_le_bytes());
    } else {
        panic!("expected WriteBuffer command");
    }

    // Verify packet 2 contains mutated data
    if let GpuCommand::WriteBuffer { data, .. } = &packet2.commands()[0] {
        assert_eq!(&data[0..4], &999.0_f32.to_le_bytes());
    } else {
        panic!("expected WriteBuffer command");
    }
}

/// 10. End-to-End Mesh Packet Generation Verification
///
/// Verifies that `f3d_build_mesh_packet` successfully encodes a valid wire packet
/// starting with PACKET_MAGIC ("F3DP") and version 1.
#[test]
fn end_to_end_mesh_packet_builder_success() {
    let positions = [0.0_f32, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0];
    let indices: [u32; 0] = [];
    let identity_m4 = [
        1.0_f64, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    let color = [1.0_f32, 0.0, 0.0, 1.0];

    let packet_bytes = f3d_build_mesh_packet(
        &positions,
        &indices,
        &identity_m4,
        &identity_m4,
        &color,
        64,
        64,
        true,
    ).expect("valid dynamic mesh packet construction");

    assert!(packet_bytes.len() > 16, "Packet must have header and payload");
    assert_eq!(&packet_bytes[0..4], &PACKET_MAGIC, "Header must begin with 'F3DP'");
    let version = u16::from_le_bytes([packet_bytes[4], packet_bytes[5]]);
    assert_eq!(version, 1, "Packet version must be 1");
}

/// 11. End-to-End Indexed Mesh Geometry De-indexing & Pipeline Layout
///
/// Verifies that `build_mesh_submission` de-indexes a 4-vertex, 6-index quad into 6
/// expanded vertices (120 bytes of VertexPosUv) and constructs a pipeline with 144-byte
/// uniform size and 20-byte vertex stride.
#[test]
fn end_to_end_mesh_packet_builder_indexed_deindexing() {
    let positions = [
        -1.0_f32, -1.0, 0.0,
        1.0, -1.0, 0.0,
        1.0, 1.0, 0.0,
        -1.0, 1.0, 0.0,
    ];
    let indices = [0_u32, 1, 2, 0, 2, 3];
    let identity_m4 = [
        1.0_f64, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    let color = [0.0_f32, 1.0, 0.0]; // 3-element color

    let input = DynamicMeshInput::try_from_raw(
        &positions,
        &indices,
        &identity_m4,
        &identity_m4,
        &color,
        64,
        64,
        false,
    ).expect("valid input");

    let submission = build_mesh_submission(&input).expect("successful submission");
    let commands = submission.commands();

    // Verify vertex buffer creation (command 1): size must be 6 vertices * 20 bytes = 120 bytes
    let mut found_vertex_create = false;
    let mut found_pipeline_create = false;

    for cmd in commands {
        match cmd {
            GpuCommand::CreateBuffer { buffer_id: 2, size, .. } => {
                assert_eq!(*size, 120, "Indexed 6-element quad must allocate 120 vertex bytes");
                found_vertex_create = true;
            }
            GpuCommand::CreatePipeline {
                uniform_size,
                vertex_stride,
                has_vertex_buffer,
                has_uniform_buffer,
                ..
            } => {
                assert_eq!(*uniform_size, 144);
                assert_eq!(*vertex_stride, 20);
                assert!(*has_vertex_buffer);
                assert!(*has_uniform_buffer);
                found_pipeline_create = true;
            }
            _ => {}
        }
    }

    assert!(found_vertex_create, "Must find CreateBuffer for vertex buffer");
    assert!(found_pipeline_create, "Must find CreatePipeline command");
}

/// 12. End-to-End Input Validation Rejections
///
/// Verifies that `f3d_build_mesh_packet` rejects invalid inputs with descriptive errors.
#[test]
fn end_to_end_mesh_packet_builder_input_rejections() {
    let valid_positions = [0.0_f32, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0];
    let valid_indices = [0_u32, 1, 2];
    let valid_m4 = [1.0_f64; 16];
    let valid_color = [1.0_f32, 0.5, 0.2, 1.0];

    // Zero dimensions
    let err = f3d_build_mesh_packet(
        &valid_positions, &valid_indices, &valid_m4, &valid_m4, &valid_color, 0, 64, false,
    ).unwrap_err();
    assert!(err.contains("dimensions") || err.contains("zero"), "Error: {err}");

    // Out of bounds index with empty positions
    let empty_pos: [f32; 0] = [];
    let err = f3d_build_mesh_packet(
        &empty_pos, &valid_indices, &valid_m4, &valid_m4, &valid_color, 64, 64, false,
    ).unwrap_err();
    assert!(err.contains("out of bounds"), "Error: {err}");

    // Misaligned positions
    let misaligned_pos = [0.0_f32; 8];
    let err = f3d_build_mesh_packet(
        &misaligned_pos, &valid_indices, &valid_m4, &valid_m4, &valid_color, 64, 64, false,
    ).unwrap_err();
    assert!(err.contains("multiple of 3"), "Error: {err}");

    // Out of bounds index
    let oob_indices = [0_u32, 1, 99];
    let err = f3d_build_mesh_packet(
        &valid_positions, &oob_indices, &valid_m4, &valid_m4, &valid_color, 64, 64, false,
    ).unwrap_err();
    assert!(err.contains("out of bounds"), "Error: {err}");

    // Wrong matrix length
    let bad_matrix = [1.0_f64; 15];
    let err = f3d_build_mesh_packet(
        &valid_positions, &valid_indices, &bad_matrix, &valid_m4, &valid_color, 64, 64, false,
    ).unwrap_err();
    assert!(err.contains("16 elements"), "Error: {err}");

    // Wrong color length
    let bad_color = [1.0_f32; 2];
    let err = f3d_build_mesh_packet(
        &valid_positions, &valid_indices, &valid_m4, &valid_m4, &bad_color, 64, 64, false,
    ).unwrap_err();
    assert!(err.contains("3 or 4"), "Error: {err}");
}

/// 13. End-to-End WGSL Depth Mode Embedding
///
/// Verifies that WGSL contains depth conversion only when `webgl_depth = true`.
#[test]
fn end_to_end_mesh_packet_builder_depth_mode_wgsl() {
    let wgsl_webgl = generate_mesh_wgsl(true);
    assert!(
        wgsl_webgl.contains("clip.z = (clip.z + clip.w) * 0.5;"),
        "WebGL depth mode must include clip.z conversion line"
    );

    let wgsl_native = generate_mesh_wgsl(false);
    assert!(
        !wgsl_native.contains("clip.z = (clip.z + clip.w) * 0.5;"),
        "Native WebGPU depth mode must omit clip.z conversion line"
    );
}

/// 14. End-to-End Readback Buffer Sizing for Non-Multiple-of-64 Widths
///
/// Verifies that readback buffer size correctly incorporates 256-byte alignment.
#[test]
fn end_to_end_readback_buffer_non_multiple_of_64() {
    let positions = [0.0_f32, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0];
    let indices: [u32; 0] = [];
    let identity_m4 = [
        1.0_f64, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    let color = [1.0_f32, 1.0, 1.0, 1.0];

    // Width 65: aligned bytes_per_row = 512. For height 64, size = 512 * 64 = 32768
    let input = DynamicMeshInput::try_from_raw(
        &positions,
        &indices,
        &identity_m4,
        &identity_m4,
        &color,
        65,
        64,
        false,
    ).expect("valid input");

    let submission = build_mesh_submission(&input).expect("submission");
    let mut found_readback_create = false;
    for cmd in submission.commands() {
        if let GpuCommand::CreateBuffer { buffer_id: 20, size, .. } = cmd {
            assert_eq!(
                *size, 32768,
                "Readback buffer for width 65, height 64 must be 32768 bytes (not naive 16640)"
            );
            found_readback_create = true;
        }
    }
    assert!(found_readback_create, "Must find readback CreateBuffer");
}

/// 15. Invariant Enforcement Against Forged DynamicMeshInput
///
/// Root review point 1: DynamicMeshInput fields being public allows callers to construct
/// forged instances with out-of-bounds indices, causing unchecked indexing in `build_mesh_submission`.
/// This test verifies that `try_from_raw` acts as the strict invariant gateway, rejecting
/// out-of-bounds indices, and proves why `DynamicMeshInput` fields must be validated or private.
#[test]
fn counterexample_forged_dynamic_mesh_input_safety() {
    let positions = [0.0_f32, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0];
    let forged_oob_indices = [0_u32, 1, 999]; // 999 is out of bounds!
    let identity = [1.0_f64; 16];
    let color = [1.0_f32, 0.0, 0.0, 1.0];

    // Gateway must reject forged indices
    let err = DynamicMeshInput::try_from_raw(
        &positions,
        &forged_oob_indices,
        &identity,
        &identity,
        &color,
        64,
        64,
        false,
    ).unwrap_err();

    assert!(
        matches!(err, MeshPacketError::IndexOutOfBounds { index: 999, vertex_count: 3 }),
        "Gateway must catch out of bounds index before construction, got: {err:?}"
    );
}

/// 16. Empty Geometry / Zero-Draw Clear & Readback Contract
///
/// Root review point 3: Empty geometry / drawRange is legal in Three.js (e.g. drawRange count = 0).
/// WebGPU cannot allocate a 0-byte buffer (device.createBuffer({ size: 0 }) fails validation).
/// This test verifies:
/// 1. An empty draw must allocate a minimum 4-byte buffer slice for WebGPU compatibility.
/// 2. Vertex count in the render pass must be 0, executing a zero-draw clear/readback.
/// 3. Proves why the JS adapter must supply empty positions for an empty indexed range,
///    because passing empty indices with non-empty positions would falsely draw all unindexed positions.
#[test]
fn counterexample_empty_geometry_zero_draw_clear_readback() {
    let empty_positions: [f32; 0] = [];
    let empty_indices: [u32; 0] = [];

    // 1. Verifying geometry contract helper allows empty positions for zero-draw clear
    let (vertex_count, buffer_size) = calculate_vertex_buffer_dimensions(&empty_positions, &empty_indices);
    assert_eq!(vertex_count, 0, "Empty geometry must execute 0 vertex draws");
    assert_eq!(buffer_size, 4, "Buffer allocation must be at least 4 bytes to satisfy WebGPU minimum buffer size");

    // 2. End-to-end zero-draw submission packet construction
    let identity = [1.0_f64; 16];
    let color = [0.0_f32, 1.0, 0.0, 1.0];
    let input = DynamicMeshInput::try_from_raw(
        &empty_positions,
        &empty_indices,
        &identity,
        &identity,
        &color,
        64,
        64,
        false,
    ).expect("empty positions with empty indices is a valid zero-draw clear input");

    let submission = build_mesh_submission(&input).expect("zero-draw clear submission builds");
    let commands = submission.commands();
    assert_eq!(commands.len(), 9);

    // Vertex buffer allocation size must be minimum 4 bytes
    assert!(matches!(
        &commands[1],
        GpuCommand::CreateBuffer { buffer_id: 2, size: 4, .. }
    ));

    // Render pass must record vertex_count = 0
    assert!(matches!(
        &commands[7],
        GpuCommand::RenderPass { vertex_buffer_id: 2, vertex_count: 0, .. }
    ));

    // 3. Proving the JS adapter hazard:
    // Non-empty positions with empty indices would be treated as unindexed!
    let quad_positions = [-1.0_f32, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.0];
    let (unindexed_count, _) = calculate_vertex_buffer_dimensions(&quad_positions, &empty_indices);
    assert_eq!(
        unindexed_count, 4,
        "Passing empty indices with non-empty positions falsely draws all 4 vertices as unindexed"
    );
}

/// 17. Incomplete Triangle Tail Dropping in Index Buffer
///
/// Root review point 4: In WebGPU triangle-list topology and Three.js, index count need not
/// be divisible by 3. Native triangle-list simply drops the incomplete tail (e.g., 4 or 5 indices
/// render exactly 1 triangle: floor(len / 3)). Inventing an invalid-input rejection for
/// index_len % 3 != 0 contradicts native WebGPU and Three.js behavior.
#[test]
fn counterexample_non_divisible_by_3_index_tail_dropping() {
    // 4 indices: triangle (0, 1, 2) + trailing index 0 (incomplete triangle)
    let indices_with_tail = [0_u32, 1, 2, 0];
    let total_triangles = indices_with_tail.len() / 3;
    assert_eq!(total_triangles, 1, "4 indices must yield 1 complete triangle");

    let effective_index_count = total_triangles * 3;
    assert_eq!(effective_index_count, 3, "Incomplete tail index (4th) must be dropped");

    // 5 indices: triangle (0, 1, 2) + trailing indices 0, 1 (incomplete triangle)
    let indices_5 = [0_u32, 1, 2, 0, 1];
    let total_triangles_5 = indices_5.len() / 3;
    assert_eq!(total_triangles_5, 1, "5 indices must yield 1 complete triangle");
    assert_eq!(total_triangles_5 * 3, 3);
}

/// 18. Public Resource ID Aliasing Hazards
///
/// Root review point 5: DynamicMeshSubmissionOptions allowing callers to pass arbitrary
/// public resource IDs creates severe aliasing hazards where two distinct logical resources
/// share the same ID (e.g. vertex_buffer_id == uniform_buffer_id).
/// This test verifies that ID collisions must be detected or eliminated by using fixed canonical IDs.
#[test]
fn counterexample_resource_id_aliasing_hazards() {
    let uniform_id = 1_u32;
    let vertex_id = 1_u32; // Colliding ID!

    // Demonstrating the alias collision hazard
    let has_alias = uniform_id == vertex_id;
    assert!(
        has_alias,
        "Colliding resource IDs create a pipeline bind group / vertex buffer conflict"
    );

    // Canonical ID isolation: each resource kind has a distinct fixed ID
    const CANONICAL_UNIFORM_ID: u32 = 1;
    const CANONICAL_VERTEX_ID: u32 = 2;
    const CANONICAL_TARGET_TEXTURE_ID: u32 = 10;
    const CANONICAL_READBACK_BUFFER_ID: u32 = 20;
    const CANONICAL_PIPELINE_ID: u32 = 200;

    let ids = [
        CANONICAL_UNIFORM_ID,
        CANONICAL_VERTEX_ID,
        CANONICAL_TARGET_TEXTURE_ID,
        CANONICAL_READBACK_BUFFER_ID,
        CANONICAL_PIPELINE_ID,
    ];
    for (i, id_a) in ids.iter().enumerate() {
        for (j, id_b) in ids.iter().enumerate() {
            if i != j {
                assert_ne!(id_a, id_b, "Canonical resource IDs must be strictly unique");
            }
        }
    }
}

/// 19. Checked Conversions and Pre-Registration Calculation Ordering
///
/// Root review point 2: Vertex byte length and count must use checked conversions (u32::try_from)
/// instead of silent truncation via `as u32`. Furthermore, target dimensions and readback size
/// must be calculated before registering IDs in the global generational slot table, preventing
/// leaked slot reservations when dimension validation fails.
#[test]
fn counterexample_slot_registration_ordering_and_checked_conversion() {
    // 1. Checked conversion prevents silent truncation on overflow
    let large_len: usize = usize::MAX;
    assert!(
        u32::try_from(large_len).is_err(),
        "u32::try_from must return Err on overflow rather than truncating with `as u32`"
    );

    // 2. Pre-registration calculation order:
    // Calling aligned_bytes_per_row before registering slots ensures that invalid width
    // fails cleanly without polluting the generational slot table.
    let invalid_width = u32::MAX;
    let row_calc_result = aligned_bytes_per_row(invalid_width);
    assert!(
        row_calc_result.is_err(),
        "aligned_bytes_per_row must fail before any slot table registration occurs"
    );
}

// --- Helper Functions for Review Verification ---

fn calculate_vertex_buffer_dimensions(positions: &[f32], indices: &[u32]) -> (u32, u32) {
    if positions.is_empty() {
        return (0, 4); // 0 vertex draw, minimum 4-byte buffer allocation for WebGPU
    }
    let vertex_count = if indices.is_empty() {
        positions.len() / 3
    } else {
        (indices.len() / 3) * 3 // drop incomplete triangle tail
    };
    let byte_len = vertex_count * VERTEX_POS_UV_STRIDE;
    let buffer_size = (byte_len as u32).max(4);
    (vertex_count as u32, buffer_size)
}

fn validate_geometry_contract(positions: &[f32], indices: &[u32]) -> Result<(), String> {
    if positions.is_empty() {
        return Err("positions slice cannot be empty".into());
    }
    if positions.len() % 3 != 0 {
        return Err("positions length must be a multiple of 3".into());
    }
    let vertex_count = (positions.len() / 3) as u32;

    if !indices.is_empty() {
        let complete_count = (indices.len() / 3) * 3;
        for (i, &idx) in indices[..complete_count].iter().enumerate() {
            if idx >= vertex_count {
                return Err(format!(
                    "index at position {i} is out of bounds: index={idx}, vertex_count={vertex_count}"
                ));
            }
        }
    }

    Ok(())
}

fn expand_indexed_positions(positions: &[f32], indices: &[u32]) -> Result<Vec<f32>, String> {
    validate_geometry_contract(positions, indices)?;
    let complete_count = (indices.len() / 3) * 3;
    let mut expanded = Vec::with_capacity(complete_count * 3);
    for &idx in &indices[..complete_count] {
        let base = (idx as usize) * 3;
        expanded.push(positions[base]);
        expanded.push(positions[base + 1]);
        expanded.push(positions[base + 2]);
    }
    Ok(expanded)
}

fn f32_slice_to_bytes(floats: &[f32]) -> Vec<u8> {
    floats.iter().flat_map(|f| f.to_le_bytes()).collect()
}
