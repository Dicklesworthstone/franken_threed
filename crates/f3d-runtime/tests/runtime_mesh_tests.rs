//! Focused unit and integration tests for dynamic mesh-to-Wasm packet building.
//!
//! Enforces:
//! 1. Dynamic input processing for unindexed and indexed geometries.
//! 2. f64 to f32 upload boundary quantization and column-major memory layout.
//! 3. WebGL-to-WebGPU depth conversion conditional shader emission.
//! 4. Input-dependent mutations in geometry, color, and matrices.
//! 5. Input validation and error discipline (bounds, lengths, dimensions).
//! 6. Exact parity between `f3d_build_mesh_packet` and `gpu_bridge_build_mesh_packet`.

use f3d_runtime::gpu_host::{
    GpuCommand, GpuSubmissionPacket, BUFFER_USAGE_COPY_DST, BUFFER_USAGE_MAP_READ,
    BUFFER_USAGE_UNIFORM, BUFFER_USAGE_VERTEX, TARGET_CANVAS,
    TARGET_FORMAT_PREFERRED_CANVAS, TARGET_FORMAT_RGBA8UNORM, TEXTURE_USAGE_COPY_SRC,
    TEXTURE_USAGE_RENDER_ATTACHMENT,
};
use f3d_runtime::mesh::{
    build_mesh_canvas_submission, build_mesh_submission, f3d_build_canvas_mesh_packet,
    f3d_build_mesh_packet, generate_mesh_wgsl, gpu_bridge_build_canvas_mesh_packet,
    gpu_bridge_build_mesh_packet, DynamicMeshInput, MeshPacketError,
    MESH_CANVAS_PIPELINE_ID, MESH_CANVAS_TARGET_ID, MESH_CLEAR_COLOR,
    MESH_UNIFORM_BUFFER_ID, MESH_VERTEX_BUFFER_ID,
};

const IDENTITY_F64: [f64; 16] = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
];

#[test]
fn test_dynamic_mesh_unindexed_packet_structure() {
    let positions = [
        0.0_f32, 0.5, 0.0,
        -0.5, -0.5, 0.0,
        0.5, -0.5, 0.0,
    ];
    let indices: [u32; 0] = [];
    let color = [0.0_f32, 1.0, 0.0, 1.0];

    let bytes = f3d_build_mesh_packet(
        &positions,
        &indices,
        &IDENTITY_F64,
        &IDENTITY_F64,
        &color,
        64,
        64,
        false,
    )
    .expect("f3d_build_mesh_packet must succeed");

    assert!(!bytes.is_empty());

    let input = DynamicMeshInput::try_from_raw(
        &positions,
        &indices,
        &IDENTITY_F64,
        &IDENTITY_F64,
        &color,
        64,
        64,
        false,
    )
    .expect("DynamicMeshInput must construct");

    let packet = build_mesh_submission(&input).expect("submission must build");
    let commands = packet.commands();
    assert_eq!(commands.len(), 9);

    // 0: Create uniform buffer
    assert!(matches!(
        &commands[0],
        GpuCommand::CreateBuffer {
            buffer_id: 1,
            size: 256,
            usage,
        } if *usage == (BUFFER_USAGE_UNIFORM | BUFFER_USAGE_COPY_DST)
    ));

    // 1: Create vertex buffer (3 vertices * 20 bytes = 60 bytes)
    assert!(matches!(
        &commands[1],
        GpuCommand::CreateBuffer {
            buffer_id: 2,
            size: 60,
            usage,
        } if *usage == (BUFFER_USAGE_VERTEX | BUFFER_USAGE_COPY_DST)
    ));

    // 2: Write vertex buffer
    if let GpuCommand::WriteBuffer { buffer_id, offset, data } = &commands[2] {
        assert_eq!(*buffer_id, 2);
        assert_eq!(*offset, 0);
        assert_eq!(data.len(), 60);

        // Verify first vertex position [0.0, 0.5, 0.0]
        let v0_x = f32::from_le_bytes(data[0..4].try_into().unwrap());
        let v0_y = f32::from_le_bytes(data[4..8].try_into().unwrap());
        let v0_z = f32::from_le_bytes(data[8..12].try_into().unwrap());
        assert_eq!([v0_x, v0_y, v0_z], [0.0, 0.5, 0.0]);
    } else {
        panic!("expected WriteBuffer at index 2");
    }

    // 3: Create target texture (64x64 RGBA8UNORM)
    assert!(matches!(
        &commands[3],
        GpuCommand::CreateTexture {
            texture_id: 10,
            width: 64,
            height: 64,
            format: TARGET_FORMAT_RGBA8UNORM,
            usage,
        } if *usage == (TEXTURE_USAGE_RENDER_ATTACHMENT | TEXTURE_USAGE_COPY_SRC)
    ));

    // 4: Create readback buffer (aligned bytes_per_row = 256, size = 256 * 64 = 16384)
    assert!(matches!(
        &commands[4],
        GpuCommand::CreateBuffer {
            buffer_id: 20,
            size: 16384,
            usage,
        } if *usage == (BUFFER_USAGE_MAP_READ | BUFFER_USAGE_COPY_DST)
    ));

    // 5: Create pipeline (uniform_size: 144, vertex_stride: 20)
    assert!(matches!(
        &commands[5],
        GpuCommand::CreatePipeline {
            pipeline_id: 100,
            target_format: TARGET_FORMAT_RGBA8UNORM,
            has_vertex_buffer: true,
            has_uniform_buffer: true,
            uniform_size: 144,
            vertex_stride: 20,
            ..
        }
    ));

    // 6: Write uniform arena (offset 0, 256 bytes)
    if let GpuCommand::WriteBuffer { buffer_id, offset, data } = &commands[6] {
        assert_eq!(*buffer_id, 1);
        assert_eq!(*offset, 0);
        assert_eq!(data.len(), 256);

        // Verify color at offset 128..144 is [0.0, 1.0, 0.0, 1.0]
        let r = f32::from_le_bytes(data[128..132].try_into().unwrap());
        let g = f32::from_le_bytes(data[132..136].try_into().unwrap());
        let b = f32::from_le_bytes(data[136..140].try_into().unwrap());
        let a = f32::from_le_bytes(data[140..144].try_into().unwrap());
        assert_eq!([r, g, b, a], [0.0, 1.0, 0.0, 1.0]);
    } else {
        panic!("expected WriteBuffer at index 6");
    }

    // 7: RenderPass
    assert!(matches!(
        &commands[7],
        GpuCommand::RenderPass {
            target_id: 10,
            pipeline_id: 100,
            vertex_buffer_id: 2,
            vertex_count: 3,
            uniform_dynamic_offset: 0,
            uniform_buffer_id: 1,
            ..
        }
    ));

    // 8: CopyTextureToBuffer
    assert!(matches!(
        &commands[8],
        GpuCommand::CopyTextureToBuffer {
            texture_id: 10,
            buffer_id: 20,
            width: 64,
            height: 64,
            ..
        }
    ));
}

#[test]
fn test_dynamic_mesh_indexed_quad_expansion() {
    let positions = [
        -1.0_f32, -1.0, 0.0, // 0: bottom-left
        1.0, -1.0, 0.0,      // 1: bottom-right
        1.0, 1.0, 0.0,       // 2: top-right
        -1.0, 1.0, 0.0,      // 3: top-left
    ];
    let indices = [0_u32, 1, 2, 0, 2, 3];
    let color = [1.0_f32, 0.5, 0.25]; // length 3 (alpha defaults to 1.0)

    let input = DynamicMeshInput::try_from_raw(
        &positions,
        &indices,
        &IDENTITY_F64,
        &IDENTITY_F64,
        &color,
        128,
        128,
        false,
    )
    .expect("indexed quad must construct");

    assert_eq!(input.color(), &[1.0, 0.5, 0.25, 1.0]);

    let packet = build_mesh_submission(&input).expect("submission must build");
    let commands = packet.commands();

    // Vertex buffer size for 6 expanded vertices: 6 * 20 = 120 bytes
    assert!(matches!(
        &commands[1],
        GpuCommand::CreateBuffer { buffer_id: 2, size: 120, .. }
    ));

    if let GpuCommand::WriteBuffer { buffer_id, offset, data } = &commands[2] {
        assert_eq!(*buffer_id, 2);
        assert_eq!(*offset, 0);
        assert_eq!(data.len(), 120);

        // Verify vertex 0: positions[0] = [-1.0, -1.0, 0.0]
        let p0_x = f32::from_le_bytes(data[0..4].try_into().unwrap());
        assert_eq!(p0_x, -1.0);

        // Verify vertex 1: positions[1] = [1.0, -1.0, 0.0] (offset 20..40)
        let p1_x = f32::from_le_bytes(data[20..24].try_into().unwrap());
        assert_eq!(p1_x, 1.0);

        // Verify vertex 4: index 2 -> positions[2] = [1.0, 1.0, 0.0] (offset 80..100)
        let p4_y = f32::from_le_bytes(data[84..88].try_into().unwrap());
        assert_eq!(p4_y, 1.0);
    } else {
        panic!("expected WriteBuffer at index 2");
    }

    // Render pass vertex count must be 6
    assert!(matches!(
        &commands[7],
        GpuCommand::RenderPass { vertex_count: 6, .. }
    ));
}

#[test]
fn test_dynamic_mesh_webgl_depth_remap_emission() {
    let wgsl_webgl = generate_mesh_wgsl(true);
    assert!(
        wgsl_webgl.contains("clip.z = (clip.z + clip.w) * 0.5;"),
        "WebGL coordinate system must emit depth remapping"
    );

    let wgsl_webgpu = generate_mesh_wgsl(false);
    assert!(
        !wgsl_webgpu.contains("clip.z = (clip.z + clip.w) * 0.5;"),
        "WebGPU native coordinate system must not emit depth remapping"
    );
}

#[test]
fn test_dynamic_mesh_input_dependent_regressions() {
    let pos_a = [0.0_f32, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0];
    let pos_b = [10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0];
    let empty_indices: [u32; 0] = [];
    let color_a = [1.0_f32, 0.0, 0.0, 1.0];
    let color_b = [0.0_f32, 0.0, 1.0, 1.0];

    // 1. Geometry change produces distinct vertex buffer payloads
    let bytes_a = f3d_build_mesh_packet(
        &pos_a,
        &empty_indices,
        &IDENTITY_F64,
        &IDENTITY_F64,
        &color_a,
        64,
        64,
        false,
    )
    .unwrap();

    let bytes_b = f3d_build_mesh_packet(
        &pos_b,
        &empty_indices,
        &IDENTITY_F64,
        &IDENTITY_F64,
        &color_a,
        64,
        64,
        false,
    )
    .unwrap();

    assert_ne!(bytes_a, bytes_b, "different geometry must produce different packet bytes");

    // 2. Color change produces distinct uniform buffer payloads
    let bytes_color = f3d_build_mesh_packet(
        &pos_a,
        &empty_indices,
        &IDENTITY_F64,
        &IDENTITY_F64,
        &color_b,
        64,
        64,
        false,
    )
    .unwrap();

    assert_ne!(bytes_a, bytes_color, "different color must produce different packet bytes");

    // 3. Matrix change produces distinct uniform buffer payloads
    let mut translated_mv = IDENTITY_F64;
    translated_mv[12] = 5.0; // Tx
    translated_mv[13] = -3.0; // Ty

    let bytes_matrix = f3d_build_mesh_packet(
        &pos_a,
        &empty_indices,
        &translated_mv,
        &IDENTITY_F64,
        &color_a,
        64,
        64,
        false,
    )
    .unwrap();

    assert_ne!(bytes_a, bytes_matrix, "different matrix must produce different packet bytes");

    // 4. WebGL depth flag toggles shader code in pipeline
    let bytes_depth = f3d_build_mesh_packet(
        &pos_a,
        &empty_indices,
        &IDENTITY_F64,
        &IDENTITY_F64,
        &color_a,
        64,
        64,
        true,
    )
    .unwrap();

    assert_ne!(bytes_a, bytes_depth, "webgl_depth toggle must produce different packet bytes");
}

#[test]
fn test_dynamic_mesh_input_validation_errors() {
    let valid_pos = [0.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let valid_indices = [0_u32, 1, 2];
    let valid_color = [1.0_f32, 1.0, 1.0, 1.0];

    // Zero width
    let err = DynamicMeshInput::try_from_raw(
        &valid_pos,
        &valid_indices,
        &IDENTITY_F64,
        &IDENTITY_F64,
        &valid_color,
        0,
        64,
        false,
    )
    .unwrap_err();
    assert!(matches!(err, MeshPacketError::ZeroDimensions { width: 0, height: 64 }));

    // Zero height
    let err = DynamicMeshInput::try_from_raw(
        &valid_pos,
        &valid_indices,
        &IDENTITY_F64,
        &IDENTITY_F64,
        &valid_color,
        64,
        0,
        false,
    )
    .unwrap_err();
    assert!(matches!(err, MeshPacketError::ZeroDimensions { width: 64, height: 0 }));

    // Positions length not divisible by 3
    let err = DynamicMeshInput::try_from_raw(
        &[0.0, 1.0, 2.0, 3.0],
        &[],
        &IDENTITY_F64,
        &IDENTITY_F64,
        &valid_color,
        64,
        64,
        false,
    )
    .unwrap_err();
    assert!(matches!(err, MeshPacketError::InvalidPositionLength { len: 4 }));

    // Index out of bounds (vertex_count = 3, index = 3)
    let err = DynamicMeshInput::try_from_raw(
        &valid_pos,
        &[0, 1, 3],
        &IDENTITY_F64,
        &IDENTITY_F64,
        &valid_color,
        64,
        64,
        false,
    )
    .unwrap_err();
    assert!(matches!(
        err,
        MeshPacketError::IndexOutOfBounds { index: 3, vertex_count: 3 }
    ));

    // Invalid model_view length
    let err = DynamicMeshInput::try_from_raw(
        &valid_pos,
        &valid_indices,
        &[1.0; 15],
        &IDENTITY_F64,
        &valid_color,
        64,
        64,
        false,
    )
    .unwrap_err();
    assert!(matches!(
        err,
        MeshPacketError::InvalidMatrixLength { name: "model_view", len: 15 }
    ));

    // Invalid projection length
    let err = DynamicMeshInput::try_from_raw(
        &valid_pos,
        &valid_indices,
        &IDENTITY_F64,
        &[1.0; 9],
        &valid_color,
        64,
        64,
        false,
    )
    .unwrap_err();
    assert!(matches!(
        err,
        MeshPacketError::InvalidMatrixLength { name: "projection", len: 9 }
    ));

    // Invalid color length
    let err = DynamicMeshInput::try_from_raw(
        &valid_pos,
        &valid_indices,
        &IDENTITY_F64,
        &IDENTITY_F64,
        &[1.0, 0.0],
        64,
        64,
        false,
    )
    .unwrap_err();
    assert!(matches!(err, MeshPacketError::InvalidColorLength { len: 2 }));
}

#[test]
fn test_dynamic_mesh_empty_geometry_zero_draw_clear_readback() {
    let empty_pos: [f32; 0] = [];
    let empty_indices: [u32; 0] = [];
    let color = [0.0_f32, 0.0, 0.0, 1.0];

    // Empty positions and empty indices must construct successfully
    let input = DynamicMeshInput::try_from_raw(
        &empty_pos,
        &empty_indices,
        &IDENTITY_F64,
        &IDENTITY_F64,
        &color,
        64,
        64,
        false,
    )
    .expect("empty geometry is legal and must construct");

    assert_eq!(input.positions().len(), 0);
    assert_eq!(input.indices().len(), 0);

    let packet = build_mesh_submission(&input).expect("zero-draw submission must build");
    let commands = packet.commands();

    // Vertex buffer must be allocated with minimum 4 bytes for WebGPU compliance
    assert!(matches!(
        &commands[1],
        GpuCommand::CreateBuffer { buffer_id: 2, size: 4, .. }
    ));

    // Vertex write buffer must write 4 zero bytes
    if let GpuCommand::WriteBuffer { buffer_id, offset, data } = &commands[2] {
        assert_eq!(*buffer_id, 2);
        assert_eq!(*offset, 0);
        assert_eq!(data.len(), 4);
        assert_eq!(data.as_slice(), &[0, 0, 0, 0]);
    } else {
        panic!("expected WriteBuffer for vertex buffer");
    }

    // Render pass must have vertex_count: 0 (zero-draw clear pass)
    assert!(matches!(
        &commands[7],
        GpuCommand::RenderPass { vertex_count: 0, .. }
    ));

    // Packet must encode without errors
    let encoded = packet.encode().expect("zero-draw packet must encode");
    assert!(!encoded.is_empty());

    // Export function must also succeed
    let export_bytes = f3d_build_mesh_packet(
        &empty_pos,
        &empty_indices,
        &IDENTITY_F64,
        &IDENTITY_F64,
        &color,
        64,
        64,
        false,
    )
    .expect("f3d_build_mesh_packet must succeed on empty geometry");
    assert_eq!(export_bytes, encoded);
}

#[test]
fn test_dynamic_mesh_incomplete_tail_indices_dropped() {
    let positions = [
        0.0_f32, 0.5, 0.0,
        -0.5, -0.5, 0.0,
        0.5, -0.5, 0.0,
    ];
    let color = [1.0_f32, 1.0, 1.0, 1.0];

    // 4 indices with 3 vertices: 1 triangle (indices 0, 1, 2), trailing index 0 is dropped
    let indices_4 = [0_u32, 1, 2, 0];
    let input_4 = DynamicMeshInput::try_from_raw(
        &positions,
        &indices_4,
        &IDENTITY_F64,
        &IDENTITY_F64,
        &color,
        64,
        64,
        false,
    )
    .expect("incomplete tail indices are legal and dropped");

    let packet_4 = build_mesh_submission(&input_4).expect("must build packet");
    let commands_4 = packet_4.commands();

    // 1 triangle = 3 vertices = 60 bytes
    assert!(matches!(
        &commands_4[1],
        GpuCommand::CreateBuffer { buffer_id: 2, size: 60, .. }
    ));
    assert!(matches!(
        &commands_4[7],
        GpuCommand::RenderPass { vertex_count: 3, .. }
    ));

    // 2 indices with 3 vertices: 0 triangles formed, 0 vertices drawn (zero-draw pass)
    let indices_2 = [0_u32, 1];
    let input_2 = DynamicMeshInput::try_from_raw(
        &positions,
        &indices_2,
        &IDENTITY_F64,
        &IDENTITY_F64,
        &color,
        64,
        64,
        false,
    )
    .expect("sub-triangle indices are legal and dropped");

    let packet_2 = build_mesh_submission(&input_2).expect("must build packet");
    let commands_2 = packet_2.commands();

    // 0 vertices: buffer has min 4 bytes, vertex_count: 0
    assert!(matches!(
        &commands_2[1],
        GpuCommand::CreateBuffer { buffer_id: 2, size: 4, .. }
    ));
    assert!(matches!(
        &commands_2[7],
        GpuCommand::RenderPass { vertex_count: 0, .. }
    ));
}

#[test]
fn test_f3d_and_gpu_bridge_alias_equivalence() {
    let positions = [0.0_f32, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0];
    let indices: [u32; 0] = [];
    let color = [0.25_f32, 0.5, 0.75, 1.0];

    let bytes_f3d = f3d_build_mesh_packet(
        &positions,
        &indices,
        &IDENTITY_F64,
        &IDENTITY_F64,
        &color,
        64,
        64,
        false,
    )
    .unwrap();

    let bytes_bridge = gpu_bridge_build_mesh_packet(
        &positions,
        &indices,
        &IDENTITY_F64,
        &IDENTITY_F64,
        &color,
        64,
        64,
        false,
    )
    .unwrap();

    assert_eq!(
        bytes_f3d, bytes_bridge,
        "f3d and gpu_bridge exports must produce identical binary packets"
    );
}

#[test]
fn test_f64_to_f32_precision_and_alignment() {
    let positions = [0.0_f32, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let indices: [u32; 0] = [];

    let mut model_view = IDENTITY_F64;
    model_view[0] = 2.5;
    model_view[12] = 10.125;
    model_view[13] = -20.25;

    let mut projection = IDENTITY_F64;
    projection[5] = 1.7320508075688772; // sqrt(3)
    projection[10] = -1.002002002002002;
    projection[14] = -0.2002002002002002;

    let color = [0.1_f32, 0.2, 0.3, 0.4];

    let input = DynamicMeshInput::try_from_raw(
        &positions,
        &indices,
        &model_view,
        &projection,
        &color,
        64,
        64,
        true,
    )
    .unwrap();

    let packet = build_mesh_submission(&input).unwrap();
    let commands = packet.commands();

    // Find the uniform WriteBuffer command (index 6)
    if let GpuCommand::WriteBuffer { buffer_id, offset, data } = &commands[6] {
        assert_eq!(*buffer_id, 1);
        assert_eq!(*offset, 0);
        assert_eq!(data.len(), 256);

        // Check model_view[0] = 2.5
        let mv0 = f32::from_le_bytes(data[0..4].try_into().unwrap());
        assert_eq!(mv0, 2.5_f32);

        // Check model_view[12] (Tx) = 10.125
        let tx = f32::from_le_bytes(data[48..52].try_into().unwrap());
        assert_eq!(tx, 10.125_f32);

        // Check projection[5] = sqrt(3) as f32
        let proj5 = f32::from_le_bytes(data[84..88].try_into().unwrap());
        assert_eq!(proj5, 1.7320508_f32);

        // Check color at offset 128..144
        let c0 = f32::from_le_bytes(data[128..132].try_into().unwrap());
        let c1 = f32::from_le_bytes(data[132..136].try_into().unwrap());
        let c2 = f32::from_le_bytes(data[136..140].try_into().unwrap());
        let c3 = f32::from_le_bytes(data[140..144].try_into().unwrap());
        assert_eq!([c0, c1, c2, c3], [0.1, 0.2, 0.3, 0.4]);
    } else {
        panic!("expected WriteBuffer for uniform buffer");
    }
}

#[test]
fn test_dynamic_mesh_canvas_packet_structure() {
    let positions = [
        0.0_f32, 0.5, 0.0,
        -0.5, -0.5, 0.0,
        0.5, -0.5, 0.0,
    ];
    let indices: [u32; 0] = [];
    let color = [1.0_f32, 0.0, 0.0, 1.0];

    let input = DynamicMeshInput::try_from_raw(
        &positions,
        &indices,
        &IDENTITY_F64,
        &IDENTITY_F64,
        &color,
        64,
        64,
        false,
    )
    .expect("valid dynamic mesh input");

    let packet = build_mesh_canvas_submission(&input)
        .expect("build_mesh_canvas_submission must succeed");

    let commands = packet.commands();
    // Exactly 6 commands:
    // 0: CreateBuffer (uniform)
    // 1: CreateBuffer (vertex)
    // 2: WriteBuffer (vertex)
    // 3: CreatePipeline (canvas preferred format)
    // 4: WriteBuffer (uniform arena)
    // 5: RenderPass (target_type: TARGET_CANVAS)
    assert_eq!(commands.len(), 6, "expected exactly 6 canvas submission commands");

    // Command 0: Create uniform buffer (MESH_UNIFORM_BUFFER_ID = 1, size 256)
    match &commands[0] {
        GpuCommand::CreateBuffer { buffer_id, size, usage } => {
            assert_eq!(*buffer_id, MESH_UNIFORM_BUFFER_ID);
            assert_eq!(*size, 256);
            assert_eq!(*usage, BUFFER_USAGE_UNIFORM | BUFFER_USAGE_COPY_DST);
        }
        other => panic!("expected CreateBuffer for uniform, got {other:?}"),
    }

    // Command 1: Create vertex buffer (MESH_VERTEX_BUFFER_ID = 2, size 60)
    match &commands[1] {
        GpuCommand::CreateBuffer { buffer_id, size, usage } => {
            assert_eq!(*buffer_id, MESH_VERTEX_BUFFER_ID);
            assert_eq!(*size, 60);
            assert_eq!(*usage, BUFFER_USAGE_VERTEX | BUFFER_USAGE_COPY_DST);
        }
        other => panic!("expected CreateBuffer for vertex, got {other:?}"),
    }

    // Command 2: Write vertex buffer
    match &commands[2] {
        GpuCommand::WriteBuffer { buffer_id, offset, data } => {
            assert_eq!(*buffer_id, MESH_VERTEX_BUFFER_ID);
            assert_eq!(*offset, 0);
            assert_eq!(data.len(), 60);
        }
        other => panic!("expected WriteBuffer for vertex, got {other:?}"),
    }

    // Command 3: Create pipeline for canvas presentation
    match &commands[3] {
        GpuCommand::CreatePipeline {
            pipeline_id,
            target_format,
            has_vertex_buffer,
            has_uniform_buffer,
            uniform_size,
            vertex_stride,
            ..
        } => {
            assert_eq!(*pipeline_id, MESH_CANVAS_PIPELINE_ID);
            assert_eq!(*target_format, TARGET_FORMAT_PREFERRED_CANVAS);
            assert!(*has_vertex_buffer);
            assert!(*has_uniform_buffer);
            assert_eq!(*uniform_size, 144);
            assert_eq!(*vertex_stride, 20);
        }
        other => panic!("expected CreatePipeline for canvas, got {other:?}"),
    }

    // Command 4: WriteBuffer for uniform arena
    match &commands[4] {
        GpuCommand::WriteBuffer { buffer_id, offset, data } => {
            assert_eq!(*buffer_id, MESH_UNIFORM_BUFFER_ID);
            assert_eq!(*offset, 0);
            assert_eq!(data.len(), 256);
        }
        other => panic!("expected WriteBuffer for uniform arena, got {other:?}"),
    }

    // Command 5: RenderPass to visible canvas
    match &commands[5] {
        GpuCommand::RenderPass {
            target_type,
            target_id,
            clear_color,
            pipeline_id,
            vertex_buffer_id,
            vertex_count,
            uniform_dynamic_offset,
            uniform_buffer_id,
            load_op,
            store_op,
            pass_flags,
        } => {
            assert_eq!(*target_type, TARGET_CANVAS);
            assert_eq!(*target_id, MESH_CANVAS_TARGET_ID);
            assert_eq!(*clear_color, MESH_CLEAR_COLOR);
            assert_eq!(*pipeline_id, MESH_CANVAS_PIPELINE_ID);
            assert_eq!(*vertex_buffer_id, MESH_VERTEX_BUFFER_ID);
            assert_eq!(*vertex_count, 3);
            assert_eq!(*uniform_dynamic_offset, 0);
            assert_eq!(*uniform_buffer_id, MESH_UNIFORM_BUFFER_ID);
            assert_eq!(*load_op, 0); // LoadOp::Clear
            assert_eq!(*store_op, 0); // StoreOp::Store
            assert_eq!(*pass_flags, 1); // PASS_FLAG_NEW_PASS
        }
        other => panic!("expected RenderPass to canvas, got {other:?}"),
    }

    // Strict absence of offscreen texture creation or readback buffer
    for cmd in commands {
        assert!(
            !matches!(cmd, GpuCommand::CreateTexture { .. }),
            "canvas packet must not allocate offscreen textures"
        );
        assert!(
            !matches!(cmd, GpuCommand::CopyTextureToBuffer { .. }),
            "canvas packet must not copy texture to readback buffer"
        );
    }
}

#[test]
fn test_dynamic_mesh_canvas_wire_encoding_and_roundtrip() {
    let positions = [
        0.0_f32, 0.5, 0.0,
        -0.5, -0.5, 0.0,
        0.5, -0.5, 0.0,
    ];
    let indices: [u32; 0] = [];
    let color = [0.0_f32, 0.0, 1.0, 1.0];

    let bytes = f3d_build_canvas_mesh_packet(
        &positions,
        &indices,
        &IDENTITY_F64,
        &IDENTITY_F64,
        &color,
        64,
        64,
        false,
    )
    .expect("f3d_build_canvas_mesh_packet must succeed");

    assert!(!bytes.is_empty());
    assert!(bytes.len() >= 16, "wire packet must at least have header");

    let decoded = GpuSubmissionPacket::decode(&bytes)
        .expect("canvas wire packet must decode cleanly");
    assert_eq!(decoded.commands().len(), 6);

    let input = DynamicMeshInput::try_from_raw(
        &positions,
        &indices,
        &IDENTITY_F64,
        &IDENTITY_F64,
        &color,
        64,
        64,
        false,
    )
    .unwrap();
    let expected = build_mesh_canvas_submission(&input).unwrap();
    assert_eq!(decoded.commands(), expected.commands());
}

#[test]
fn test_dynamic_mesh_canvas_export_parity() {
    let positions = [
        0.0_f32, 0.5, 0.0,
        -0.5, -0.5, 0.0,
        0.5, -0.5, 0.0,
    ];
    let indices: [u32; 0] = [];
    let color = [0.5_f32, 0.5, 0.5, 1.0];

    let b_f3d_canvas = f3d_build_canvas_mesh_packet(
        &positions, &indices, &IDENTITY_F64, &IDENTITY_F64, &color, 64, 64, false,
    ).unwrap();

    let b_gpu_bridge_canvas = gpu_bridge_build_canvas_mesh_packet(
        &positions, &indices, &IDENTITY_F64, &IDENTITY_F64, &color, 64, 64, false,
    ).unwrap();

    assert_eq!(b_f3d_canvas, b_gpu_bridge_canvas);
}

#[test]
fn test_dynamic_mesh_canvas_zero_draw_clear() {
    let positions: [f32; 0] = [];
    let indices: [u32; 0] = [];
    let color = [1.0_f32, 1.0, 1.0, 1.0];

    let input = DynamicMeshInput::try_from_raw(
        &positions,
        &indices,
        &IDENTITY_F64,
        &IDENTITY_F64,
        &color,
        64,
        64,
        false,
    )
    .expect("empty geometry is legal input");

    let packet = build_mesh_canvas_submission(&input)
        .expect("zero-draw clear submission must succeed");

    let commands = packet.commands();
    assert_eq!(commands.len(), 6);

    // Minimum 4 bytes vertex buffer allocation
    if let GpuCommand::CreateBuffer { size, .. } = &commands[1] {
        assert_eq!(*size, 4);
    } else {
        panic!("expected CreateBuffer for vertex");
    }

    // Zero draw vertices on canvas pass
    if let GpuCommand::RenderPass { vertex_count, target_type, .. } = &commands[5] {
        assert_eq!(*vertex_count, 0);
        assert_eq!(*target_type, TARGET_CANVAS);
    } else {
        panic!("expected RenderPass");
    }

    let bytes = f3d_build_canvas_mesh_packet(
        &positions, &indices, &IDENTITY_F64, &IDENTITY_F64, &color, 64, 64, false,
    ).expect("wire encoding zero-draw canvas packet succeeds");
    assert!(!bytes.is_empty());
}

#[test]
fn test_dynamic_mesh_canvas_indexed_geometry_quad() {
    // 4 vertices, 6 indices (2 triangles forming a quad)
    let positions = [
        -1.0_f32, -1.0, 0.0,
         1.0,     -1.0, 0.0,
         1.0,      1.0, 0.0,
        -1.0,      1.0, 0.0,
    ];
    let indices = [0u32, 1, 2, 0, 2, 3];
    let color = [0.2_f32, 0.4, 0.6, 1.0];

    let input = DynamicMeshInput::try_from_raw(
        &positions,
        &indices,
        &IDENTITY_F64,
        &IDENTITY_F64,
        &color,
        128,
        128,
        false,
    )
    .expect("quad input is valid");

    let packet = build_mesh_canvas_submission(&input)
        .expect("canvas quad submission must succeed");
    let commands = packet.commands();
    assert_eq!(commands.len(), 6);

    // Vertex buffer size: 6 vertices * 20 bytes = 120 bytes
    if let GpuCommand::CreateBuffer { size, .. } = &commands[1] {
        assert_eq!(*size, 120);
    } else {
        panic!("expected CreateBuffer for vertex");
    }

    // Render pass vertex count: 6
    if let GpuCommand::RenderPass { vertex_count, target_type, pipeline_id, .. } = &commands[5] {
        assert_eq!(*vertex_count, 6);
        assert_eq!(*target_type, TARGET_CANVAS);
        assert_eq!(*pipeline_id, MESH_CANVAS_PIPELINE_ID);
    } else {
        panic!("expected RenderPass");
    }
}

#[test]
fn test_dynamic_mesh_canvas_webgl_depth_handling() {
    let positions = [
        0.0_f32, 0.5, 0.0,
        -0.5, -0.5, 0.0,
        0.5, -0.5, 0.0,
    ];
    let indices: [u32; 0] = [];
    let color = [1.0_f32, 0.0, 0.0, 1.0];

    // webgl_depth = true
    let input_webgl = DynamicMeshInput::try_from_raw(
        &positions, &indices, &IDENTITY_F64, &IDENTITY_F64, &color, 64, 64, true,
    ).unwrap();
    let packet_webgl = build_mesh_canvas_submission(&input_webgl).unwrap();
    if let GpuCommand::CreatePipeline { wgsl_code, .. } = &packet_webgl.commands()[3] {
        assert!(
            wgsl_code.contains("clip.z = (clip.z + clip.w) * 0.5;"),
            "canvas pipeline must remap depth when webgl_depth is true"
        );
    } else {
        panic!("expected CreatePipeline");
    }

    // webgl_depth = false
    let input_webgpu = DynamicMeshInput::try_from_raw(
        &positions, &indices, &IDENTITY_F64, &IDENTITY_F64, &color, 64, 64, false,
    ).unwrap();
    let packet_webgpu = build_mesh_canvas_submission(&input_webgpu).unwrap();
    if let GpuCommand::CreatePipeline { wgsl_code, .. } = &packet_webgpu.commands()[3] {
        assert!(
            !wgsl_code.contains("clip.z = (clip.z + clip.w) * 0.5;"),
            "canvas pipeline must not remap depth when webgl_depth is false"
        );
    } else {
        panic!("expected CreatePipeline");
    }
}

#[test]
fn test_dynamic_mesh_canvas_validation_errors() {
    let positions = [
        0.0_f32, 0.5, 0.0,
        -0.5, -0.5, 0.0,
        0.5, -0.5, 0.0,
    ];
    let indices: [u32; 0] = [];
    let color = [1.0_f32, 0.0, 0.0, 1.0];

    // Zero width
    let err_w = f3d_build_canvas_mesh_packet(
        &positions, &indices, &IDENTITY_F64, &IDENTITY_F64, &color, 0, 64, false,
    ).unwrap_err();
    assert!(err_w.contains("dimensions"));

    // Zero height
    let err_h = f3d_build_canvas_mesh_packet(
        &positions, &indices, &IDENTITY_F64, &IDENTITY_F64, &color, 64, 0, false,
    ).unwrap_err();
    assert!(err_h.contains("dimensions"));

    // Bad position length
    let bad_pos = [0.0_f32, 1.0];
    let err_p = f3d_build_canvas_mesh_packet(
        &bad_pos, &indices, &IDENTITY_F64, &IDENTITY_F64, &color, 64, 64, false,
    ).unwrap_err();
    assert!(err_p.contains("positions length"));

    // Out-of-bounds index
    let oob_idx = [0u32, 1, 99];
    let err_i = f3d_build_canvas_mesh_packet(
        &positions, &oob_idx, &IDENTITY_F64, &IDENTITY_F64, &color, 64, 64, false,
    ).unwrap_err();
    assert!(err_i.contains("out of bounds"));

    // Bad matrix length
    let bad_mv = [1.0_f64; 15];
    let err_m = f3d_build_canvas_mesh_packet(
        &positions, &indices, &bad_mv, &IDENTITY_F64, &color, 64, 64, false,
    ).unwrap_err();
    assert!(err_m.contains("model_view matrix"));

    // Bad color length
    let bad_c = [1.0_f32; 5];
    let err_c = f3d_build_canvas_mesh_packet(
        &positions, &indices, &IDENTITY_F64, &IDENTITY_F64, &bad_c, 64, 64, false,
    ).unwrap_err();
    assert!(err_c.contains("color must contain"));
}
