//! Focused unit and integration tests for dynamic mesh-to-Wasm packet building.
//!
//! Enforces:
//! 1. Dynamic input processing for unindexed and indexed geometries.
//! 2. f64 to f32 upload boundary quantization and column-major memory layout.
//! 3. WebGL-to-WebGPU depth conversion conditional shader emission.
//! 4. Input-dependent mutations in geometry, color, and matrices.
//! 5. Input validation and error discipline (bounds, lengths, dimensions).
//! 6. Exact parity between `f3d_build_mesh_packet` and `gpu_bridge_build_mesh_packet`.

use f3d_core::ownership::Epoch;
use f3d_graph::pass::DepthStencilAttachment;
use f3d_graph::resource::ResourceId;
use f3d_runtime::frame::{FrameSession, RenderContext};
use f3d_runtime::gpu_host::{
    GpuCommand, GpuSubmissionPacket, BUFFER_USAGE_COPY_DST, BUFFER_USAGE_MAP_READ,
    BUFFER_USAGE_UNIFORM, BUFFER_USAGE_VERTEX, CULL_MODE_BACK, CULL_MODE_FRONT,
    CULL_MODE_NONE, DEPTH_COMPARE_ALWAYS, DEPTH_COMPARE_GREATER, DEPTH_COMPARE_LESS,
    FRONT_FACE_CCW, FRONT_FACE_CW,
    OPCODE_CREATE_PIPELINE, OPCODE_CREATE_PIPELINE_CULL, OPCODE_CREATE_PIPELINE_DEPTH,
    OPCODE_CREATE_PIPELINE_DEPTH_CULL, OPCODE_CREATE_PIPELINE_DEPTH_CULL_COLOR,
    TARGET_CANVAS, TARGET_FORMAT_DEPTH24PLUS,
    TARGET_FORMAT_PREFERRED_CANVAS, TARGET_FORMAT_RGBA8UNORM, TEXTURE_USAGE_COPY_SRC,
    TEXTURE_USAGE_RENDER_ATTACHMENT,
};
use f3d_runtime::mesh::{
    build_mesh_canvas_depth_submission, build_mesh_canvas_submission,
    build_mesh_depth_submission, build_mesh_submission,
    build_multi_mesh_canvas_depth_submission, build_multi_mesh_canvas_submission,
    build_multi_mesh_depth_submission, build_multi_mesh_submission,
    f3d_build_canvas_mesh_depth_packet, f3d_build_canvas_mesh_packet,
    f3d_build_mesh_batch_cull_depth_color_packet,
    f3d_build_mesh_batch_cull_depth_packet, f3d_build_mesh_batch_cull_packet,
    f3d_build_mesh_batch_packet,
    f3d_build_mesh_batch_vertex_color_packet,
    build_mesh_batch_vertex_color_packet_impl,
    f3d_build_mesh_depth_packet, f3d_build_mesh_packet,
    generate_mesh_wgsl,
    gpu_bridge_build_canvas_mesh_depth_packet, gpu_bridge_build_canvas_mesh_packet,
    gpu_bridge_build_mesh_depth_packet,
    gpu_bridge_build_mesh_packet,
    DynamicMeshInput, MeshDepthOptions,
    MeshPacketError, MESH_CANVAS_PIPELINE_ID, MESH_CANVAS_TARGET_ID,
    MESH_CLEAR_COLOR, MESH_DEPTH_TEXTURE_ID, MESH_PIPELINE_ID, MESH_READBACK_BUFFER_ID,
    MESH_TARGET_TEXTURE_ID, MESH_UNIFORM_BUFFER_ID, MESH_VERTEX_BUFFER_ID,
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
    let expected_bytes = expected.encode().expect("canvas submission must encode");
    assert_eq!(bytes, expected_bytes);
    assert_eq!(expected.commands().len(), 6);
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

#[test]
fn test_dynamic_mesh_depth_offscreen_packet_structure() {
    let positions = [
        0.0_f32, 0.5, 0.0,
        -0.5, -0.5, 0.0,
        0.5, -0.5, 0.0,
    ];
    let indices: [u32; 0] = [];
    let color = [0.0_f32, 1.0, 0.0, 1.0];

    let bytes = f3d_build_mesh_depth_packet(
        &positions,
        &indices,
        &IDENTITY_F64,
        &IDENTITY_F64,
        &color,
        64,
        64,
        false,
        true,
        true,
        DEPTH_COMPARE_LESS,
    )
    .expect("f3d_build_mesh_depth_packet must succeed");

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

    let packet = build_mesh_depth_submission(&input, true, true, DEPTH_COMPARE_LESS)
        .expect("depth submission must build");
    let commands = packet.commands();
    assert_eq!(commands.len(), 10);

    // 0: Create uniform buffer (size 256)
    assert!(matches!(
        &commands[0],
        GpuCommand::CreateBuffer { buffer_id: 1, size: 256, .. }
    ));

    // 1: Create vertex buffer (size 60)
    assert!(matches!(
        &commands[1],
        GpuCommand::CreateBuffer { buffer_id: 2, size: 60, .. }
    ));

    // 2: Write vertex buffer
    assert!(matches!(
        &commands[2],
        GpuCommand::WriteBuffer { buffer_id: 2, offset: 0, .. }
    ));

    // 3: Create color target texture (id 10, format RGBA8UNORM)
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

    // 4: Create depth target texture (id 11, format TARGET_FORMAT_DEPTH24PLUS)
    assert!(matches!(
        &commands[4],
        GpuCommand::CreateTexture {
            texture_id: MESH_DEPTH_TEXTURE_ID,
            width: 64,
            height: 64,
            format: TARGET_FORMAT_DEPTH24PLUS,
            usage,
        } if *usage == TEXTURE_USAGE_RENDER_ATTACHMENT
    ));

    // 5: Create readback buffer
    assert!(matches!(
        &commands[5],
        GpuCommand::CreateBuffer { buffer_id: 20, size: 16384, .. }
    ));

    // 6: Create pipeline depth
    if let GpuCommand::CreatePipelineDepth {
        pipeline_id,
        target_format,
        depth_format,
        depth_write_enabled,
        depth_compare,
        uniform_size,
        vertex_stride,
        ..
    } = &commands[6]
    {
        assert_eq!(*pipeline_id, MESH_PIPELINE_ID);
        assert_eq!(*target_format, TARGET_FORMAT_RGBA8UNORM);
        assert_eq!(*depth_format, TARGET_FORMAT_DEPTH24PLUS);
        assert_eq!(*depth_write_enabled, true);
        assert_eq!(*depth_compare, DEPTH_COMPARE_LESS);
        assert_eq!(*uniform_size, 144);
        assert_eq!(*vertex_stride, 20);
    } else {
        panic!("expected CreatePipelineDepth at index 6");
    }

    // 7: Write uniform buffer
    assert!(matches!(
        &commands[7],
        GpuCommand::WriteBuffer { buffer_id: 1, offset: 0, .. }
    ));

    // 8: Render pass depth
    if let GpuCommand::RenderPassDepth {
        pipeline_id,
        vertex_count,
        depth_target_id,
        depth_clear_value,
        depth_read_only,
        ..
    } = &commands[8]
    {
        assert_eq!(*pipeline_id, MESH_PIPELINE_ID);
        assert_eq!(*vertex_count, 3);
        assert_eq!(*depth_target_id, MESH_DEPTH_TEXTURE_ID);
        assert_eq!(*depth_clear_value, 1.0);
        assert_eq!(*depth_read_only, false);
    } else {
        panic!("expected RenderPassDepth at index 8");
    }

    // 9: CopyTextureToBuffer
    assert!(matches!(
        &commands[9],
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
fn test_dynamic_mesh_depth_disabled_pinned_three_invariants() {
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
    .unwrap();

    // When depth_test is false: depth_write MUST NOT be blindly forced to false.
    // depth_write is accepted directly as the effective GPU pipeline write flag (r186 WebGPUPipelineUtils.js:224).
    // depth_compare is forced to DEPTH_COMPARE_ALWAYS (8).
    let packet_write_true = build_mesh_depth_submission(&input, false, true, DEPTH_COMPARE_LESS).unwrap();
    let commands_true = packet_write_true.commands();

    if let GpuCommand::CreatePipelineDepth {
        depth_write_enabled,
        depth_compare,
        ..
    } = &commands_true[6]
    {
        assert_eq!(
            *depth_write_enabled, true,
            "depth_write_enabled must preserve effective depth_write flag when depth_test is false"
        );
        assert_eq!(
            *depth_compare, DEPTH_COMPARE_ALWAYS,
            "depth_compare must be DEPTH_COMPARE_ALWAYS (8) when depth_test is false"
        );
    } else {
        panic!("expected CreatePipelineDepth at index 6");
    }

    let packet_write_false = build_mesh_depth_submission(&input, false, false, DEPTH_COMPARE_LESS).unwrap();
    let commands_false = packet_write_false.commands();

    if let GpuCommand::CreatePipelineDepth {
        depth_write_enabled,
        depth_compare,
        ..
    } = &commands_false[6]
    {
        assert_eq!(
            *depth_write_enabled, false,
            "depth_write_enabled must be false when depth_write is false"
        );
        assert_eq!(
            *depth_compare, DEPTH_COMPARE_ALWAYS,
            "depth_compare must be DEPTH_COMPARE_ALWAYS (8) when depth_test is false"
        );
    } else {
        panic!("expected CreatePipelineDepth at index 6");
    }
}

#[test]
fn test_dynamic_mesh_depth_compare_validation() {
    let positions = [
        0.0_f32, 0.5, 0.0,
        -0.5, -0.5, 0.0,
        0.5, -0.5, 0.0,
    ];
    let indices: [u32; 0] = [];
    let color = [1.0_f32, 0.0, 0.0, 1.0];

    let input = DynamicMeshInput::try_from_raw(
        &positions, &indices, &IDENTITY_F64, &IDENTITY_F64, &color, 64, 64, false,
    ).unwrap();

    // 0 is invalid depth compare
    let err_0 = build_mesh_depth_submission(&input, true, true, 0).unwrap_err();
    assert_eq!(err_0, MeshPacketError::InvalidDepthCompare { value: 0 });

    // 9 is invalid depth compare
    let err_9 = build_mesh_depth_submission(&input, true, true, 9).unwrap_err();
    assert_eq!(err_9, MeshPacketError::InvalidDepthCompare { value: 9 });

    // String export error checks
    let str_err_0 = f3d_build_mesh_depth_packet(
        &positions, &indices, &IDENTITY_F64, &IDENTITY_F64, &color, 64, 64, false, true, true, 0,
    ).unwrap_err();
    assert!(str_err_0.contains("depth compare function code must be between 1 and 8"));

    let str_err_9 = f3d_build_canvas_mesh_depth_packet(
        &positions, &indices, &IDENTITY_F64, &IDENTITY_F64, &color, 64, 64, false, true, true, 9,
    ).unwrap_err();
    assert!(str_err_9.contains("depth compare function code must be between 1 and 8"));
}

#[test]
fn test_dynamic_mesh_depth_canvas_packet_structure() {
    let positions = [
        0.0_f32, 0.5, 0.0,
        -0.5, -0.5, 0.0,
        0.5, -0.5, 0.0,
    ];
    let indices: [u32; 0] = [];
    let color = [0.0_f32, 0.0, 1.0, 1.0];

    let bytes = f3d_build_canvas_mesh_depth_packet(
        &positions,
        &indices,
        &IDENTITY_F64,
        &IDENTITY_F64,
        &color,
        128,
        128,
        false,
        true,
        false,
        DEPTH_COMPARE_LESS,
    )
    .expect("f3d_build_canvas_mesh_depth_packet must succeed");

    assert!(!bytes.is_empty());

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
    .unwrap();

    let packet = build_mesh_canvas_depth_submission(&input, true, false, DEPTH_COMPARE_LESS).unwrap();
    let commands = packet.commands();
    assert_eq!(commands.len(), 7);

    // 0: CreateBuffer uniform
    assert!(matches!(&commands[0], GpuCommand::CreateBuffer { buffer_id: 1, .. }));
    // 1: CreateBuffer vertex
    assert!(matches!(&commands[1], GpuCommand::CreateBuffer { buffer_id: 2, .. }));
    // 2: WriteBuffer vertex
    assert!(matches!(&commands[2], GpuCommand::WriteBuffer { buffer_id: 2, .. }));

    // 3: CreateTexture depth (id 11, format TARGET_FORMAT_DEPTH24PLUS, size 128x128)
    assert!(matches!(
        &commands[3],
        GpuCommand::CreateTexture {
            texture_id: MESH_DEPTH_TEXTURE_ID,
            width: 128,
            height: 128,
            format: TARGET_FORMAT_DEPTH24PLUS,
            usage,
        } if *usage == TEXTURE_USAGE_RENDER_ATTACHMENT
    ));

    // 4: CreatePipelineDepth (target_format preferred canvas, depth format depth24plus)
    if let GpuCommand::CreatePipelineDepth {
        pipeline_id,
        target_format,
        depth_format,
        depth_write_enabled,
        depth_compare,
        ..
    } = &commands[4]
    {
        assert_eq!(*pipeline_id, MESH_CANVAS_PIPELINE_ID);
        assert_eq!(*target_format, TARGET_FORMAT_PREFERRED_CANVAS);
        assert_eq!(*depth_format, TARGET_FORMAT_DEPTH24PLUS);
        assert_eq!(*depth_write_enabled, false);
        assert_eq!(*depth_compare, DEPTH_COMPARE_LESS);
    } else {
        panic!("expected CreatePipelineDepth at index 4");
    }

    // 5: WriteBuffer uniform
    assert!(matches!(&commands[5], GpuCommand::WriteBuffer { buffer_id: 1, .. }));

    // 6: RenderPassDepth targeting canvas
    if let GpuCommand::RenderPassDepth {
        target_type,
        target_id,
        pipeline_id,
        vertex_count,
        depth_target_id,
        depth_clear_value,
        ..
    } = &commands[6]
    {
        assert_eq!(*target_type, TARGET_CANVAS);
        assert_eq!(*target_id, MESH_CANVAS_TARGET_ID);
        assert_eq!(*pipeline_id, MESH_CANVAS_PIPELINE_ID);
        assert_eq!(*vertex_count, 3);
        assert_eq!(*depth_target_id, MESH_DEPTH_TEXTURE_ID);
        assert_eq!(*depth_clear_value, 1.0);
    } else {
        panic!("expected RenderPassDepth at index 6");
    }
}

#[test]
fn test_dynamic_mesh_depth_bridge_parity() {
    let positions = [
        0.0_f32, 0.5, 0.0,
        -0.5, -0.5, 0.0,
        0.5, -0.5, 0.0,
    ];
    let indices: [u32; 0] = [];
    let color = [0.25_f32, 0.5, 0.75, 1.0];

    // Offscreen parity
    let f3d_offscreen = f3d_build_mesh_depth_packet(
        &positions, &indices, &IDENTITY_F64, &IDENTITY_F64, &color, 64, 64, false, true, true, DEPTH_COMPARE_LESS,
    ).unwrap();
    let bridge_offscreen = gpu_bridge_build_mesh_depth_packet(
        &positions, &indices, &IDENTITY_F64, &IDENTITY_F64, &color, 64, 64, false, true, true, DEPTH_COMPARE_LESS,
    ).unwrap();
    assert_eq!(f3d_offscreen, bridge_offscreen, "offscreen depth packets must be identical");

    // Canvas parity
    let f3d_canvas = f3d_build_canvas_mesh_depth_packet(
        &positions, &indices, &IDENTITY_F64, &IDENTITY_F64, &color, 64, 64, false, true, false, DEPTH_COMPARE_ALWAYS,
    ).unwrap();
    let bridge_canvas = gpu_bridge_build_canvas_mesh_depth_packet(
        &positions, &indices, &IDENTITY_F64, &IDENTITY_F64, &color, 64, 64, false, true, false, DEPTH_COMPARE_ALWAYS,
    ).unwrap();
    assert_eq!(f3d_canvas, bridge_canvas, "canvas depth packets must be identical");
}

#[test]
fn test_frame_session_depth_begin_seam() {
    let root_ctx = RenderContext::new_offscreen(
        ResourceId::new(10),
        64,
        64,
        Epoch::ZERO,
    );
    let mut session = FrameSession::new(root_ctx, 256).unwrap();

    let dsa = DepthStencilAttachment::new_depth_clear(ResourceId::new(11), 1.0);
    let pass_id = session
        .begin_render_pass_with_depth("test_pass", [0.0; 4], Some(dsa))
        .expect("begin_render_pass_with_depth must succeed");
    assert_eq!(pass_id.get(), 1);

    // Record a draw and complete pass
    session.record_direct_draw(100, 2, 3, None).unwrap();
    session.end_render_pass().unwrap();

    let packet = session.build_submission_packet().unwrap();
    let commands = packet.commands();

    // Verify RenderPassDepth is emitted with depth target 11 and clear 1.0
    assert!(commands.iter().any(|cmd| matches!(
        cmd,
        GpuCommand::RenderPassDepth {
            depth_target_id: 11,
            depth_clear_value,
            ..
        } if (*depth_clear_value - 1.0).abs() < f32::EPSILON
    )));
}

#[test]
fn test_mesh_packet_pipeline_emitted_shader_matrix() {
    // 1. Direct unit verification of public generate_mesh_wgsl(webgl_depth):
    // Must always retain linear-sRGB offscreen output behavior matching default upstream RenderTarget.
    for &webgl_depth in &[false, true] {
        let code = generate_mesh_wgsl(webgl_depth);
        assert!(
            !code.contains("srgb_transfer_oetf"),
            "public generate_mesh_wgsl must NOT define or invoke srgb_transfer_oetf"
        );
        assert!(
            code.contains("return uniforms.color;"),
            "public generate_mesh_wgsl fs_main must retain linear-sRGB return uniforms.color"
        );
        if webgl_depth {
            assert!(
                code.contains("clip.z = (clip.z + clip.w) * 0.5;"),
                "webgl_depth=true must remap clip.z"
            );
        } else {
            assert!(
                !code.contains("clip.z = (clip.z + clip.w) * 0.5;"),
                "webgl_depth=false must not remap clip.z"
            );
        }
    }

    // 2. Compact matrix over:
    // target (canvas vs offscreen) x depth (with vs without) x projection (webgl vs native)
    let positions = [0.0_f32, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0];
    let indices: [u32; 0] = [];
    let color = [1.0_f32, 0.5, 0.25, 1.0];

    struct TestCase {
        is_canvas: bool,
        has_depth: bool,
        webgl_depth: bool,
    }

    let matrix = [
        // Offscreen variants (linear-sRGB working space)
        TestCase { is_canvas: false, has_depth: false, webgl_depth: false },
        TestCase { is_canvas: false, has_depth: false, webgl_depth: true },
        TestCase { is_canvas: false, has_depth: true, webgl_depth: false },
        TestCase { is_canvas: false, has_depth: true, webgl_depth: true },
        // Canvas variants (sRGB OETF encoded output)
        TestCase { is_canvas: true, has_depth: false, webgl_depth: false },
        TestCase { is_canvas: true, has_depth: false, webgl_depth: true },
        TestCase { is_canvas: true, has_depth: true, webgl_depth: false },
        TestCase { is_canvas: true, has_depth: true, webgl_depth: true },
    ];

    for tc in matrix {
        let input = DynamicMeshInput::try_from_raw(
            &positions,
            &indices,
            &IDENTITY_F64,
            &IDENTITY_F64,
            &color,
            64,
            64,
            tc.webgl_depth,
        )
        .expect("valid dynamic mesh input");

        let packet = match (tc.is_canvas, tc.has_depth) {
            (false, false) => build_mesh_submission(&input).expect("offscreen submission"),
            (false, true) => {
                build_mesh_depth_submission(&input, true, true, DEPTH_COMPARE_LESS)
                    .expect("offscreen depth submission")
            }
            (true, false) => build_mesh_canvas_submission(&input).expect("canvas submission"),
            (true, true) => {
                build_mesh_canvas_depth_submission(&input, true, true, DEPTH_COMPARE_LESS)
                    .expect("canvas depth submission")
            }
        };

        // Search commands by enum variant rather than hardcoded index
        let pipeline_wgsl = packet
            .commands()
            .iter()
            .find_map(|cmd| match cmd {
                GpuCommand::CreatePipeline { wgsl_code, .. }
                | GpuCommand::CreatePipelineDepth { wgsl_code, .. } => Some(wgsl_code.as_str()),
                _ => None,
            })
            .expect("packet must contain pipeline creation command");

        if tc.is_canvas {
            assert!(
                pipeline_wgsl.contains("fn srgb_transfer_oetf(color: vec3<f32>) -> vec3<f32>"),
                "canvas pipeline must define srgb_transfer_oetf (depth={}, webgl={})",
                tc.has_depth,
                tc.webgl_depth
            );
            assert!(
                pipeline_wgsl.contains("select(a, b, color <= vec3<f32>(0.0031308))"),
                "canvas pipeline must use threshold 0.0031308 select (depth={}, webgl={})",
                tc.has_depth,
                tc.webgl_depth
            );
            assert!(
                pipeline_wgsl.contains("pow(clamped, vec3<f32>(0.41666))"),
                "canvas pipeline must use exponent 0.41666 (depth={}, webgl={})",
                tc.has_depth,
                tc.webgl_depth
            );
            assert!(
                pipeline_wgsl.contains("let srgb_rgb = srgb_transfer_oetf(uniforms.color.rgb);"),
                "canvas pipeline must encode uniforms.color.rgb (depth={}, webgl={})",
                tc.has_depth,
                tc.webgl_depth
            );
            assert!(
                pipeline_wgsl.contains("return vec4<f32>(srgb_rgb, uniforms.color.a);"),
                "canvas pipeline must return encoded RGB with unmodified alpha (depth={}, webgl={})",
                tc.has_depth,
                tc.webgl_depth
            );
            assert!(
                !pipeline_wgsl.contains("return uniforms.color;"),
                "canvas pipeline must not return unencoded uniforms.color (depth={}, webgl={})",
                tc.has_depth,
                tc.webgl_depth
            );
        } else {
            assert!(
                !pipeline_wgsl.contains("srgb_transfer_oetf"),
                "offscreen pipeline must not define srgb_transfer_oetf (depth={}, webgl={})",
                tc.has_depth,
                tc.webgl_depth
            );
            assert!(
                pipeline_wgsl.contains("return uniforms.color;"),
                "offscreen pipeline must retain linear-sRGB return uniforms.color (depth={}, webgl={})",
                tc.has_depth,
                tc.webgl_depth
            );
        }

        if tc.webgl_depth {
            assert!(
                pipeline_wgsl.contains("clip.z = (clip.z + clip.w) * 0.5;"),
                "webgl_depth=true pipeline must remap clip.z (canvas={}, depth={})",
                tc.is_canvas,
                tc.has_depth
            );
        } else {
            assert!(
                !pipeline_wgsl.contains("clip.z = (clip.z + clip.w) * 0.5;"),
                "webgl_depth=false pipeline must not remap clip.z (canvas={}, depth={})",
                tc.is_canvas,
                tc.has_depth
            );
        }
    }
}

#[test]
fn test_multi_mesh_empty_inputs_rejected() {
    let empty_offscreen = build_multi_mesh_submission(&[]);
    assert_eq!(empty_offscreen.unwrap_err(), MeshPacketError::EmptyMeshList);

    let empty_depth = build_multi_mesh_depth_submission(&[], true, true, 2);
    assert_eq!(empty_depth.unwrap_err(), MeshPacketError::EmptyMeshList);

    let empty_canvas = build_multi_mesh_canvas_submission(&[]);
    assert_eq!(empty_canvas.unwrap_err(), MeshPacketError::EmptyMeshList);

    let empty_canvas_depth = build_multi_mesh_canvas_depth_submission(&[], true, true, 2);
    assert_eq!(empty_canvas_depth.unwrap_err(), MeshPacketError::EmptyMeshList);

    let batch_err = f3d_build_mesh_batch_packet(
        &[],
        &[],
        &[],
        &IDENTITY_F64,
        &[],
        64,
        64,
        false,
        false,
        false,
        1,
        false,
    );
    assert!(batch_err.is_err(), "empty vertex_counts must be rejected");
}

#[test]
fn test_multi_mesh_dimension_mismatch_rejected() {
    let tri = [0.0_f32, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0];
    let col = [1.0_f32, 0.0, 0.0, 1.0];
    let indices: [u32; 0] = [];

    let mesh_64 = DynamicMeshInput::try_from_raw(
        &tri, &indices, &IDENTITY_F64, &IDENTITY_F64, &col, 64, 64, false,
    ).unwrap();
    let mesh_128 = DynamicMeshInput::try_from_raw(
        &tri, &indices, &IDENTITY_F64, &IDENTITY_F64, &col, 128, 128, false,
    ).unwrap();

    let err = build_multi_mesh_submission(&[mesh_64.clone(), mesh_128]).unwrap_err();
    assert_eq!(
        err,
        MeshPacketError::MismatchedDimensions {
            expected_width: 64,
            expected_height: 64,
            actual_width: 128,
            actual_height: 128,
        }
    );

    let mesh_webgl = DynamicMeshInput::try_from_raw(
        &tri, &indices, &IDENTITY_F64, &IDENTITY_F64, &col, 64, 64, true,
    ).unwrap();
    let err_depth = build_multi_mesh_submission(&[mesh_64, mesh_webgl]).unwrap_err();
    assert!(matches!(err_depth, MeshPacketError::InvalidDimensions(_)));
}

#[test]
fn test_multi_mesh_batch_packet_structure_and_offsets() {
    // Mesh 0: triangle (3 vertices), red
    let pos0 = [0.0_f32, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0];
    let col0 = [1.0_f32, 0.0, 0.0, 1.0];

    // Mesh 1: quad (6 unindexed vertices), blue
    let pos1 = [
        -0.5_f32,  0.5, 0.0,
        -0.5,     -0.5, 0.0,
         0.5,     -0.5, 0.0,
        -0.5,      0.5, 0.0,
         0.5,     -0.5, 0.0,
         0.5,      0.5, 0.0,
    ];
    let col1 = [0.0_f32, 0.0, 1.0, 1.0];

    let mut positions = Vec::new();
    positions.extend_from_slice(&pos0);
    positions.extend_from_slice(&pos1);

    let vertex_counts = [3u32, 6u32];

    let mut model_views = Vec::new();
    model_views.extend_from_slice(&IDENTITY_F64);
    model_views.extend_from_slice(&IDENTITY_F64);

    let mut colors = Vec::new();
    colors.extend_from_slice(&col0);
    colors.extend_from_slice(&col1);

    // Encode via f3d_build_mesh_batch_packet (offscreen)
    let bytes = f3d_build_mesh_batch_packet(
        &positions,
        &vertex_counts,
        &model_views,
        &IDENTITY_F64,
        &colors,
        64,
        64,
        false,
        false,
        false,
        1,
        false,
    ).expect("f3d_build_mesh_batch_packet must succeed");
    assert!(!bytes.is_empty());

    let mesh0 = DynamicMeshInput::try_from_raw(
        &pos0, &[], &IDENTITY_F64, &IDENTITY_F64, &col0, 64, 64, false,
    ).unwrap();
    let mesh1 = DynamicMeshInput::try_from_raw(
        &pos1, &[], &IDENTITY_F64, &IDENTITY_F64, &col1, 64, 64, false,
    ).unwrap();
    let packet = build_multi_mesh_submission(&[mesh0, mesh1]).expect("submission must build");
    assert_eq!(packet.encode().expect("encode"), bytes);
    let commands = packet.commands();

    // 0: Uniform buffer size must be 2 * 256 = 512 bytes
    assert!(matches!(
        &commands[0],
        GpuCommand::CreateBuffer { buffer_id: 1, size: 512, usage }
            if *usage == (BUFFER_USAGE_UNIFORM | BUFFER_USAGE_COPY_DST)
    ));

    // 1: Vertex buffer size must be 9 vertices * 20 bytes = 180 bytes
    assert!(matches!(
        &commands[1],
        GpuCommand::CreateBuffer { buffer_id: 2, size: 180, usage }
            if *usage == (BUFFER_USAGE_VERTEX | BUFFER_USAGE_COPY_DST)
    ));

    // 2: Write vertex buffer
    assert!(matches!(
        &commands[2],
        GpuCommand::WriteBuffer { buffer_id: 2, offset: 0, data } if data.len() == 180
    ));

    // 3: Offscreen target texture
    assert!(matches!(&commands[3], GpuCommand::CreateTexture { texture_id: 10, .. }));

    // 4: Readback buffer
    assert!(matches!(&commands[4], GpuCommand::CreateBuffer { buffer_id: 20, .. }));

    // 5: Pipeline 100
    assert!(matches!(&commands[5], GpuCommand::CreatePipeline { pipeline_id: 100, .. }));

    // 6: Uniform WriteBuffer emitted by FrameSession (size 512)
    assert!(matches!(
        &commands[6],
        GpuCommand::WriteBuffer { buffer_id: 1, offset: 0, data } if data.len() == 512
    ));

    // 7: Draw 0 (Mesh 0, 3 vertices at dynamic uniform offset 0)
    assert!(matches!(
        &commands[7],
        GpuCommand::RenderPass {
            pipeline_id: 100,
            vertex_buffer_id: 2,
            vertex_count: 3,
            uniform_dynamic_offset: 0,
            ..
        }
    ));

    // 8: SetDrawParameters for Draw 1 (first_vertex = 3)
    assert!(matches!(
        &commands[8],
        GpuCommand::SetDrawParameters {
            instance_count: 1,
            first_vertex: 3,
            first_instance: 0,
        }
    ));

    // 9: Draw 1 (Mesh 1, 6 vertices at dynamic uniform offset 256)
    assert!(matches!(
        &commands[9],
        GpuCommand::RenderPass {
            pipeline_id: 100,
            vertex_buffer_id: 2,
            vertex_count: 6,
            uniform_dynamic_offset: 256,
            ..
        }
    ));

    // 10: CopyTextureToBuffer
    assert!(matches!(
        &commands[10],
        GpuCommand::CopyTextureToBuffer { texture_id: 10, buffer_id: 20, .. }
    ));
}

#[test]
fn test_multi_mesh_canvas_depth_batch_packet() {
    let tri = [0.0_f32, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0];
    let mut positions = Vec::new();
    positions.extend_from_slice(&tri);
    positions.extend_from_slice(&tri);

    let vertex_counts = [3u32, 3u32];

    let mut model_views = Vec::new();
    model_views.extend_from_slice(&IDENTITY_F64);
    model_views.extend_from_slice(&IDENTITY_F64);

    let colors = [1.0_f32, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];

    let bytes = f3d_build_mesh_batch_packet(
        &positions,
        &vertex_counts,
        &model_views,
        &IDENTITY_F64,
        &colors,
        64,
        64,
        false,
        true,  // depth_test
        true,  // depth_write
        2,     // Less
        true,  // canvas
    ).expect("canvas depth batch packet must succeed");

    let mesh0 = DynamicMeshInput::try_from_raw(
        &tri, &[], &IDENTITY_F64, &IDENTITY_F64, &colors[0..4], 64, 64, false,
    ).unwrap();
    let mesh1 = DynamicMeshInput::try_from_raw(
        &tri, &[], &IDENTITY_F64, &IDENTITY_F64, &colors[4..8], 64, 64, false,
    ).unwrap();
    let packet = build_multi_mesh_canvas_depth_submission(&[mesh0, mesh1], true, true, 2)
        .expect("canvas depth submission must succeed");
    assert_eq!(packet.encode().expect("encode"), bytes);
    let commands = packet.commands();

    // Verify depth texture created
    assert!(commands.iter().any(|c| matches!(
        c,
        GpuCommand::CreateTexture { texture_id: 11, format, .. }
            if *format == TARGET_FORMAT_DEPTH24PLUS
    )));

    // Verify CreatePipelineDepth with sRGB OETF
    assert!(commands.iter().any(|c| matches!(
        c,
        GpuCommand::CreatePipelineDepth { pipeline_id: 101, wgsl_code, depth_compare: 2, .. }
            if wgsl_code.contains("srgb_transfer_oetf")
    )));

    // Verify RenderPassDepth commands
    let depth_passes: Vec<_> = commands
        .iter()
        .filter(|c| matches!(c, GpuCommand::RenderPassDepth { .. }))
        .collect();
    assert_eq!(depth_passes.len(), 2, "must have 2 RenderPassDepth draw commands");

    if let GpuCommand::RenderPassDepth {
        target_type,
        vertex_count,
        uniform_dynamic_offset,
        depth_target_id,
        ..
    } = depth_passes[0] {
        assert_eq!(*target_type, TARGET_CANVAS);
        assert_eq!(*vertex_count, 3);
        assert_eq!(*uniform_dynamic_offset, 0);
        assert_eq!(*depth_target_id, 11);
    }

    if let GpuCommand::RenderPassDepth {
        target_type,
        vertex_count,
        uniform_dynamic_offset,
        depth_target_id,
        ..
    } = depth_passes[1] {
        assert_eq!(*target_type, TARGET_CANVAS);
        assert_eq!(*vertex_count, 3);
        assert_eq!(*uniform_dynamic_offset, 256);
        assert_eq!(*depth_target_id, 11);
    }
}

#[test]
fn test_multi_mesh_batch_slice_length_validations() {
    let pos = [0.0_f32; 9];
    let v_counts = [3u32];
    let mv = IDENTITY_F64;
    let proj = IDENTITY_F64;
    let col = [1.0_f32, 0.0, 0.0, 1.0];

    // Invalid positions length (expected 9, got 8)
    assert!(f3d_build_mesh_batch_packet(
        &pos[..8], &v_counts, &mv, &proj, &col, 64, 64, false, false, false, 1, false,
    ).is_err());

    // Invalid model_views length (expected 16, got 15)
    assert!(f3d_build_mesh_batch_packet(
        &pos, &v_counts, &mv[..15], &proj, &col, 64, 64, false, false, false, 1, false,
    ).is_err());

    // Invalid projection length (expected 16, got 12)
    assert!(f3d_build_mesh_batch_packet(
        &pos, &v_counts, &mv, &proj[..12], &col, 64, 64, false, false, false, 1, false,
    ).is_err());

    // Invalid colors length (expected 4, got 3)
    assert!(f3d_build_mesh_batch_packet(
        &pos, &v_counts, &mv, &proj, &col[..3], 64, 64, false, false, false, 1, false,
    ).is_err());
}

#[test]
fn test_multi_mesh_empty_first_preserves_vertex_alignment() {
    let tri_positions = [
        0.0_f32, 0.5, 0.0,
        -0.5, -0.5, 0.0,
        0.5, -0.5, 0.0,
    ];
    let empty_positions: [f32; 0] = [];
    let mut positions = Vec::new();
    positions.extend_from_slice(&empty_positions);
    positions.extend_from_slice(&tri_positions);

    let vertex_counts = [0u32, 3u32];

    let mut model_views = Vec::new();
    model_views.extend_from_slice(&IDENTITY_F64);
    model_views.extend_from_slice(&IDENTITY_F64);

    let colors = [
        1.0_f32, 0.0, 0.0, 1.0, // Mesh 0 (empty)
        0.0_f32, 1.0, 0.0, 1.0, // Mesh 1 (triangle)
    ];

    let bytes = f3d_build_mesh_batch_packet(
        &positions,
        &vertex_counts,
        &model_views,
        &IDENTITY_F64,
        &colors,
        64,
        64,
        false,
        false,
        false,
        1,
        false,
    ).expect("f3d_build_mesh_batch_packet must succeed");
    assert!(!bytes.is_empty());

    let mesh0 = DynamicMeshInput::try_from_raw(
        &empty_positions, &[], &IDENTITY_F64, &IDENTITY_F64, &colors[0..4], 64, 64, false,
    ).unwrap();
    let mesh1 = DynamicMeshInput::try_from_raw(
        &tri_positions, &[], &IDENTITY_F64, &IDENTITY_F64, &colors[4..8], 64, 64, false,
    ).unwrap();

    let packet = build_multi_mesh_submission(&[mesh0, mesh1]).expect("submission must build");
    assert_eq!(packet.encode().expect("encode"), bytes);
    let commands = packet.commands();

    // 1: Vertex buffer size must be exactly 3 vertices * 20 bytes = 60 bytes (not 60 + 4 = 64 bytes)
    assert!(matches!(
        &commands[1],
        GpuCommand::CreateBuffer { buffer_id: 2, size: 60, usage }
            if *usage == (BUFFER_USAGE_VERTEX | BUFFER_USAGE_COPY_DST)
    ));

    // 2: Write vertex buffer must have exactly 60 bytes matching Mesh 1's vertices with no 4-byte offset
    let vertex_write = commands.iter().find_map(|c| match c {
        GpuCommand::WriteBuffer { buffer_id: 2, offset: 0, data } => Some(data),
        _ => None,
    }).expect("must have vertex WriteBuffer");
    assert_eq!(vertex_write.len(), 60, "vertex buffer must contain exactly 60 bytes");

    // First vertex of Mesh 1 must start at offset 0 of vertex_write (x = 0.0, y = 0.5, z = 0.0)
    let x0 = f32::from_le_bytes(vertex_write[0..4].try_into().unwrap());
    let y0 = f32::from_le_bytes(vertex_write[4..8].try_into().unwrap());
    let z0 = f32::from_le_bytes(vertex_write[8..12].try_into().unwrap());
    assert_eq!(x0, 0.0);
    assert_eq!(y0, 0.5);
    assert_eq!(z0, 0.0);

    // Draw passes
    let render_passes: Vec<_> = commands
        .iter()
        .filter(|c| matches!(c, GpuCommand::RenderPass { .. }))
        .collect();
    assert_eq!(render_passes.len(), 2);

    // Mesh 0: vertex_count 0, uniform_dynamic_offset 0
    if let GpuCommand::RenderPass { vertex_count, uniform_dynamic_offset, .. } = render_passes[0] {
        assert_eq!(*vertex_count, 0);
        assert_eq!(*uniform_dynamic_offset, 0);
    }

    // Mesh 1: vertex_count 3, uniform_dynamic_offset 256
    if let GpuCommand::RenderPass { vertex_count, uniform_dynamic_offset, .. } = render_passes[1] {
        assert_eq!(*vertex_count, 3);
        assert_eq!(*uniform_dynamic_offset, 256);
    }

    // Both draws have first_vertex = 0, so no SetDrawParameters with first_vertex != 0 exists
    for cmd in commands.iter() {
        if let GpuCommand::SetDrawParameters { first_vertex, .. } = cmd {
            assert_eq!(*first_vertex, 0, "first_vertex must be 0 when mesh 0 was empty");
        }
    }
}

#[test]
fn test_multi_mesh_empty_middle_preserves_vertex_alignment() {
    let tri0 = [
        0.0_f32, 0.5, 0.0,
        -0.5, -0.5, 0.0,
        0.5, -0.5, 0.0,
    ];
    let empty: [f32; 0] = [];
    let tri2 = [
        1.0_f32, 1.5, 0.0,
        0.5, 0.5, 0.0,
        1.5, 0.5, 0.0,
    ];

    let mut positions = Vec::new();
    positions.extend_from_slice(&tri0);
    positions.extend_from_slice(&empty);
    positions.extend_from_slice(&tri2);

    let vertex_counts = [3u32, 0u32, 3u32];

    let mut model_views = Vec::new();
    model_views.extend_from_slice(&IDENTITY_F64);
    model_views.extend_from_slice(&IDENTITY_F64);
    model_views.extend_from_slice(&IDENTITY_F64);

    let colors = [
        1.0_f32, 0.0, 0.0, 1.0, // Mesh 0
        0.0_f32, 1.0, 0.0, 1.0, // Mesh 1 (empty)
        0.0_f32, 0.0, 1.0, 1.0, // Mesh 2
    ];

    let bytes = f3d_build_mesh_batch_packet(
        &positions,
        &vertex_counts,
        &model_views,
        &IDENTITY_F64,
        &colors,
        64,
        64,
        false,
        false,
        false,
        1,
        false,
    ).expect("f3d_build_mesh_batch_packet must succeed");
    assert!(!bytes.is_empty());

    let mesh0 = DynamicMeshInput::try_from_raw(
        &tri0, &[], &IDENTITY_F64, &IDENTITY_F64, &colors[0..4], 64, 64, false,
    ).unwrap();
    let mesh1 = DynamicMeshInput::try_from_raw(
        &empty, &[], &IDENTITY_F64, &IDENTITY_F64, &colors[4..8], 64, 64, false,
    ).unwrap();
    let mesh2 = DynamicMeshInput::try_from_raw(
        &tri2, &[], &IDENTITY_F64, &IDENTITY_F64, &colors[8..12], 64, 64, false,
    ).unwrap();

    let packet = build_multi_mesh_submission(&[mesh0, mesh1, mesh2]).expect("submission must build");
    assert_eq!(packet.encode().expect("encode"), bytes);
    let commands = packet.commands();

    // Vertex buffer size must be exactly 6 vertices * 20 bytes = 120 bytes (not 120 + 4 = 124 bytes)
    assert!(matches!(
        &commands[1],
        GpuCommand::CreateBuffer { buffer_id: 2, size: 120, usage }
            if *usage == (BUFFER_USAGE_VERTEX | BUFFER_USAGE_COPY_DST)
    ));

    let vertex_write = commands.iter().find_map(|c| match c {
        GpuCommand::WriteBuffer { buffer_id: 2, offset: 0, data } => Some(data),
        _ => None,
    }).expect("must have vertex WriteBuffer");
    assert_eq!(vertex_write.len(), 120, "vertex buffer must contain exactly 120 bytes");

    // Mesh 0 vertices at bytes 0..60
    let x0 = f32::from_le_bytes(vertex_write[0..4].try_into().unwrap());
    let y0 = f32::from_le_bytes(vertex_write[4..8].try_into().unwrap());
    assert_eq!(x0, 0.0);
    assert_eq!(y0, 0.5);

    // Mesh 2 vertices at bytes 60..120 (starts at byte 60, matching first_vertex = 3 * 20 bytes)
    let x2 = f32::from_le_bytes(vertex_write[60..64].try_into().unwrap());
    let y2 = f32::from_le_bytes(vertex_write[64..68].try_into().unwrap());
    assert_eq!(x2, 1.0);
    assert_eq!(y2, 1.5);

    // Draw parameters and render passes
    let render_passes: Vec<_> = commands
        .iter()
        .filter(|c| matches!(c, GpuCommand::RenderPass { .. }))
        .collect();
    assert_eq!(render_passes.len(), 3);

    // Mesh 0: vertex_count 3, offset 0
    if let GpuCommand::RenderPass { vertex_count, uniform_dynamic_offset, .. } = render_passes[0] {
        assert_eq!(*vertex_count, 3);
        assert_eq!(*uniform_dynamic_offset, 0);
    }

    // Mesh 1: vertex_count 0, offset 256
    if let GpuCommand::RenderPass { vertex_count, uniform_dynamic_offset, .. } = render_passes[1] {
        assert_eq!(*vertex_count, 0);
        assert_eq!(*uniform_dynamic_offset, 256);
    }

    // Mesh 2: vertex_count 3, offset 512
    if let GpuCommand::RenderPass { vertex_count, uniform_dynamic_offset, .. } = render_passes[2] {
        assert_eq!(*vertex_count, 3);
        assert_eq!(*uniform_dynamic_offset, 512);
    }

    // Verify SetDrawParameters for Mesh 1 and Mesh 2 has first_vertex = 3
    let draw_params: Vec<_> = commands
        .iter()
        .filter(|c| matches!(c, GpuCommand::SetDrawParameters { .. }))
        .collect();
    assert_eq!(draw_params.len(), 2, "Mesh 1 and Mesh 2 must have SetDrawParameters");

    if let GpuCommand::SetDrawParameters { first_vertex, .. } = draw_params[0] {
        assert_eq!(*first_vertex, 3, "Mesh 1 has first_vertex 3");
    }
    if let GpuCommand::SetDrawParameters { first_vertex, .. } = draw_params[1] {
        assert_eq!(*first_vertex, 3, "Mesh 2 has first_vertex 3");
    }
}

#[test]
fn test_opcode_create_pipeline_cull_binary_encoding() {
    let mut packet = GpuSubmissionPacket::new();
    let wgsl = "@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }";
    packet.push(GpuCommand::CreatePipelineCull {
        pipeline_id: 104,
        wgsl_code: wgsl.to_string(),
        target_format: TARGET_FORMAT_RGBA8UNORM,
        has_vertex_buffer: true,
        has_uniform_buffer: true,
        uniform_size: 144,
        vertex_stride: 20,
        cull_mode: CULL_MODE_BACK,
        front_face: FRONT_FACE_CW,
    });

    let encoded = packet.encode().expect("encoding CreatePipelineCull");
    assert_eq!(&encoded[0..4], b"F3DP");
    let cmd_count = u32::from_le_bytes(encoded[8..12].try_into().unwrap());
    assert_eq!(cmd_count, 1);

    // Opcode at offset 16 is u16 = 14
    let op = u16::from_le_bytes(encoded[16..18].try_into().unwrap());
    assert_eq!(op, OPCODE_CREATE_PIPELINE_CULL);

    // Payload length is exactly 40 bytes:
    // 18..22: pipeline_id
    let pipeline_id = u32::from_le_bytes(encoded[18..22].try_into().unwrap());
    assert_eq!(pipeline_id, 104);
    // 22..26: code_offset
    let code_offset = u32::from_le_bytes(encoded[22..26].try_into().unwrap());
    assert_eq!(code_offset, 0);
    // 26..30: code_len
    let code_len = u32::from_le_bytes(encoded[26..30].try_into().unwrap());
    assert_eq!(code_len, wgsl.len() as u32);
    // 30..34: target_format
    let target_format = u32::from_le_bytes(encoded[30..34].try_into().unwrap());
    assert_eq!(target_format, TARGET_FORMAT_RGBA8UNORM);
    // 34..38: has_vertex_buffer
    assert_eq!(u32::from_le_bytes(encoded[34..38].try_into().unwrap()), 1);
    // 38..42: has_uniform_buffer
    assert_eq!(u32::from_le_bytes(encoded[38..42].try_into().unwrap()), 1);
    // 42..46: uniform_size
    assert_eq!(u32::from_le_bytes(encoded[42..46].try_into().unwrap()), 144);
    // 46..50: vertex_stride
    assert_eq!(u32::from_le_bytes(encoded[46..50].try_into().unwrap()), 20);
    // 50..54: cull_mode
    assert_eq!(u32::from_le_bytes(encoded[50..54].try_into().unwrap()), CULL_MODE_BACK);
    // 54..58: front_face
    assert_eq!(u32::from_le_bytes(encoded[54..58].try_into().unwrap()), FRONT_FACE_CW);

    // Header size = 16 (file header) + 2 (opcode) + 40 (payload) = 58 bytes
    // Variable shader code follows at offset 58
    assert_eq!(&encoded[58..58 + wgsl.len()], wgsl.as_bytes());
}

#[test]
fn test_opcode_create_pipeline_depth_cull_binary_encoding() {
    let mut packet = GpuSubmissionPacket::new();
    let wgsl = "@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }";
    packet.push(GpuCommand::CreatePipelineDepthCull {
        pipeline_id: 105,
        wgsl_code: wgsl.to_string(),
        target_format: TARGET_FORMAT_PREFERRED_CANVAS,
        has_vertex_buffer: true,
        has_uniform_buffer: true,
        uniform_size: 144,
        vertex_stride: 20,
        depth_format: TARGET_FORMAT_DEPTH24PLUS,
        depth_write_enabled: true,
        depth_compare: DEPTH_COMPARE_LESS,
        cull_mode: CULL_MODE_FRONT,
        front_face: FRONT_FACE_CCW,
    });

    let encoded = packet.encode().expect("encoding CreatePipelineDepthCull");
    assert_eq!(&encoded[0..4], b"F3DP");
    let cmd_count = u32::from_le_bytes(encoded[8..12].try_into().unwrap());
    assert_eq!(cmd_count, 1);

    // Opcode at offset 16 is u16 = 15
    let op = u16::from_le_bytes(encoded[16..18].try_into().unwrap());
    assert_eq!(op, OPCODE_CREATE_PIPELINE_DEPTH_CULL);

    // Payload length is exactly 52 bytes:
    // 18..22: pipeline_id
    assert_eq!(u32::from_le_bytes(encoded[18..22].try_into().unwrap()), 105);
    // 22..26: code_offset
    assert_eq!(u32::from_le_bytes(encoded[22..26].try_into().unwrap()), 0);
    // 26..30: code_len
    assert_eq!(u32::from_le_bytes(encoded[26..30].try_into().unwrap()), wgsl.len() as u32);
    // 30..34: target_format
    assert_eq!(u32::from_le_bytes(encoded[30..34].try_into().unwrap()), TARGET_FORMAT_PREFERRED_CANVAS);
    // 34..38: has_vertex_buffer
    assert_eq!(u32::from_le_bytes(encoded[34..38].try_into().unwrap()), 1);
    // 38..42: has_uniform_buffer
    assert_eq!(u32::from_le_bytes(encoded[38..42].try_into().unwrap()), 1);
    // 42..46: uniform_size
    assert_eq!(u32::from_le_bytes(encoded[42..46].try_into().unwrap()), 144);
    // 46..50: vertex_stride
    assert_eq!(u32::from_le_bytes(encoded[46..50].try_into().unwrap()), 20);
    // 50..54: depth_format
    assert_eq!(u32::from_le_bytes(encoded[50..54].try_into().unwrap()), TARGET_FORMAT_DEPTH24PLUS);
    // 54..58: depth_write_enabled
    assert_eq!(u32::from_le_bytes(encoded[54..58].try_into().unwrap()), 1);
    // 58..62: depth_compare
    assert_eq!(u32::from_le_bytes(encoded[58..62].try_into().unwrap()), DEPTH_COMPARE_LESS);
    // 62..66: cull_mode
    assert_eq!(u32::from_le_bytes(encoded[62..66].try_into().unwrap()), CULL_MODE_FRONT);
    // 66..70: front_face
    assert_eq!(u32::from_le_bytes(encoded[66..70].try_into().unwrap()), FRONT_FACE_CCW);

    // Total header = 16 + 2 + 52 = 70 bytes; shader code follows
    assert_eq!(&encoded[70..70 + wgsl.len()], wgsl.as_bytes());
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParsedPipelineCull {
    pipeline_id: u32,
    cull_mode: u32,
    front_face: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParsedPipelineDepthCull {
    pipeline_id: u32,
    target_format: u32,
    depth_format: u32,
    cull_mode: u32,
    front_face: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DetailedPipelineDepthCull {
    pipeline_id: u32,
    target_format: u32,
    depth_format: u32,
    depth_write_enabled: bool,
    depth_compare: u32,
    cull_mode: u32,
    front_face: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParsedDrawPass {
    opcode: u16,
    pipeline_id: u32,
    pass_flags: u32,
    vertex_count: u32,
    dynamic_offset: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParsedPipelineDepthCullColor {
    pipeline_id: u32,
    target_format: u32,
    depth_format: u32,
    depth_write_enabled: bool,
    depth_compare: u32,
    cull_mode: u32,
    front_face: u32,
    write_mask: u32,
}

#[derive(Debug, Default)]
struct ParsedPacketSummary {
    pipelines: Vec<u32>,
    depth_pipelines: Vec<u32>,
    cull_pipelines: Vec<ParsedPipelineCull>,
    depth_cull_pipelines: Vec<ParsedPipelineDepthCull>,
    detailed_depth_cull_pipelines: Vec<DetailedPipelineDepthCull>,
    depth_cull_color_pipelines: Vec<ParsedPipelineDepthCullColor>,
    draw_pipeline_ids: Vec<u32>,
    draw_passes: Vec<ParsedDrawPass>,
    vertex_strides: Vec<u32>,
    write_buffers: Vec<(u32, u32, Vec<u8>)>,
}

fn scan_packet_commands(packet_bytes: &[u8]) -> ParsedPacketSummary {
    assert!(packet_bytes.len() >= 16, "packet too short for header");
    assert_eq!(&packet_bytes[0..4], b"F3DP", "magic mismatch");
    let total_packet_len = u32::from_le_bytes(packet_bytes[4..8].try_into().unwrap()) as usize;
    let cmd_count = u32::from_le_bytes(packet_bytes[8..12].try_into().unwrap());
    let total_data_len = u32::from_le_bytes(packet_bytes[12..16].try_into().unwrap()) as usize;
    let data_payload_start = total_packet_len - total_data_len;
    let mut cursor = 16usize;
    let mut summary = ParsedPacketSummary::default();

    for _ in 0..cmd_count {
        assert!(cursor + 2 <= packet_bytes.len(), "cursor overflow reading opcode");
        let op = u16::from_le_bytes(packet_bytes[cursor..cursor + 2].try_into().unwrap());
        cursor += 2;
        match op {
            1 => cursor += 12, // CREATE_BUFFER
            2 => { // WRITE_BUFFER
                let buffer_id = u32::from_le_bytes(packet_bytes[cursor..cursor + 4].try_into().unwrap());
                let offset = u32::from_le_bytes(packet_bytes[cursor + 4..cursor + 8].try_into().unwrap());
                let data_offset = u32::from_le_bytes(packet_bytes[cursor + 8..cursor + 12].try_into().unwrap()) as usize;
                let data_len = u32::from_le_bytes(packet_bytes[cursor + 12..cursor + 16].try_into().unwrap()) as usize;
                let abs_start = data_payload_start + data_offset;
                let data = packet_bytes[abs_start..abs_start + data_len].to_vec();
                summary.write_buffers.push((buffer_id, offset, data));
                cursor += 16;
            }
            OPCODE_CREATE_PIPELINE => { // 3
                let pid = u32::from_le_bytes(packet_bytes[cursor..cursor + 4].try_into().unwrap());
                let stride = u32::from_le_bytes(packet_bytes[cursor + 28..cursor + 32].try_into().unwrap());
                summary.pipelines.push(pid);
                summary.vertex_strides.push(stride);
                cursor += 32;
            }
            4 => { // RENDER_PASS
                let raw_target = u32::from_le_bytes(packet_bytes[cursor..cursor + 4].try_into().unwrap());
                let pass_flags = (raw_target >> 24) & 0xFF;
                let pid = u32::from_le_bytes(packet_bytes[cursor + 24..cursor + 28].try_into().unwrap());
                let v_count = u32::from_le_bytes(packet_bytes[cursor + 32..cursor + 36].try_into().unwrap());
                let dyn_offset = u32::from_le_bytes(packet_bytes[cursor + 36..cursor + 40].try_into().unwrap());
                summary.draw_pipeline_ids.push(pid);
                summary.draw_passes.push(ParsedDrawPass {
                    opcode: 4,
                    pipeline_id: pid,
                    pass_flags,
                    vertex_count: v_count,
                    dynamic_offset: dyn_offset,
                });
                cursor += 44;
            }
            5 => cursor += 24, // COPY_TEXTURE_TO_BUFFER
            6 => cursor += 20, // CREATE_TEXTURE
            7 => cursor += 28, // RECORD_BUNDLE
            8 => { // EXECUTE_BUNDLES
                let cnt = u32::from_le_bytes(packet_bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
                cursor += 4 + cnt * 4;
            }
            9 => cursor += 24,  // SET_VIEWPORT
            10 => cursor += 16, // SET_SCISSOR_RECT
            11 => cursor += 12, // SET_DRAW_PARAMETERS
            OPCODE_CREATE_PIPELINE_DEPTH => { // 12
                let pid = u32::from_le_bytes(packet_bytes[cursor..cursor + 4].try_into().unwrap());
                let stride = u32::from_le_bytes(packet_bytes[cursor + 28..cursor + 32].try_into().unwrap());
                summary.depth_pipelines.push(pid);
                summary.vertex_strides.push(stride);
                cursor += 44;
            }
            13 => { // RENDER_PASS_DEPTH
                let raw_target = u32::from_le_bytes(packet_bytes[cursor..cursor + 4].try_into().unwrap());
                let pass_flags = (raw_target >> 24) & 0xFF;
                let pid = u32::from_le_bytes(packet_bytes[cursor + 24..cursor + 28].try_into().unwrap());
                let v_count = u32::from_le_bytes(packet_bytes[cursor + 32..cursor + 36].try_into().unwrap());
                let dyn_offset = u32::from_le_bytes(packet_bytes[cursor + 36..cursor + 40].try_into().unwrap());
                summary.draw_pipeline_ids.push(pid);
                summary.draw_passes.push(ParsedDrawPass {
                    opcode: 13,
                    pipeline_id: pid,
                    pass_flags,
                    vertex_count: v_count,
                    dynamic_offset: dyn_offset,
                });
                cursor += 56;
            }
            OPCODE_CREATE_PIPELINE_CULL => { // 14
                let pid = u32::from_le_bytes(packet_bytes[cursor..cursor + 4].try_into().unwrap());
                let stride = u32::from_le_bytes(packet_bytes[cursor + 28..cursor + 32].try_into().unwrap());
                let cm = u32::from_le_bytes(packet_bytes[cursor + 32..cursor + 36].try_into().unwrap());
                let ff = u32::from_le_bytes(packet_bytes[cursor + 36..cursor + 40].try_into().unwrap());
                summary.cull_pipelines.push(ParsedPipelineCull {
                    pipeline_id: pid,
                    cull_mode: cm,
                    front_face: ff,
                });
                summary.vertex_strides.push(stride);
                cursor += 40;
            }
            OPCODE_CREATE_PIPELINE_DEPTH_CULL => { // 15
                let pid = u32::from_le_bytes(packet_bytes[cursor..cursor + 4].try_into().unwrap());
                let target_format = u32::from_le_bytes(packet_bytes[cursor + 12..cursor + 16].try_into().unwrap());
                let stride = u32::from_le_bytes(packet_bytes[cursor + 28..cursor + 32].try_into().unwrap());
                let depth_format = u32::from_le_bytes(packet_bytes[cursor + 32..cursor + 36].try_into().unwrap());
                let depth_write_enabled = u32::from_le_bytes(packet_bytes[cursor + 36..cursor + 40].try_into().unwrap()) == 1;
                let depth_compare = u32::from_le_bytes(packet_bytes[cursor + 40..cursor + 44].try_into().unwrap());
                let cm = u32::from_le_bytes(packet_bytes[cursor + 44..cursor + 48].try_into().unwrap());
                let ff = u32::from_le_bytes(packet_bytes[cursor + 48..cursor + 52].try_into().unwrap());
                summary.depth_cull_pipelines.push(ParsedPipelineDepthCull {
                    pipeline_id: pid,
                    target_format,
                    depth_format,
                    cull_mode: cm,
                    front_face: ff,
                });
                summary.detailed_depth_cull_pipelines.push(DetailedPipelineDepthCull {
                    pipeline_id: pid,
                    target_format,
                    depth_format,
                    depth_write_enabled,
                    depth_compare,
                    cull_mode: cm,
                    front_face: ff,
                });
                summary.vertex_strides.push(stride);
                cursor += 52;
            }
            OPCODE_CREATE_PIPELINE_DEPTH_CULL_COLOR => { // 16
                let pid = u32::from_le_bytes(packet_bytes[cursor..cursor + 4].try_into().unwrap());
                let target_format = u32::from_le_bytes(packet_bytes[cursor + 12..cursor + 16].try_into().unwrap());
                let stride = u32::from_le_bytes(packet_bytes[cursor + 28..cursor + 32].try_into().unwrap());
                let depth_format = u32::from_le_bytes(packet_bytes[cursor + 32..cursor + 36].try_into().unwrap());
                let depth_write_enabled = u32::from_le_bytes(packet_bytes[cursor + 36..cursor + 40].try_into().unwrap()) == 1;
                let depth_compare = u32::from_le_bytes(packet_bytes[cursor + 40..cursor + 44].try_into().unwrap());
                let cm = u32::from_le_bytes(packet_bytes[cursor + 44..cursor + 48].try_into().unwrap());
                let ff = u32::from_le_bytes(packet_bytes[cursor + 48..cursor + 52].try_into().unwrap());
                let write_mask = u32::from_le_bytes(packet_bytes[cursor + 52..cursor + 56].try_into().unwrap());
                summary.depth_cull_color_pipelines.push(ParsedPipelineDepthCullColor {
                    pipeline_id: pid,
                    target_format,
                    depth_format,
                    depth_write_enabled,
                    depth_compare,
                    cull_mode: cm,
                    front_face: ff,
                    write_mask,
                });
                summary.vertex_strides.push(stride);
                cursor += 56;
            }
            other => panic!("scan_packet_commands: unexpected opcode {other} at cursor {cursor}"),
        }
    }
    summary
}

#[test]
fn test_mesh_batch_cull_single_mesh_n1() {
    let positions = [0.0f32, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0];
    let vertex_counts = [3u32];
    let model_views = IDENTITY_F64;
    let projection = IDENTITY_F64;
    let colors = [1.0f32, 0.0, 0.0, 1.0];
    let cull_modes = [CULL_MODE_BACK as u8]; // 2
    let front_faces = [FRONT_FACE_CCW as u8]; // 0

    let packet_bytes = f3d_build_mesh_batch_cull_packet(
        &positions,
        &vertex_counts,
        &model_views,
        &projection,
        &colors,
        &cull_modes,
        &front_faces,
        64,
        64,
        false,
        false,
        false,
        DEPTH_COMPARE_ALWAYS,
        false,
    )
    .expect("single mesh cull packet should build");

    let summary = scan_packet_commands(&packet_bytes);

    // Expected pipeline ID: MESH_PIPELINE_ID (100) + (2 * 2 + 0) = 104
    assert_eq!(summary.cull_pipelines.len(), 1, "CreatePipelineCull (opcode 14) must be emitted");
    assert_eq!(summary.cull_pipelines[0].pipeline_id, 104);
    assert_eq!(summary.cull_pipelines[0].cull_mode, CULL_MODE_BACK);
    assert_eq!(summary.cull_pipelines[0].front_face, FRONT_FACE_CCW);

    assert_eq!(summary.draw_pipeline_ids, vec![104], "RenderPass referencing pipeline 104 must be emitted");
}

#[test]
fn test_mesh_batch_cull_multiple_distinct_and_deduplicated_pipelines() {
    // 4 meshes:
    // Mesh 0: None, CCW (0, 0) -> pipeline 100
    // Mesh 1: Back, CCW (2, 0) -> pipeline 104
    // Mesh 2: Back, CW  (2, 1) -> pipeline 105
    // Mesh 3: Back, CCW (2, 0) -> pipeline 104 (deduplicated!)
    let mut positions = Vec::new();
    for _ in 0..4 {
        positions.extend_from_slice(&[0.0f32, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0]);
    }
    let vertex_counts = [3u32, 3, 3, 3];
    let mut model_views = Vec::new();
    for _ in 0..4 {
        model_views.extend_from_slice(&IDENTITY_F64);
    }
    let projection = IDENTITY_F64;
    let colors = [
        1.0f32, 0.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 1.0,
        0.0, 0.0, 1.0, 1.0,
        1.0, 1.0, 0.0, 1.0,
    ];
    let cull_modes = [0u8, 2, 2, 2];
    let front_faces = [0u8, 0, 1, 0];

    let packet_bytes = f3d_build_mesh_batch_cull_packet(
        &positions,
        &vertex_counts,
        &model_views,
        &projection,
        &colors,
        &cull_modes,
        &front_faces,
        64,
        64,
        false,
        false,
        false,
        DEPTH_COMPARE_ALWAYS,
        false,
    )
    .expect("multi-mesh cull packet should build");

    let summary = scan_packet_commands(&packet_bytes);

    // Exactly 3 unique pipelines created: 100 (0,0), 104 (2,0), 105 (2,1)
    assert_eq!(summary.cull_pipelines.len(), 3);
    assert_eq!(summary.cull_pipelines[0], ParsedPipelineCull { pipeline_id: 100, cull_mode: 0, front_face: 0 });
    assert_eq!(summary.cull_pipelines[1], ParsedPipelineCull { pipeline_id: 104, cull_mode: 2, front_face: 0 });
    assert_eq!(summary.cull_pipelines[2], ParsedPipelineCull { pipeline_id: 105, cull_mode: 2, front_face: 1 });

    // Draws: 4 draws routed to 100, 104, 105, 104
    assert_eq!(summary.draw_pipeline_ids, vec![100, 104, 105, 104]);
}

#[test]
fn test_mesh_batch_cull_canvas_depth() {
    let mut positions = Vec::new();
    for _ in 0..2 {
        positions.extend_from_slice(&[0.0f32, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0]);
    }
    let vertex_counts = [3u32, 3];
    let mut model_views = Vec::new();
    for _ in 0..2 {
        model_views.extend_from_slice(&IDENTITY_F64);
    }
    let projection = IDENTITY_F64;
    let colors = [
        1.0f32, 0.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 1.0,
    ];
    let cull_modes = [2u8, 0]; // BackSide (2), DoubleSide (0)
    let front_faces = [0u8, 0]; // CCW (0)

    let packet_bytes = f3d_build_mesh_batch_cull_packet(
        &positions,
        &vertex_counts,
        &model_views,
        &projection,
        &colors,
        &cull_modes,
        &front_faces,
        64,
        64,
        false,
        true, // depth_test
        true, // depth_write
        DEPTH_COMPARE_LESS,
        true, // canvas
    )
    .expect("canvas depth cull packet should build");

    let summary = scan_packet_commands(&packet_bytes);

    // Base canvas pipeline ID = 101
    // Mesh 0: Back (2), CCW (0) -> 101 + 4 = 105
    // Mesh 1: None (0), CCW (0) -> 101 + 0 = 101
    assert_eq!(summary.depth_cull_pipelines.len(), 2);
    assert_eq!(summary.depth_cull_pipelines[0], ParsedPipelineDepthCull {
        pipeline_id: 105,
        target_format: TARGET_FORMAT_PREFERRED_CANVAS,
        depth_format: TARGET_FORMAT_DEPTH24PLUS,
        cull_mode: 2,
        front_face: 0,
    });
    assert_eq!(summary.depth_cull_pipelines[1], ParsedPipelineDepthCull {
        pipeline_id: 101,
        target_format: TARGET_FORMAT_PREFERRED_CANVAS,
        depth_format: TARGET_FORMAT_DEPTH24PLUS,
        cull_mode: 0,
        front_face: 0,
    });

    assert_eq!(summary.draw_pipeline_ids, vec![105, 101]);
}

#[test]
fn test_mesh_batch_cull_validation_discipline() {
    let positions = [0.0f32, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0];
    let vertex_counts = [3u32];
    let model_views = IDENTITY_F64;
    let projection = IDENTITY_F64;
    let colors = [1.0f32, 0.0, 0.0, 1.0];

    // 1. Empty mesh batch
    let err = f3d_build_mesh_batch_cull_packet(
        &[], &[], &[], &projection, &[], &[], &[], 64, 64, false, false, false, 8, false,
    ).unwrap_err();
    assert!(err.contains("at least one mesh"));

    // 2. Mismatched cull_modes length
    let err = f3d_build_mesh_batch_cull_packet(
        &positions, &vertex_counts, &model_views, &projection, &colors,
        &[0u8, 1u8], &[0u8], // cull_modes len 2 != 1
        64, 64, false, false, false, 8, false,
    ).unwrap_err();
    assert!(err.contains("cull_modes array length must match mesh count 1 (got 2)"));

    // 3. Mismatched front_faces length
    let err = f3d_build_mesh_batch_cull_packet(
        &positions, &vertex_counts, &model_views, &projection, &colors,
        &[0u8], &[], // front_faces len 0 != 1
        64, 64, false, false, false, 8, false,
    ).unwrap_err();
    assert!(err.contains("front_faces array length must match mesh count 1 (got 0)"));

    // 4. Invalid cull_mode code (3)
    let err = f3d_build_mesh_batch_cull_packet(
        &positions, &vertex_counts, &model_views, &projection, &colors,
        &[3u8], &[0u8],
        64, 64, false, false, false, 8, false,
    ).unwrap_err();
    assert!(err.contains("cull mode code must be between 0 and 2 (got 3)"));

    // 5. Invalid front_face code (2)
    let err = f3d_build_mesh_batch_cull_packet(
        &positions, &vertex_counts, &model_views, &projection, &colors,
        &[0u8], &[2u8],
        64, 64, false, false, false, 8, false,
    ).unwrap_err();
    assert!(err.contains("front face code must be 0 (CCW) or 1 (CW) (got 2)"));
}

#[test]
fn test_legacy_f3d_build_mesh_batch_packet_preserves_opcode_3_and_12() {
    let positions = [0.0f32, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0];
    let vertex_counts = [3u32];
    let model_views = IDENTITY_F64;
    let projection = IDENTITY_F64;
    let colors = [1.0f32, 0.0, 0.0, 1.0];

    // Offscreen non-depth batch
    let packet_bytes = f3d_build_mesh_batch_packet(
        &positions, &vertex_counts, &model_views, &projection, &colors,
        64, 64, false, false, false, 8, false,
    ).expect("legacy batch should build");

    let summary_non_depth = scan_packet_commands(&packet_bytes);
    assert!(summary_non_depth.pipelines.contains(&100), "Legacy batch must emit OPCODE_CREATE_PIPELINE (3)");

    // Offscreen depth batch
    let depth_packet_bytes = f3d_build_mesh_batch_packet(
        &positions, &vertex_counts, &model_views, &projection, &colors,
        64, 64, false, true, true, DEPTH_COMPARE_LESS, false,
    ).expect("legacy depth batch should build");

    let summary_depth = scan_packet_commands(&depth_packet_bytes);
    assert!(summary_depth.depth_pipelines.contains(&100), "Legacy depth batch must emit OPCODE_CREATE_PIPELINE_DEPTH (12)");
}

#[test]
fn test_mesh_batch_cull_mixed_side_pipeline_switch_sequence() {
    // 4 meshes in a single depth pass matching Checkpoint 9c:
    // Mesh 0: DoubleSide, CCW (0, 0) -> pipeline 100
    // Mesh 1: FrontSide,  CCW (2, 0) -> pipeline 104
    // Mesh 2: FrontSide,  CCW (2, 0) -> pipeline 104 (deduplicated)
    // Mesh 3: BackSide,   CW  (2, 1) -> pipeline 105
    let mut positions = Vec::new();
    for _ in 0..4 {
        positions.extend_from_slice(&[0.0f32, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0]);
    }
    let vertex_counts = [3u32, 3, 3, 3];
    let mut model_views = Vec::new();
    for _ in 0..4 {
        model_views.extend_from_slice(&IDENTITY_F64);
    }
    let projection = IDENTITY_F64;
    let colors = [
        0.0f32, 0.0, 1.0, 1.0, // Blue (DoubleSide)
        0.0, 1.0, 0.0, 1.0,    // Green (FrontSide)
        1.0, 0.0, 0.0, 1.0,    // Red (FrontSide)
        1.0, 1.0, 0.0, 1.0,    // Yellow (BackSide)
    ];
    let cull_modes = [0u8, 2, 2, 2];
    let front_faces = [0u8, 0, 0, 1];

    let packet_bytes = f3d_build_mesh_batch_cull_packet(
        &positions,
        &vertex_counts,
        &model_views,
        &projection,
        &colors,
        &cull_modes,
        &front_faces,
        64,
        64,
        false,
        true, // depth_test
        true, // depth_write
        DEPTH_COMPARE_LESS,
        false,
    )
    .expect("mixed-side cull packet should build");

    let summary = scan_packet_commands(&packet_bytes);

    // 1. Assert exactly 3 unique depth-cull pipelines created:
    // 100 (cull_mode=0, front_face=0), 104 (cull_mode=2, front_face=0), 105 (cull_mode=2, front_face=1)
    assert_eq!(summary.depth_cull_pipelines.len(), 3);
    assert_eq!(summary.depth_cull_pipelines[0].pipeline_id, 100);
    assert_eq!(summary.depth_cull_pipelines[0].cull_mode, 0);
    assert_eq!(summary.depth_cull_pipelines[0].front_face, 0);
    assert_eq!(summary.depth_cull_pipelines[1].pipeline_id, 104);
    assert_eq!(summary.depth_cull_pipelines[1].cull_mode, 2);
    assert_eq!(summary.depth_cull_pipelines[1].front_face, 0);
    assert_eq!(summary.depth_cull_pipelines[2].pipeline_id, 105);
    assert_eq!(summary.depth_cull_pipelines[2].cull_mode, 2);
    assert_eq!(summary.depth_cull_pipelines[2].front_face, 1);

    // 2. Assert per-draw pipeline IDs on wire
    assert_eq!(summary.draw_pipeline_ids, vec![100, 104, 104, 105]);

    // 3. Assert exact draw passes, pass_flags, and dynamic uniform offsets
    assert_eq!(summary.draw_passes.len(), 4);
    // Draw 0: opens pass (pass_flags = 1 = PASS_FLAG_NEW_PASS), uses pipeline 100, dynamic offset 0
    assert_eq!(summary.draw_passes[0], ParsedDrawPass {
        opcode: 13,
        pipeline_id: 100,
        pass_flags: 1,
        vertex_count: 3,
        dynamic_offset: 0,
    });
    // Draw 1: keeps pass open (pass_flags = 0), switches pipeline to 104, dynamic offset 256
    assert_eq!(summary.draw_passes[1], ParsedDrawPass {
        opcode: 13,
        pipeline_id: 104,
        pass_flags: 0,
        vertex_count: 3,
        dynamic_offset: 256,
    });
    // Draw 2: keeps pass open (pass_flags = 0), retains pipeline 104, dynamic offset 512
    assert_eq!(summary.draw_passes[2], ParsedDrawPass {
        opcode: 13,
        pipeline_id: 104,
        pass_flags: 0,
        vertex_count: 3,
        dynamic_offset: 512,
    });
    // Draw 3: keeps pass open (pass_flags = 0), switches pipeline to 105, dynamic offset 768
    assert_eq!(summary.draw_passes[3], ParsedDrawPass {
        opcode: 13,
        pipeline_id: 105,
        pass_flags: 0,
        vertex_count: 3,
        dynamic_offset: 768,
    });
}

#[test]
fn test_mesh_batch_cull_depth_mixed_write_single_pass() {
    let mut positions = Vec::new();
    for _ in 0..2 {
        positions.extend_from_slice(&[0.0f32, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0]);
    }
    let vertex_counts = [3u32, 3];
    let mut model_views = Vec::new();
    for _ in 0..2 {
        model_views.extend_from_slice(&IDENTITY_F64);
    }
    let projection = IDENTITY_F64;
    let colors = [
        1.0f32, 0.0, 0.0, 1.0, // Red (opaque writer)
        0.0, 1.0, 0.0, 0.5,    // Green (blended/no-write)
    ];
    let cull_modes = [0u8, 0];
    let front_faces = [0u8, 0];
    let depth_tests = [1u8, 1];
    let depth_writes = [1u8, 0]; // Mixed: mesh 0 writes depth, mesh 1 does not
    let depth_compares = [DEPTH_COMPARE_LESS, DEPTH_COMPARE_LESS];

    let packet_bytes = f3d_build_mesh_batch_cull_depth_packet(
        &positions,
        &vertex_counts,
        &model_views,
        &projection,
        &colors,
        &cull_modes,
        &front_faces,
        &depth_tests,
        &depth_writes,
        &depth_compares,
        64,
        64,
        false,
        false,
    )
    .expect("mixed-write batch packet should build");

    let summary = scan_packet_commands(&packet_bytes);

    // 2 unique pipelines:
    // Pipeline 100: DoubleSide, depth_write=true, compare=Less (tag 0) -> 100 + 0 + 0 = 100
    // Pipeline 112: DoubleSide, depth_write=false, compare=Less (tag 2) -> 100 + 0 + 12 = 112
    assert_eq!(summary.detailed_depth_cull_pipelines.len(), 2);
    assert_eq!(summary.detailed_depth_cull_pipelines[0], DetailedPipelineDepthCull {
        pipeline_id: 100,
        target_format: TARGET_FORMAT_RGBA8UNORM,
        depth_format: TARGET_FORMAT_DEPTH24PLUS,
        depth_write_enabled: true,
        depth_compare: DEPTH_COMPARE_LESS,
        cull_mode: 0,
        front_face: 0,
    });
    assert_eq!(summary.detailed_depth_cull_pipelines[1], DetailedPipelineDepthCull {
        pipeline_id: 112,
        target_format: TARGET_FORMAT_RGBA8UNORM,
        depth_format: TARGET_FORMAT_DEPTH24PLUS,
        depth_write_enabled: false,
        depth_compare: DEPTH_COMPARE_LESS,
        cull_mode: 0,
        front_face: 0,
    });

    // Draws: Draw 0 -> 100 (pass open flag 1), Draw 1 -> 112 (pass retained flag 0)
    assert_eq!(summary.draw_pipeline_ids, vec![100, 112]);
    assert_eq!(summary.draw_passes.len(), 2);
    assert_eq!(summary.draw_passes[0].pass_flags, 1);
    assert_eq!(summary.draw_passes[1].pass_flags, 0);
}

#[test]
fn test_mesh_batch_cull_depth_mixed_compare_single_pass() {
    let mut positions = Vec::new();
    for _ in 0..2 {
        positions.extend_from_slice(&[0.0f32, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0]);
    }
    let vertex_counts = [3u32, 3];
    let mut model_views = Vec::new();
    for _ in 0..2 {
        model_views.extend_from_slice(&IDENTITY_F64);
    }
    let projection = IDENTITY_F64;
    let colors = [
        1.0f32, 0.0, 0.0, 1.0,
        0.0, 0.0, 1.0, 1.0,
    ];
    let cull_modes = [0u8, 0];
    let front_faces = [0u8, 0];
    let depth_tests = [1u8, 1];
    let depth_writes = [1u8, 1];
    let depth_compares = [DEPTH_COMPARE_LESS, DEPTH_COMPARE_GREATER]; // Mixed compare functions

    let packet_bytes = f3d_build_mesh_batch_cull_depth_packet(
        &positions,
        &vertex_counts,
        &model_views,
        &projection,
        &colors,
        &cull_modes,
        &front_faces,
        &depth_tests,
        &depth_writes,
        &depth_compares,
        64,
        64,
        false,
        false,
    )
    .expect("mixed-compare batch packet should build");

    let summary = scan_packet_commands(&packet_bytes);

    // Pipeline 100: DoubleSide, write=true, compare=Less (tag 0) -> 100 + 0 + 0 = 100
    // Pipeline 172: DoubleSide, write=true, compare=Greater (tag 12) -> 100 + 0 + 12*6 = 172
    assert_eq!(summary.detailed_depth_cull_pipelines.len(), 2);
    assert_eq!(summary.detailed_depth_cull_pipelines[0].pipeline_id, 100);
    assert_eq!(summary.detailed_depth_cull_pipelines[0].depth_compare, DEPTH_COMPARE_LESS);
    assert_eq!(summary.detailed_depth_cull_pipelines[1].pipeline_id, 172);
    assert_eq!(summary.detailed_depth_cull_pipelines[1].depth_compare, DEPTH_COMPARE_GREATER);

    assert_eq!(summary.draw_pipeline_ids, vec![100, 172]);
}

#[test]
fn test_mesh_batch_cull_depth_disabled_interleaving() {
    let mut positions = Vec::new();
    for _ in 0..3 {
        positions.extend_from_slice(&[0.0f32, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0]);
    }
    let vertex_counts = [3u32, 3, 3];
    let mut model_views = Vec::new();
    for _ in 0..3 {
        model_views.extend_from_slice(&IDENTITY_F64);
    }
    let projection = IDENTITY_F64;
    let colors = [
        1.0f32, 0.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 1.0,
        0.0, 0.0, 1.0, 1.0,
    ];
    let cull_modes = [0u8, 0, 0];
    let front_faces = [0u8, 0, 0];
    // Mesh 0: enabled, Mesh 1: disabled, Mesh 2: enabled (dedup with Mesh 0)
    let depth_tests = [1u8, 0, 1];
    let depth_writes = [1u8, 0, 1];
    let depth_compares = [DEPTH_COMPARE_LESS, DEPTH_COMPARE_LESS, DEPTH_COMPARE_LESS];

    let packet_bytes = f3d_build_mesh_batch_cull_depth_packet(
        &positions,
        &vertex_counts,
        &model_views,
        &projection,
        &colors,
        &cull_modes,
        &front_faces,
        &depth_tests,
        &depth_writes,
        &depth_compares,
        64,
        64,
        false,
        false,
    )
    .expect("disabled-interleaved batch packet should build");

    let summary = scan_packet_commands(&packet_bytes);

    // Mesh 1 has depth_test=0 -> resolves to (write=false, compare=ALWAYS)
    // depth_tag for (false, ALWAYS=8) = 8.
    // Pipeline ID = 100 + 0 + 8*6 = 148.
    // Mesh 2 has same config as Mesh 0 -> pipeline 100 deduplicated.
    assert_eq!(summary.detailed_depth_cull_pipelines.len(), 2);
    assert_eq!(summary.detailed_depth_cull_pipelines[0].pipeline_id, 100);
    assert_eq!(summary.detailed_depth_cull_pipelines[0].depth_compare, DEPTH_COMPARE_LESS);
    assert_eq!(summary.detailed_depth_cull_pipelines[1].pipeline_id, 148);
    assert_eq!(summary.detailed_depth_cull_pipelines[1].depth_compare, DEPTH_COMPARE_ALWAYS);
    assert_eq!(summary.detailed_depth_cull_pipelines[1].depth_write_enabled, false);

    assert_eq!(summary.draw_pipeline_ids, vec![100, 148, 100]);
}

#[test]
fn test_mesh_batch_cull_depth_mixed_side_and_depth() {
    let mut positions = Vec::new();
    for _ in 0..4 {
        positions.extend_from_slice(&[0.0f32, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0]);
    }
    let vertex_counts = [3u32, 3, 3, 3];
    let mut model_views = Vec::new();
    for _ in 0..4 {
        model_views.extend_from_slice(&IDENTITY_F64);
    }
    let projection = IDENTITY_F64;
    let colors = [
        1.0f32, 0.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 1.0,
        0.0, 0.0, 1.0, 1.0,
        1.0, 1.0, 0.0, 1.0,
    ];
    // Mesh 0: DoubleSide (0,0), write=true, compare=Less -> pipeline 100
    // Mesh 1: FrontSide  (2,0), write=false, compare=Less (tag 2) -> 100 + 4 + 12 = 116
    // Mesh 2: FrontSide  (2,0), write=false, compare=Less -> 116 (deduplicated!)
    // Mesh 3: BackSide   (2,1), depth_test=0 -> write=false, compare=Always (tag 8) -> 100 + 5 + 48 = 153
    let cull_modes = [0u8, 2, 2, 2];
    let front_faces = [0u8, 0, 0, 1];
    let depth_tests = [1u8, 1, 1, 0];
    let depth_writes = [1u8, 0, 0, 0];
    let depth_compares = [DEPTH_COMPARE_LESS, DEPTH_COMPARE_LESS, DEPTH_COMPARE_LESS, DEPTH_COMPARE_LESS];

    let packet_bytes = f3d_build_mesh_batch_cull_depth_packet(
        &positions,
        &vertex_counts,
        &model_views,
        &projection,
        &colors,
        &cull_modes,
        &front_faces,
        &depth_tests,
        &depth_writes,
        &depth_compares,
        64,
        64,
        false,
        false,
    )
    .expect("mixed side and depth packet should build");

    let summary = scan_packet_commands(&packet_bytes);

    assert_eq!(summary.detailed_depth_cull_pipelines.len(), 3);
    assert_eq!(summary.detailed_depth_cull_pipelines[0].pipeline_id, 100);
    assert_eq!(summary.detailed_depth_cull_pipelines[1].pipeline_id, 116);
    assert_eq!(summary.detailed_depth_cull_pipelines[2].pipeline_id, 153);

    assert_eq!(summary.draw_pipeline_ids, vec![100, 116, 116, 153]);
}

#[test]
fn test_mesh_batch_cull_depth_canvas() {
    let mut positions = Vec::new();
    for _ in 0..2 {
        positions.extend_from_slice(&[0.0f32, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0]);
    }
    let vertex_counts = [3u32, 3];
    let mut model_views = Vec::new();
    for _ in 0..2 {
        model_views.extend_from_slice(&IDENTITY_F64);
    }
    let projection = IDENTITY_F64;
    let colors = [
        1.0f32, 0.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 1.0,
    ];
    let cull_modes = [0u8, 2];
    let front_faces = [0u8, 0];
    let depth_tests = [1u8, 1];
    let depth_writes = [1u8, 0];
    let depth_compares = [DEPTH_COMPARE_LESS, DEPTH_COMPARE_LESS];

    let packet_bytes = f3d_build_mesh_batch_cull_depth_packet(
        &positions,
        &vertex_counts,
        &model_views,
        &projection,
        &colors,
        &cull_modes,
        &front_faces,
        &depth_tests,
        &depth_writes,
        &depth_compares,
        64,
        64,
        false,
        true, // canvas = true
    )
    .expect("canvas mixed depth batch packet should build");

    let summary = scan_packet_commands(&packet_bytes);

    // Canvas base ID = 101
    // Mesh 0: 101 + 0 + 0 = 101
    // Mesh 1: 101 + 4 + 12 = 117
    assert_eq!(summary.detailed_depth_cull_pipelines.len(), 2);
    assert_eq!(summary.detailed_depth_cull_pipelines[0].pipeline_id, 101);
    assert_eq!(summary.detailed_depth_cull_pipelines[0].target_format, TARGET_FORMAT_PREFERRED_CANVAS);
    assert_eq!(summary.detailed_depth_cull_pipelines[1].pipeline_id, 117);
    assert_eq!(summary.detailed_depth_cull_pipelines[1].target_format, TARGET_FORMAT_PREFERRED_CANVAS);

    assert_eq!(summary.draw_pipeline_ids, vec![101, 117]);
}

#[test]
fn test_mesh_batch_cull_depth_all_disabled_emits_opcode_14() {
    let mut positions = Vec::new();
    for _ in 0..2 {
        positions.extend_from_slice(&[0.0f32, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0]);
    }
    let vertex_counts = [3u32, 3];
    let mut model_views = Vec::new();
    for _ in 0..2 {
        model_views.extend_from_slice(&IDENTITY_F64);
    }
    let projection = IDENTITY_F64;
    let colors = [
        1.0f32, 0.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 1.0,
    ];
    let cull_modes = [0u8, 2];
    let front_faces = [0u8, 0];
    let depth_tests = [0u8, 0];  // Both disabled
    let depth_writes = [0u8, 0]; // Both disabled
    let depth_compares = [DEPTH_COMPARE_ALWAYS, DEPTH_COMPARE_ALWAYS];

    let packet_bytes = f3d_build_mesh_batch_cull_depth_packet(
        &positions,
        &vertex_counts,
        &model_views,
        &projection,
        &colors,
        &cull_modes,
        &front_faces,
        &depth_tests,
        &depth_writes,
        &depth_compares,
        64,
        64,
        false,
        false,
    )
    .expect("all disabled depth batch should build");

    let summary = scan_packet_commands(&packet_bytes);

    // Since has_depth is false, OPCODE_CREATE_PIPELINE_CULL (14) must be emitted, NOT depth pipelines
    assert_eq!(summary.cull_pipelines.len(), 2);
    assert_eq!(summary.depth_cull_pipelines.len(), 0);
    assert_eq!(summary.depth_pipelines.len(), 0);
    assert_eq!(summary.draw_pipeline_ids, vec![100, 104]);
}

#[test]
fn test_mesh_batch_cull_depth_validation_errors() {
    let positions = [0.0f32, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0];
    let vertex_counts = [3u32];
    let model_views = IDENTITY_F64;
    let projection = IDENTITY_F64;
    let colors = [1.0f32, 0.0, 0.0, 1.0];
    let cull_modes = [0u8];
    let front_faces = [0u8];
    let depth_tests = [1u8];
    let depth_writes = [1u8];
    let depth_compares = [DEPTH_COMPARE_LESS];

    // 1. Mismatched depth_tests length
    let err = f3d_build_mesh_batch_cull_depth_packet(
        &positions, &vertex_counts, &model_views, &projection, &colors,
        &cull_modes, &front_faces,
        &[1u8, 0u8], &depth_writes, &depth_compares,
        64, 64, false, false,
    ).unwrap_err();
    assert!(err.contains("depth_tests array length must match mesh count 1 (got 2)"));

    // 2. Mismatched depth_writes length
    let err = f3d_build_mesh_batch_cull_depth_packet(
        &positions, &vertex_counts, &model_views, &projection, &colors,
        &cull_modes, &front_faces,
        &depth_tests, &[], &depth_compares,
        64, 64, false, false,
    ).unwrap_err();
    assert!(err.contains("depth_writes array length must match mesh count 1 (got 0)"));

    // 3. Mismatched depth_compares length
    let err = f3d_build_mesh_batch_cull_depth_packet(
        &positions, &vertex_counts, &model_views, &projection, &colors,
        &cull_modes, &front_faces,
        &depth_tests, &depth_writes, &[DEPTH_COMPARE_LESS, DEPTH_COMPARE_LESS],
        64, 64, false, false,
    ).unwrap_err();
    assert!(err.contains("depth_compares array length must match mesh count 1 (got 2)"));

    // 4. Invalid depth_test code (2)
    let err = f3d_build_mesh_batch_cull_depth_packet(
        &positions, &vertex_counts, &model_views, &projection, &colors,
        &cull_modes, &front_faces,
        &[2u8], &depth_writes, &depth_compares,
        64, 64, false, false,
    ).unwrap_err();
    assert!(err.contains("depth test code must be 0 (false) or 1 (true) (got 2)"));

    // 5. Invalid depth_write code (3)
    let err = f3d_build_mesh_batch_cull_depth_packet(
        &positions, &vertex_counts, &model_views, &projection, &colors,
        &cull_modes, &front_faces,
        &depth_tests, &[3u8], &depth_compares,
        64, 64, false, false,
    ).unwrap_err();
    assert!(err.contains("depth write code must be 0 (false) or 1 (true) (got 3)"));

    // 6. Invalid depth_compare code (0)
    let err = f3d_build_mesh_batch_cull_depth_packet(
        &positions, &vertex_counts, &model_views, &projection, &colors,
        &cull_modes, &front_faces,
        &depth_tests, &depth_writes, &[0u32],
        64, 64, false, false,
    ).unwrap_err();
    assert!(err.contains("depth compare function code must be between 1 and 8 (got 0)"));

    // 7. Invalid depth_compare code (9)
    let err = f3d_build_mesh_batch_cull_depth_packet(
        &positions, &vertex_counts, &model_views, &projection, &colors,
        &cull_modes, &front_faces,
        &depth_tests, &depth_writes, &[9u32],
        64, 64, false, false,
    ).unwrap_err();
    assert!(err.contains("depth compare function code must be between 1 and 8 (got 9)"));
}

#[test]
fn test_dynamic_mesh_input_with_depth_validates_raw_struct() {
    let positions = [0.0f32, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0];
    let indices: [u32; 0] = [];
    let mesh = DynamicMeshInput::try_from_raw(
        &positions,
        &indices,
        &IDENTITY_F64,
        &IDENTITY_F64,
        &[1.0, 0.0, 0.0, 1.0],
        64,
        64,
        false,
    )
    .expect("valid mesh input");

    // Invalid depth compare code (u32::MAX) bypass via raw struct must be rejected
    let raw_invalid = MeshDepthOptions {
        depth_test: true,
        depth_write: false,
        depth_compare: u32::MAX,
    };
    let err = mesh.clone().with_depth(raw_invalid).unwrap_err();
    assert!(matches!(err, MeshPacketError::InvalidDepthCompare { value } if value == u32::MAX));

    // Valid raw struct succeeds
    let raw_valid = MeshDepthOptions {
        depth_test: true,
        depth_write: true,
        depth_compare: DEPTH_COMPARE_LESS,
    };
    let configured = mesh.with_depth(raw_valid).expect("valid depth options accepted");
    assert_eq!(configured.depth(), Some(raw_valid));
}

#[test]
fn test_opcode_16_create_pipeline_depth_cull_color_binary_layout() {
    use f3d_runtime::gpu_host::GpuCommand;

    let mut packet = f3d_runtime::gpu_host::GpuSubmissionPacket::new();
    let wgsl = "@vertex fn vs() -> @builtin(position) vec4f { return vec4f(0); }\n@fragment fn fs() -> @location(0) vec4f { return vec4f(1); }";
    packet.push(GpuCommand::CreatePipelineDepthCullColor {
        pipeline_id: 196,
        wgsl_code: wgsl.to_string(),
        target_format: TARGET_FORMAT_RGBA8UNORM,
        has_vertex_buffer: true,
        has_uniform_buffer: true,
        uniform_size: 144,
        vertex_stride: 20,
        depth_format: TARGET_FORMAT_DEPTH24PLUS,
        depth_write_enabled: true,
        depth_compare: DEPTH_COMPARE_LESS,
        cull_mode: CULL_MODE_NONE,
        front_face: FRONT_FACE_CCW,
        write_mask: 0x0,
    });

    let encoded = packet.encode().expect("encoding CreatePipelineDepthCullColor");
    assert_eq!(&encoded[0..4], b"F3DP");
    let cmd_count = u32::from_le_bytes(encoded[8..12].try_into().unwrap());
    assert_eq!(cmd_count, 1);

    // Opcode at offset 16 is u16 = 16
    let op = u16::from_le_bytes(encoded[16..18].try_into().unwrap());
    assert_eq!(op, OPCODE_CREATE_PIPELINE_DEPTH_CULL_COLOR);

    // Payload length is exactly 56 bytes:
    // 18..22: pipeline_id (196)
    assert_eq!(u32::from_le_bytes(encoded[18..22].try_into().unwrap()), 196);
    // 22..26: code_offset
    assert_eq!(u32::from_le_bytes(encoded[22..26].try_into().unwrap()), 0);
    // 26..30: code_len
    assert_eq!(u32::from_le_bytes(encoded[26..30].try_into().unwrap()), wgsl.len() as u32);
    // 30..34: target_format
    assert_eq!(u32::from_le_bytes(encoded[30..34].try_into().unwrap()), TARGET_FORMAT_RGBA8UNORM);
    // 34..38: has_vertex_buffer
    assert_eq!(u32::from_le_bytes(encoded[34..38].try_into().unwrap()), 1);
    // 38..42: has_uniform_buffer
    assert_eq!(u32::from_le_bytes(encoded[38..42].try_into().unwrap()), 1);
    // 42..46: uniform_size
    assert_eq!(u32::from_le_bytes(encoded[42..46].try_into().unwrap()), 144);
    // 46..50: vertex_stride
    assert_eq!(u32::from_le_bytes(encoded[46..50].try_into().unwrap()), 20);
    // 50..54: depth_format
    assert_eq!(u32::from_le_bytes(encoded[50..54].try_into().unwrap()), TARGET_FORMAT_DEPTH24PLUS);
    // 54..58: depth_write_enabled
    assert_eq!(u32::from_le_bytes(encoded[54..58].try_into().unwrap()), 1);
    // 58..62: depth_compare
    assert_eq!(u32::from_le_bytes(encoded[58..62].try_into().unwrap()), DEPTH_COMPARE_LESS);
    // 62..66: cull_mode
    assert_eq!(u32::from_le_bytes(encoded[62..66].try_into().unwrap()), CULL_MODE_NONE);
    // 66..70: front_face
    assert_eq!(u32::from_le_bytes(encoded[66..70].try_into().unwrap()), FRONT_FACE_CCW);
    // 70..74: write_mask (0x0)
    assert_eq!(u32::from_le_bytes(encoded[70..74].try_into().unwrap()), 0x0);

    // Verify scan_packet_commands parses opcode 16 properly
    let summary = scan_packet_commands(&encoded);
    assert_eq!(summary.depth_cull_color_pipelines.len(), 1);
    let p = &summary.depth_cull_color_pipelines[0];
    assert_eq!(p.pipeline_id, 196);
    assert_eq!(p.write_mask, 0x0);
    assert_eq!(p.depth_write_enabled, true);
    assert_eq!(p.depth_compare, DEPTH_COMPARE_LESS);
}

#[test]
fn test_mesh_batch_cull_depth_color_mixed_batch() {
    let tri = [
        0.0f32,  0.5, -2.0,
       -0.5,   -0.5, -2.0,
        0.5,   -0.5, -2.0,
    ];
    let positions = [tri, tri].concat();
    let vertex_counts = [3u32, 3];
    let model_views = [IDENTITY_F64, IDENTITY_F64].concat();
    let projection = IDENTITY_F64;
    let colors = [
        1.0f32, 0.0, 0.0, 1.0, // Mesh 0: Red
        0.0, 1.0, 0.0, 1.0,    // Mesh 1: Green
    ];
    let cull_modes = [CULL_MODE_NONE as u8, CULL_MODE_NONE as u8];
    let front_faces = [FRONT_FACE_CCW as u8, FRONT_FACE_CCW as u8];
    let depth_tests = [1u8, 1];
    let depth_writes = [1u8, 1];
    let depth_compares = [DEPTH_COMPARE_LESS, DEPTH_COMPARE_LESS];
    // Mesh 0: colorWrite = true (1), Mesh 1: colorWrite = false (0) -> invisible occluder
    let color_writes = [1u8, 0];

    let packet_bytes = f3d_build_mesh_batch_cull_depth_color_packet(
        &positions,
        &vertex_counts,
        &model_views,
        &projection,
        &colors,
        &cull_modes,
        &front_faces,
        &depth_tests,
        &depth_writes,
        &depth_compares,
        &color_writes,
        64,
        64,
        false,
        false,
    ).expect("building mixed color_write batch");

    let summary = scan_packet_commands(&packet_bytes);
    // Mesh 0: colorWrite=true -> default pipeline ID 100 (opcode 15)
    // Mesh 1: colorWrite=false -> occluder pipeline ID 100 + 96 = 196 (opcode 16, write_mask=0)
    assert_eq!(summary.depth_cull_pipelines.len(), 1, "Expected 1 opcode 15 pipeline for colorWrite=true");
    assert_eq!(summary.depth_cull_pipelines[0].pipeline_id, 100);

    assert_eq!(summary.depth_cull_color_pipelines.len(), 1, "Expected 1 opcode 16 pipeline for colorWrite=false");
    assert_eq!(summary.depth_cull_color_pipelines[0].pipeline_id, 196);
    assert_eq!(summary.depth_cull_color_pipelines[0].write_mask, 0);

    // Two draw passes binding distinct pipelines
    assert_eq!(summary.draw_pipeline_ids, vec![100, 196]);
    assert_eq!(summary.draw_passes.len(), 2);
    assert_eq!(summary.draw_passes[0].pipeline_id, 100);
    assert_eq!(summary.draw_passes[1].pipeline_id, 196);
}

#[test]
fn test_mesh_batch_cull_depth_color_canvas_batch() {
    let tri = [
        0.0f32,  0.5, -2.0,
       -0.5,   -0.5, -2.0,
        0.5,   -0.5, -2.0,
    ];
    let positions = [tri, tri].concat();
    let vertex_counts = [3u32, 3];
    let model_views = [IDENTITY_F64, IDENTITY_F64].concat();
    let projection = IDENTITY_F64;
    let colors = [
        1.0f32, 0.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 1.0,
    ];
    let cull_modes = [CULL_MODE_NONE as u8, CULL_MODE_NONE as u8];
    let front_faces = [FRONT_FACE_CCW as u8, FRONT_FACE_CCW as u8];
    let depth_tests = [1u8, 1];
    let depth_writes = [1u8, 1];
    let depth_compares = [DEPTH_COMPARE_LESS, DEPTH_COMPARE_LESS];
    let color_writes = [1u8, 0];

    let packet_bytes = f3d_build_mesh_batch_cull_depth_color_packet(
        &positions,
        &vertex_counts,
        &model_views,
        &projection,
        &colors,
        &cull_modes,
        &front_faces,
        &depth_tests,
        &depth_writes,
        &depth_compares,
        &color_writes,
        64,
        64,
        false,
        true, // Canvas target
    ).expect("building canvas color_write batch");

    let summary = scan_packet_commands(&packet_bytes);
    // Canvas target: base 101, occluder 101 + 96 = 197
    assert_eq!(summary.depth_cull_pipelines[0].pipeline_id, 101);
    assert_eq!(summary.depth_cull_color_pipelines[0].pipeline_id, 197);
    assert_eq!(summary.draw_pipeline_ids, vec![101, 197]);
}

#[test]
fn test_mesh_batch_cull_depth_color_validation_errors() {
    let tri = [0.0f32, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0];
    let positions = tri;
    let vertex_counts = [3u32];
    let model_views = IDENTITY_F64;
    let projection = IDENTITY_F64;
    let colors = [1.0f32, 0.0, 0.0, 1.0];
    let cull_modes = [0u8];
    let front_faces = [0u8];
    let depth_tests = [1u8];
    let depth_writes = [1u8];
    let depth_compares = [DEPTH_COMPARE_LESS];

    // Array length mismatch
    let err = f3d_build_mesh_batch_cull_depth_color_packet(
        &positions, &vertex_counts, &model_views, &projection, &colors,
        &cull_modes, &front_faces,
        &depth_tests, &depth_writes, &depth_compares,
        &[], // Empty color_writes for 1 mesh
        64, 64, false, false,
    ).unwrap_err();
    assert!(err.contains("color_writes array length must match mesh count 1 (got 0)"));

    // Invalid color_write value > 1
    let err2 = f3d_build_mesh_batch_cull_depth_color_packet(
        &positions, &vertex_counts, &model_views, &projection, &colors,
        &cull_modes, &front_faces,
        &depth_tests, &depth_writes, &depth_compares,
        &[2u8], // invalid
        64, 64, false, false,
    ).unwrap_err();
    assert!(err2.contains("color write code must be 0 (false) or 1 (true) (got 2)"));
}

#[test]
fn test_mesh_batch_color_write_disabled_without_depth_has_compatible_attachments() {
    let tri = [0.0f32, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0];
    let color = [1.0f32, 0.0, 0.0, 1.0];
    let visible = DynamicMeshInput::try_from_raw(
        &tri, &[], &IDENTITY_F64, &IDENTITY_F64, &color, 64, 64, false,
    ).unwrap()
        .with_cull(CULL_MODE_NONE, FRONT_FACE_CCW).unwrap()
        .with_depth_options(false, false, DEPTH_COMPARE_LESS).unwrap();
    let inputs = [visible.clone(), visible.with_color_write(false)];

    for canvas in [false, true] {
        let packet = if canvas {
            build_multi_mesh_canvas_submission(&inputs)
        } else {
            build_multi_mesh_submission(&inputs)
        }.expect("mixed color masks with disabled depth must produce a valid packet");
        let depth_textures = packet.commands().iter().filter(|command| matches!(command,
            GpuCommand::CreateTexture {
                texture_id: MESH_DEPTH_TEXTURE_ID, format: TARGET_FORMAT_DEPTH24PLUS, ..
            }
        )).count();
        assert_eq!(depth_textures, 1, "both pipeline masks require the same depth format");
        let attached_depth_targets: Vec<u32> = packet.commands().iter().filter_map(|command| {
            if let GpuCommand::RenderPassDepth { depth_target_id, .. } = command {
                Some(*depth_target_id)
            } else {
                None
            }
        }).collect();
        assert_eq!(attached_depth_targets, vec![MESH_DEPTH_TEXTURE_ID; 2]);

        let bytes = f3d_build_mesh_batch_cull_depth_color_packet(
            &[tri, tri].concat(), &[3, 3], &[IDENTITY_F64, IDENTITY_F64].concat(),
            &IDENTITY_F64, &[color, color].concat(), &[0, 0], &[0, 0],
            &[0, 0], &[0, 0], &[DEPTH_COMPARE_LESS; 2], &[1, 0],
            64, 64, false, canvas,
        ).unwrap();
        assert_eq!(bytes, packet.encode().unwrap(), "public export must preserve the typed packet");
        let summary = scan_packet_commands(&bytes);
        assert_eq!(summary.detailed_depth_cull_pipelines.len(), 1, "visible pipeline retains full color writes");
        assert_eq!(summary.depth_cull_color_pipelines.len(), 1);
        let visible_pipeline = &summary.detailed_depth_cull_pipelines[0];
        let invisible_pipeline = &summary.depth_cull_color_pipelines[0];
        assert_eq!(invisible_pipeline.write_mask, 0);
        for (format, write, compare) in [
            (visible_pipeline.depth_format, visible_pipeline.depth_write_enabled, visible_pipeline.depth_compare),
            (invisible_pipeline.depth_format, invisible_pipeline.depth_write_enabled, invisible_pipeline.depth_compare),
        ] {
            assert_eq!(format, TARGET_FORMAT_DEPTH24PLUS);
            assert!(!write, "the compatibility attachment must not enable depth writes");
            assert_eq!(compare, DEPTH_COMPARE_ALWAYS, "depth testing must remain disabled");
        }
        assert_eq!(summary.draw_pipeline_ids, vec![visible_pipeline.pipeline_id, invisible_pipeline.pipeline_id]);
        assert_eq!(summary.draw_passes.len(), 2);
        assert!(summary.draw_passes.iter().all(|pass| pass.opcode == 13));
    }
}

#[test]
fn test_mesh_batch_cull_depth_default_color_write_preserves_pipeline_ids() {
    let tri = [0.0f32, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0];
    let positions = tri;
    let vertex_counts = [3u32];
    let model_views = IDENTITY_F64;
    let projection = IDENTITY_F64;
    let colors = [1.0f32, 0.0, 0.0, 1.0];
    let cull_modes = [0u8];
    let front_faces = [0u8];
    let depth_tests = [1u8];
    let depth_writes = [1u8];
    let depth_compares = [DEPTH_COMPARE_LESS];

    // 1. Offscreen packet via legacy cull_depth export (all color_writes = true)
    let legacy_bytes = f3d_build_mesh_batch_cull_depth_packet(
        &positions, &vertex_counts, &model_views, &projection, &colors,
        &cull_modes, &front_faces,
        &depth_tests, &depth_writes, &depth_compares,
        64, 64, false, false,
    ).expect("building legacy depth batch");

    // 2. Offscreen packet via cull_depth_color export with color_writes = [1]
    let color_bytes = f3d_build_mesh_batch_cull_depth_color_packet(
        &positions, &vertex_counts, &model_views, &projection, &colors,
        &cull_modes, &front_faces,
        &depth_tests, &depth_writes, &depth_compares,
        &[1u8],
        64, 64, false, false,
    ).expect("building color_write batch");

    // Assert exact byte-identity
    assert_eq!(legacy_bytes, color_bytes, "Default colorWrite=true batch must be byte-identical to legacy cull_depth packet");

    // Assert pipeline ID bit-identity (100 for offscreen)
    let legacy_summary = scan_packet_commands(&legacy_bytes);
    let color_summary = scan_packet_commands(&color_bytes);
    assert_eq!(legacy_summary.depth_cull_pipelines.len(), 1);
    assert_eq!(legacy_summary.depth_cull_pipelines[0].pipeline_id, 100);
    assert_eq!(color_summary.depth_cull_pipelines.len(), 1);
    assert_eq!(color_summary.depth_cull_pipelines[0].pipeline_id, 100);
    assert_eq!(color_summary.depth_cull_color_pipelines.len(), 0);

    // Canvas packet pipeline ID bit-identity (101 for canvas)
    let canvas_bytes = f3d_build_mesh_batch_cull_depth_color_packet(
        &positions, &vertex_counts, &model_views, &projection, &colors,
        &cull_modes, &front_faces,
        &depth_tests, &depth_writes, &depth_compares,
        &[1u8],
        64, 64, false, true,
    ).expect("building canvas default color batch");
    let canvas_summary = scan_packet_commands(&canvas_bytes);
    assert_eq!(canvas_summary.depth_cull_pipelines.len(), 1);
    assert_eq!(canvas_summary.depth_cull_pipelines[0].pipeline_id, 101);
    assert_eq!(canvas_summary.depth_cull_color_pipelines.len(), 0);
}

#[test]
fn test_dynamic_mesh_input_with_vertex_colors() {
    let tri = [0.0f32, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0];
    let colors = [
        1.0f32, 0.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 1.0,
        0.0, 0.0, 1.0, 1.0,
    ];
    let mesh = DynamicMeshInput::try_from_raw(
        &tri,
        &[],
        &IDENTITY_F64,
        &IDENTITY_F64,
        &[1.0, 1.0, 1.0, 1.0],
        64,
        64,
        false,
    )
    .expect("valid dynamic mesh input");

    // Successfully configure vertex colors
    let with_vc = mesh.clone().with_vertex_colors(&colors).expect("valid vertex colors");
    assert_eq!(with_vc.vertex_colors(), Some(&colors[..]));

    // Reject mismatched length (expected 12 floats for 3 vertices, got 4)
    let err = mesh.with_vertex_colors(&colors[..4]).unwrap_err();
    assert!(matches!(
        err,
        MeshPacketError::InvalidVertexColorLength { expected: 12, actual: 4 }
    ));
}

#[test]
fn test_mesh_batch_vertex_color_decoded_bytes_and_stride_28() {
    let tri0 = [
        0.0f32, 0.5, 0.0,
        -0.5, -0.5, 0.0,
        0.5, -0.5, 0.0,
    ];
    let tri1 = [
        1.0f32, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ];
    let positions = [tri0, tri1].concat();
    let vertex_counts = [3u32, 3];
    let model_views = [IDENTITY_F64, IDENTITY_F64].concat();
    let projection = IDENTITY_F64;
    let colors = [
        1.0f32, 1.0, 1.0, 1.0, // Mesh 0 material multiplier
        0.5, 0.5, 0.5, 1.0,    // Mesh 1 material multiplier
    ];
    let cull_modes = [0u8, 0];
    let front_faces = [0u8, 0];
    let depth_tests = [1u8, 1];
    let depth_writes = [1u8, 1];
    let depth_compares = [DEPTH_COMPARE_LESS, DEPTH_COMPARE_LESS];
    let color_writes = [1u8, 1];

    let vc0 = [
        1.0f32, 0.0, 0.0, 1.0, // Red
        0.0, 1.0, 0.0, 1.0,    // Green
        0.0, 0.0, 1.0, 1.0,    // Blue
    ];
    let vc1 = [
        1.0f32, 1.0, 0.0, 1.0, // Yellow
        0.0, 1.0, 1.0, 1.0,    // Cyan
        1.0, 0.0, 1.0, 1.0,    // Magenta
    ];
    let vertex_colors = [vc0, vc1].concat();

    let packet_bytes = f3d_build_mesh_batch_vertex_color_packet(
        &positions,
        &vertex_counts,
        &model_views,
        &projection,
        &colors,
        &cull_modes,
        &front_faces,
        &depth_tests,
        &depth_writes,
        &depth_compares,
        &color_writes,
        64,
        64,
        false,
        false,
        &vertex_colors,
    )
    .expect("building vertex color batch");

    let summary = scan_packet_commands(&packet_bytes);

    // 1. Assert all created pipelines specify vertex_stride == 28
    assert!(!summary.vertex_strides.is_empty(), "must create at least one pipeline");
    for &stride in &summary.vertex_strides {
        assert_eq!(stride, 28, "vertex colors pipeline must specify stride 28");
    }

    // 2. Assert vertex buffer upload exists, has exact size 6 * 28 = 168 bytes
    let vb_upload = summary
        .write_buffers
        .iter()
        .find(|(buf_id, _, _)| *buf_id == MESH_VERTEX_BUFFER_ID)
        .expect("vertex buffer upload must exist in packet");
    assert_eq!(vb_upload.1, 0, "upload starts at offset 0");
    let upload_data = &vb_upload.2;
    assert_eq!(upload_data.len(), 6 * 28, "6 vertices * 28 bytes per vertex");

    // 3. Verify each vertex has exact 28 bytes: 12 bytes position + 16 bytes color
    for i in 0..6 {
        let chunk = &upload_data[i * 28..(i + 1) * 28];
        let px = f32::from_le_bytes(chunk[0..4].try_into().unwrap());
        let py = f32::from_le_bytes(chunk[4..8].try_into().unwrap());
        let pz = f32::from_le_bytes(chunk[8..12].try_into().unwrap());
        assert_eq!([px, py, pz], [positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]]);

        let cr = f32::from_le_bytes(chunk[12..16].try_into().unwrap());
        let cg = f32::from_le_bytes(chunk[16..20].try_into().unwrap());
        let cb = f32::from_le_bytes(chunk[20..24].try_into().unwrap());
        let ca = f32::from_le_bytes(chunk[24..28].try_into().unwrap());
        assert_eq!(
            [cr, cg, cb, ca],
            [
                vertex_colors[i * 4],
                vertex_colors[i * 4 + 1],
                vertex_colors[i * 4 + 2],
                vertex_colors[i * 4 + 3],
            ]
        );
    }
}

#[test]
fn test_mesh_batch_vertex_color_mixed_uncolored_receives_white_color() {
    let tri0 = [0.0f32, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0];
    let tri1 = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let vc0 = [
        1.0f32, 0.2, 0.3, 1.0,
        0.4, 1.0, 0.6, 1.0,
        0.7, 0.8, 1.0, 1.0,
    ];

    let mesh0 = DynamicMeshInput::try_from_raw(
        &tri0,
        &[],
        &IDENTITY_F64,
        &IDENTITY_F64,
        &[1.0, 0.5, 0.25, 1.0],
        64,
        64,
        false,
    )
    .unwrap()
    .with_vertex_colors(&vc0)
    .unwrap();

    // mesh1 has NO vertex colors (None)
    let mesh1 = DynamicMeshInput::try_from_raw(
        &tri1,
        &[],
        &IDENTITY_F64,
        &IDENTITY_F64,
        &[0.2, 0.4, 0.8, 1.0],
        64,
        64,
        false,
    )
    .unwrap();

    let submission = build_multi_mesh_submission(&[mesh0, mesh1])
        .expect("multi-mesh submission with mixed vertex colors");
    let packet_bytes = submission.encode().expect("encoding submission");
    let summary = scan_packet_commands(&packet_bytes);

    // Both meshes share stride 28
    assert_eq!(summary.vertex_strides, vec![28]);

    let vb_upload = summary
        .write_buffers
        .iter()
        .find(|(buf_id, _, _)| *buf_id == MESH_VERTEX_BUFFER_ID)
        .expect("vertex buffer upload");
    let data = &vb_upload.2;
    assert_eq!(data.len(), 6 * 28);

    // Mesh 0 has specified colors
    for i in 0..3 {
        let chunk = &data[i * 28..(i + 1) * 28];
        let cr = f32::from_le_bytes(chunk[12..16].try_into().unwrap());
        let cg = f32::from_le_bytes(chunk[16..20].try_into().unwrap());
        let cb = f32::from_le_bytes(chunk[20..24].try_into().unwrap());
        let ca = f32::from_le_bytes(chunk[24..28].try_into().unwrap());
        assert_eq!([cr, cg, cb, ca], [vc0[i * 4], vc0[i * 4 + 1], vc0[i * 4 + 2], vc0[i * 4 + 3]]);
    }

    // Mesh 1 (uncolored) has white fallback [1.0, 1.0, 1.0, 1.0] for all vertices
    for i in 3..6 {
        let chunk = &data[i * 28..(i + 1) * 28];
        let cr = f32::from_le_bytes(chunk[12..16].try_into().unwrap());
        let cg = f32::from_le_bytes(chunk[16..20].try_into().unwrap());
        let cb = f32::from_le_bytes(chunk[20..24].try_into().unwrap());
        let ca = f32::from_le_bytes(chunk[24..28].try_into().unwrap());
        assert_eq!([cr, cg, cb, ca], [1.0, 1.0, 1.0, 1.0], "uncolored mesh must receive white vertex color fallback");
    }
}

#[test]
fn test_mesh_batch_vertex_color_shader_semantics_offscreen_and_canvas() {
    use f3d_runtime::mesh::generate_mesh_wgsl;

    // Standard legacy shader (no vertex color)
    let legacy_wgsl = generate_mesh_wgsl(false);
    assert!(!legacy_wgsl.contains("@location(1) color: vec4<f32>"));
    assert!(legacy_wgsl.contains("@location(1) uv: vec2<f32>"));
    assert!(legacy_wgsl.contains("return uniforms.color;"));

    // Offscreen with vertex colors
    let tri = [0.0f32, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0];
    let vc = [1.0f32, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0];
    let mesh = DynamicMeshInput::try_from_raw(
        &tri, &[], &IDENTITY_F64, &IDENTITY_F64, &[1.0, 1.0, 1.0, 1.0], 64, 64, false,
    ).unwrap().with_vertex_colors(&vc).unwrap();

    let offscreen_sub = build_multi_mesh_submission(&[mesh.clone()]).unwrap();
    let offscreen_cmd = offscreen_sub.commands().iter().find(|c| matches!(c, GpuCommand::CreatePipeline { .. })).unwrap();
    if let GpuCommand::CreatePipeline { wgsl_code, vertex_stride, .. } = offscreen_cmd {
        assert_eq!(*vertex_stride, 28);
        assert!(wgsl_code.contains("@location(1) color: vec4<f32>"));
        assert!(wgsl_code.contains("out.color = in.color;"));
        assert!(wgsl_code.contains("return in.color * uniforms.color;"));
        assert!(!wgsl_code.contains("linear_to_srgb"));
    } else {
        panic!("expected CreatePipeline command");
    }

    // Canvas with vertex colors
    let canvas_sub = build_multi_mesh_canvas_submission(&[mesh]).unwrap();
    let canvas_cmd = canvas_sub.commands().iter().find(|c| matches!(c, GpuCommand::CreatePipeline { .. })).unwrap();
    if let GpuCommand::CreatePipeline { wgsl_code, vertex_stride, .. } = canvas_cmd {
        assert_eq!(*vertex_stride, 28);
        assert!(wgsl_code.contains("@location(1) color: vec4<f32>"));
        assert!(wgsl_code.contains("out.color = in.color;"));
        assert!(wgsl_code.contains("let linear_color = in.color * uniforms.color;"));
        assert!(wgsl_code.contains("return linear_to_srgb(linear_color);"));
    } else {
        panic!("expected CreatePipeline command");
    }
}

#[test]
fn test_mesh_batch_vertex_color_validation_errors() {
    let tri = [0.0f32, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0];
    let positions = tri;
    let vertex_counts = [3u32];
    let model_views = IDENTITY_F64;
    let projection = IDENTITY_F64;
    let colors = [1.0f32, 0.0, 0.0, 1.0];
    let cull_modes = [0u8];
    let front_faces = [0u8];
    let depth_tests = [1u8];
    let depth_writes = [1u8];
    let depth_compares = [DEPTH_COMPARE_LESS];
    let color_writes = [1u8];

    // Slices length mismatch: 3 vertices require 3*4 = 12 floats, passed 8 floats
    let err = f3d_build_mesh_batch_vertex_color_packet(
        &positions,
        &vertex_counts,
        &model_views,
        &projection,
        &colors,
        &cull_modes,
        &front_faces,
        &depth_tests,
        &depth_writes,
        &depth_compares,
        &color_writes,
        64,
        64,
        false,
        false,
        &[1.0f32; 8],
    )
    .unwrap_err();
    assert!(err.contains("vertex_colors array length must match total vertex count * 4 = 12 (got 8)"));
}

#[test]
fn test_mesh_batch_vertex_color_preserves_old_packets_byte_identical() {
    let tri = [0.0f32, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0];
    let positions = tri;
    let vertex_counts = [3u32];
    let model_views = IDENTITY_F64;
    let projection = IDENTITY_F64;
    let colors = [1.0f32, 0.0, 0.0, 1.0];
    let cull_modes = [0u8];
    let front_faces = [0u8];
    let depth_tests = [1u8];
    let depth_writes = [1u8];
    let depth_compares = [DEPTH_COMPARE_LESS];
    let color_writes = [1u8];

    let cull_depth_color_bytes = f3d_build_mesh_batch_cull_depth_color_packet(
        &positions,
        &vertex_counts,
        &model_views,
        &projection,
        &colors,
        &cull_modes,
        &front_faces,
        &depth_tests,
        &depth_writes,
        &depth_compares,
        &color_writes,
        64,
        64,
        false,
        false,
    )
    .expect("cull_depth_color packet");

    let impl_bytes = build_mesh_batch_cull_depth_color_packet_impl(
        &positions,
        &vertex_counts,
        &model_views,
        &projection,
        &colors,
        &cull_modes,
        &front_faces,
        &depth_tests,
        &depth_writes,
        &depth_compares,
        &color_writes,
        64,
        64,
        false,
        false,
    )
    .expect("cull_depth_color impl packet");

    assert_eq!(cull_depth_color_bytes, impl_bytes);

    let summary = scan_packet_commands(&cull_depth_color_bytes);
    // Uncolored batch uses stride 20 (VertexPosUv)
    assert_eq!(summary.vertex_strides, vec![20]);
}
