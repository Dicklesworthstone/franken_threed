//! Integration and contract tests for `f3d-core` GPU wire layout and byte encoding.
//!
//! Bead: `f3d-05-ids-layouts-epochs-transport-vqa.2`
//! Owned path: `crates/f3d-core/tests/layout_tests.rs`
//!
//! Note on Claims (Binding):
//! Native layout tests verify byte-exact memory packing, row mapping conventions,
//! and WebGPU/WGSL alignment constraints. They do NOT prove GPU execution or complete
//! matrix semantics (which belong to `f3d-math`, `f3d-shader`, and `f3d-gpu`).

use f3d_core::layout::*;

#[test]
fn affine_rows_exact_size_and_alignment() {
    assert_eq!(AffineRows::byte_size(), 48);
    assert_eq!(AffineRows::alignment(), 16);
    assert_eq!(AFFINE_ROWS_BYTES, 48);
    assert_eq!(AFFINE_ROWS_ALIGNMENT, 16);

    let id = AffineRows::identity();
    assert_eq!(id.r0, [1.0, 0.0, 0.0, 0.0]);
    assert_eq!(id.r1, [0.0, 1.0, 0.0, 0.0]);
    assert_eq!(id.r2, [0.0, 0.0, 1.0, 0.0]);
}

#[test]
fn affine_rows_mapping_from_column_major() {
    // Column-major Matrix4 representation:
    // Col 0: e0,  e1,  e2,  e3
    // Col 1: e4,  e5,  e6,  e7
    // Col 2: e8,  e9,  e10, e11
    // Col 3: e12, e13, e14, e15
    let e = [
        1.0, 2.0, 3.0, 0.0,   // col 0
        4.0, 5.0, 6.0, 0.0,   // col 1
        7.0, 8.0, 9.0, 0.0,   // col 2
        10.0, 11.0, 12.0, 1.0, // col 3
    ];

    let affine = AffineRows::from_column_major(&e).expect("valid affine matrix");

    // Check row mapping:
    // Row 0: [e0, e4, e8, e12]
    assert_eq!(affine.r0, [1.0, 4.0, 7.0, 10.0]);
    // Row 1: [e1, e5, e9, e13]
    assert_eq!(affine.r1, [2.0, 5.0, 8.0, 11.0]);
    // Row 2: [e2, e6, e10, e14]
    assert_eq!(affine.r2, [3.0, 6.0, 9.0, 12.0]);

    // Round-trip back to column-major matrix
    let restored = affine.to_column_major();
    assert_eq!(restored, e);
}

#[test]
fn positive_translation_roundtrip_byte_exact() {
    let tx = 15.5f32;
    let ty = -42.25f32;
    let tz = 100.125f32;

    let e = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        tx,  ty,  tz,  1.0,
    ];

    let affine = AffineRows::from_column_major(&e).expect("translation is affine");
    assert_eq!(affine.r0, [1.0, 0.0, 0.0, tx]);
    assert_eq!(affine.r1, [0.0, 1.0, 0.0, ty]);
    assert_eq!(affine.r2, [0.0, 0.0, 1.0, tz]);

    // Byte serialization
    let bytes = affine.to_bytes();
    assert_eq!(bytes.len(), 48);

    // Byte deserialization
    let restored = AffineRows::from_bytes(&bytes);
    assert_eq!(affine, restored);

    // Slices round-trip
    let mut buf = [0u8; 48];
    affine.write_to_slice(&mut buf).expect("write ok");
    assert_eq!(buf, bytes);
    let from_slice = AffineRows::read_from_slice(&buf).expect("read ok");
    assert_eq!(affine, from_slice);

    // Transform evaluation
    let origin = [0.0, 0.0, 0.0];
    assert_eq!(affine.transform_point(origin), [tx, ty, tz]);

    let pt = [1.0, 2.0, 3.0];
    assert_eq!(affine.transform_point(pt), [1.0 + tx, 2.0 + ty, 3.0 + tz]);

    // Direction vector should not translate
    let dir = [1.0, 2.0, 3.0];
    assert_eq!(affine.transform_vector(dir), [1.0, 2.0, 3.0]);
}

#[test]
fn positive_negative_scale_roundtrip_byte_exact() {
    let sx = -2.0f32;
    let sy = 3.5f32;
    let sz = -0.5f32;

    let e = [
        sx,  0.0, 0.0, 0.0,
        0.0, sy,  0.0, 0.0,
        0.0, 0.0, sz,  0.0,
        0.0, 0.0, 0.0, 1.0,
    ];

    let affine = AffineRows::from_column_major(&e).expect("scale is affine");
    assert_eq!(affine.r0, [sx, 0.0, 0.0, 0.0]);
    assert_eq!(affine.r1, [0.0, sy, 0.0, 0.0]);
    assert_eq!(affine.r2, [0.0, 0.0, sz, 0.0]);

    let bytes = affine.to_bytes();
    let restored = AffineRows::from_bytes(&bytes);
    assert_eq!(affine, restored);

    let pt = [10.0, -4.0, 8.0];
    assert_eq!(affine.transform_point(pt), [10.0 * sx, -4.0 * sy, 8.0 * sz]);
}

#[test]
fn positive_shear_and_composite_roundtrip_byte_exact() {
    // Shear transform with non-zero off-diagonals:
    // x' = x + 0.5*y - 1.5*z + 10.0
    // y' = y + 2.0*z - 20.0
    // z' = z + 30.0
    let r0 = [1.0, 0.5, -1.5, 10.0];
    let r1 = [0.0, 1.0, 2.0, -20.0];
    let r2 = [0.0, 0.0, 1.0, 30.0];

    let affine = AffineRows::new(r0, r1, r2);
    let col_major = affine.to_column_major();

    let reconstructed = AffineRows::from_column_major(&col_major).expect("affine shear");
    assert_eq!(affine, reconstructed);

    let bytes = affine.to_bytes();
    let from_bytes = AffineRows::from_bytes(&bytes);
    assert_eq!(affine, from_bytes);

    let test_pt = [2.0, 4.0, -6.0];
    let expected_x = 1.0 * 2.0 + 0.5 * 4.0 + (-1.5) * (-6.0) + 10.0; // 2 + 2 + 9 + 10 = 23
    let expected_y = 0.0 * 2.0 + 1.0 * 4.0 + 2.0 * (-6.0) - 20.0;    // 4 - 12 - 20 = -28
    let expected_z = 0.0 * 2.0 + 0.0 * 4.0 + 1.0 * (-6.0) + 30.0;    // -6 + 30 = 24

    assert_eq!(affine.transform_point(test_pt), [expected_x, expected_y, expected_z]);
}

#[test]
fn negative_non_affine_matrix_rejected_and_projective_retained() {
    // Perspective projection matrix (e11 = -1.0, e15 = 0.0, row 3 is not [0, 0, 0, 1])
    let proj_elements = [
        1.299, 0.0,   0.0,    0.0,
        0.0,   1.732, 0.0,    0.0,
        0.0,   0.0,   -1.002, -1.0,
        0.0,   0.0,   -0.2,   0.0,
    ];

    // Attempting to convert into AffineRows must be rejected
    let err = AffineRows::from_column_major(&proj_elements).expect_err("perspective is not affine");
    assert_eq!(err, LayoutError::NonAffineMatrix);

    // Must be retained as full ProjectiveMat4 (64 bytes)
    let proj = ProjectiveMat4::from_elements(proj_elements);
    assert!(!proj.is_affine());
    assert_eq!(proj.to_affine_rows().expect_err("cannot be affine"), LayoutError::NonAffineMatrix);

    // Byte round-trip for 64-byte projective matrix
    let bytes = proj.to_bytes();
    assert_eq!(bytes.len(), 64);
    let restored = ProjectiveMat4::from_bytes(&bytes);
    assert_eq!(proj, restored);

    let mut buf = [0u8; 64];
    proj.write_to_slice(&mut buf).expect("write projective ok");
    let from_slice = ProjectiveMat4::read_from_slice(&buf).expect("read projective ok");
    assert_eq!(proj, from_slice);
}

#[test]
fn negative_wgsl_mat4x3_padded_64b_rejected_for_48b_packed() {
    // In WGSL, mat4x3 is 4 columns of vec3<f32>.
    // alignof(vec3) == 16, so each column occupies 16 bytes.
    // 4 columns * 16 bytes = 64 bytes total.
    assert_eq!(GpuMatrixLayout::WgslMat4x3Padded64.byte_size(), 64);
    assert_eq!(GpuMatrixLayout::AffineRows48.byte_size(), 48);

    // The validator must explicitly reject targeting a WGSL mat4x3 buffer with packed 48-byte AffineRows
    let err = validate_affine_target(GpuMatrixLayout::WgslMat4x3Padded64)
        .expect_err("mat4x3 is 64 bytes and cannot accept 48-byte packed affine");
    assert_eq!(
        err,
        LayoutError::IncompatibleTargetLayout {
            target_size: 64,
            source_size: 48,
        }
    );

    // Targeting AffineRows48 must succeed
    assert!(validate_affine_target(GpuMatrixLayout::AffineRows48).is_ok());
}

#[test]
fn negative_dynamic_uniform_alignment_256_is_not_storage_stride_48() {
    let min_uniform_alignment = DEFAULT_MIN_UNIFORM_BUFFER_OFFSET_ALIGNMENT; // 256
    let storage_stride = AFFINE_ROWS_BYTES; // 48

    // Offset 0 is aligned for both
    assert!(validate_dynamic_uniform_offset(0, min_uniform_alignment).is_ok());

    // Storage stride offset (48) is NOT aligned for dynamic uniforms (requires multiple of 256)
    let err = validate_dynamic_uniform_offset(storage_stride, min_uniform_alignment)
        .expect_err("48 is not a multiple of 256");
    assert_eq!(
        err,
        LayoutError::UnalignedOffset {
            offset: 48,
            required_alignment: 256,
        }
    );

    // Multiples of 48 (96, 144, 192) fail dynamic uniform offset validation
    assert!(validate_dynamic_uniform_offset(96, min_uniform_alignment).is_err());
    assert!(validate_dynamic_uniform_offset(144, min_uniform_alignment).is_err());

    // Multiple of 256 passes dynamic uniform offset validation
    assert!(validate_dynamic_uniform_offset(256, min_uniform_alignment).is_ok());
    assert!(validate_dynamic_uniform_offset(512, min_uniform_alignment).is_ok());

    // Storage array stride validation: 48 is a valid storage stride (multiple of 16 and >= 48)
    assert!(validate_storage_array_stride(48, 48).is_ok());
    // Stride smaller than element size is rejected
    assert_eq!(
        validate_storage_array_stride(32, 48),
        Err(LayoutError::BufferTooSmall { required: 48, provided: 32 })
    );
    // Stride not aligned to 16 bytes is rejected
    assert_eq!(
        validate_storage_array_stride(50, 48),
        Err(LayoutError::UnalignedOffset { offset: 50, required_alignment: 16 })
    );
}

#[test]
fn negative_undersized_and_overflow_output_rejected() {
    let affine = AffineRows::identity();

    // 47 bytes is 1 byte short of 48 bytes
    let mut small_buf = [0u8; 47];
    let err = affine.write_to_slice(&mut small_buf).expect_err("undersized buffer must fail");
    assert_eq!(
        err,
        LayoutError::BufferTooSmall {
            required: 48,
            provided: 47,
        }
    );

    let read_err = AffineRows::read_from_slice(&small_buf).expect_err("undersized read must fail");
    assert_eq!(
        read_err,
        LayoutError::BufferTooSmall {
            required: 48,
            provided: 47,
        }
    );

    // ProjectiveMat4 needs 64 bytes
    let proj = ProjectiveMat4::identity();
    let mut small_proj_buf = [0u8; 63];
    assert_eq!(
        proj.write_to_slice(&mut small_proj_buf),
        Err(LayoutError::BufferTooSmall { required: 64, provided: 63 })
    );
    assert_eq!(
        ProjectiveMat4::read_from_slice(&small_proj_buf),
        Err(LayoutError::BufferTooSmall { required: 64, provided: 63 })
    );
}

#[test]
fn operation_specific_alignments_writebuffer_and_texture_copies() {
    // WebGPU writeBuffer requires 4-byte offset and size alignment
    assert!(validate_write_buffer_alignment(0, 48).is_ok());
    assert!(validate_write_buffer_alignment(4, 64).is_ok());
    assert_eq!(
        validate_write_buffer_alignment(2, 48),
        Err(LayoutError::UnalignedWriteBuffer { value: 2 })
    );
    assert_eq!(
        validate_write_buffer_alignment(0, 47),
        Err(LayoutError::UnalignedWriteBuffer { value: 47 })
    );

    // WebGPU copyBytesPerRow requires multiple of 256 bytes
    assert!(validate_copy_bytes_per_row(256).is_ok());
    assert!(validate_copy_bytes_per_row(512).is_ok());
    assert!(validate_copy_bytes_per_row(1024).is_ok());
    assert_eq!(
        validate_copy_bytes_per_row(300),
        Err(LayoutError::UnalignedBytesPerRow {
            bytes_per_row: 300,
            required_alignment: 256,
        })
    );
}

#[test]
fn instance_record_and_indirect_args_wire_packing() {
    let inst = InstanceRecord::new(AffineRows::identity(), 77);
    assert_eq!(InstanceRecord::BYTE_SIZE, 64);
    assert_eq!(InstanceRecord::ALIGNMENT, 16);

    let mut buf = [0u8; 64];
    inst.write_to_slice(&mut buf).expect("write instance ok");
    let restored = InstanceRecord::read_from_slice(&buf).expect("read instance ok");
    assert_eq!(inst, restored);
    assert_eq!(restored.instance_id, 77);

    // DrawIndirectArgs (16 bytes)
    let draw_args = DrawIndirectArgs {
        vertex_count: 36,
        instance_count: 5,
        first_vertex: 0,
        first_instance: 1,
    };
    let mut draw_buf = [0u8; 16];
    draw_args.write_to_slice(&mut draw_buf).expect("write draw args ok");
    assert_eq!(&draw_buf[0..4], &36u32.to_le_bytes());
    assert_eq!(&draw_buf[4..8], &5u32.to_le_bytes());

    // DrawIndexedIndirectArgs (20 bytes)
    let indexed_args = DrawIndexedIndirectArgs {
        index_count: 128,
        instance_count: 2,
        first_index: 0,
        base_vertex: -4,
        first_instance: 0,
    };
    let mut indexed_buf = [0u8; 20];
    indexed_args.write_to_slice(&mut indexed_buf).expect("write indexed args ok");
    assert_eq!(&indexed_buf[0..4], &128u32.to_le_bytes());
    assert_eq!(&indexed_buf[12..16], &(-4i32).to_le_bytes());
}

#[test]
fn generated_wgsl_contains_affine_declarations_for_bridge() {
    let wgsl = generate_wgsl_declarations();

    // Verify key struct definitions consumed by ChartreuseFern's bridge
    assert!(wgsl.contains("struct AffineRows {"));
    assert!(wgsl.contains("r0: vec4<f32>"));
    assert!(wgsl.contains("r1: vec4<f32>"));
    assert!(wgsl.contains("r2: vec4<f32>"));
    assert!(wgsl.contains("fn transform_affine_point(m: AffineRows, p: vec3<f32>) -> vec3<f32>"));
    assert!(wgsl.contains("fn affine_to_mat4x4(m: AffineRows) -> mat4x4<f32>"));
    assert!(wgsl.contains("struct ProjectiveMat4 {"));
}
