//! Integration and contract tests for `f3d-core` GPU wire layout and byte encoding.
//!
//! Beads: `f3d-05-ids-layouts-epochs-transport-vqa.2`, `f3d-05-ids-layouts-epochs-transport-vqa.3`
//! Owned path: `crates/f3d-core/tests/layout_tests.rs`
//!
//! Note on Claims (Binding):
//! Native layout tests verify byte-exact memory packing, row mapping conventions,
//! and WebGPU/WGSL alignment constraints. They do NOT prove GPU execution or complete
//! matrix semantics (which belong to `f3d-math`, `f3d-shader`, and `f3d-gpu`).

use f3d_core::layout::*;
use f3d_core::{DataVersion, Epoch};

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

    // Storage array stride validation: 48 is a valid storage stride for 16-byte aligned AffineRows
    assert!(validate_storage_array_stride(48, 48, 16).is_ok());
    assert!(validate_composite_storage_array_stride(48, 48).is_ok());
    // Stride smaller than element size is rejected
    assert_eq!(
        validate_storage_array_stride(32, 48, 16),
        Err(LayoutError::BufferTooSmall { required: 48, provided: 32 })
    );
    // Stride not aligned to 16 bytes is rejected
    assert_eq!(
        validate_storage_array_stride(50, 48, 16),
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

#[test]
fn regression_exact_structural_affine_preserves_small_projective_terms() {
    // Regression for root review defect:
    // Any epsilon check (e.g. 1e-6) illegally discards small perspective terms.
    // Analytical counterexample:
    // Let e[3] = 5e-7, and point x = 1e7, y = 0, z = 0, w = 1.
    // Homogeneous w' = e[3]*x + e[7]*y + e[11]*z + e[15]*w
    //               = (5e-7 * 1e7) + 0 + 0 + 1 = 5.0 + 1.0 = 6.0 != 1.0.
    // Perspective divide yields x'/w' = 1e7 / 6.0 approx 1.666667e6.
    // Silently discarding e[3] as "zero" yields w' = 1.0 and x'/w' = 1e7 (a 6x geometric error!).
    let elements = [
        1.0f32, 0.0, 0.0, 5e-7,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];

    // Must fail exact structural check
    assert!(!is_matrix4_affine(&elements));
    let err = AffineRows::from_column_major(&elements).expect_err("must reject small perspective term");
    assert_eq!(err, LayoutError::NonAffineMatrix);

    // Must be retained as ProjectiveMat4
    let proj = ProjectiveMat4::from_elements(elements);
    assert!(!proj.is_affine());
    let h = proj.transform_homogeneous([1e7, 0.0, 0.0, 1.0]);
    assert_eq!(h[3], 6.0);
    assert_eq!(h[0], 1e7);

    // Subnormal perspective component (e.g. 1e-40 or f32::from_bits(1))
    let mut subnormal_elements = [
        1.0f32, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    subnormal_elements[7] = f32::from_bits(1); // smallest positive subnormal f32
    assert!(!is_matrix4_affine(&subnormal_elements));
    assert_eq!(
        AffineRows::from_column_major(&subnormal_elements),
        Err(LayoutError::NonAffineMatrix)
    );

    // Near-1 e15 values: e15 must be exactly 1.0
    let near_one_high = [
        1.0f32, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0 + 1e-7,
    ];
    assert!(!is_matrix4_affine(&near_one_high));
    assert_eq!(
        AffineRows::from_column_major(&near_one_high),
        Err(LayoutError::NonAffineMatrix)
    );

    let near_one_low = [
        1.0f32, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 0.9999999,
    ];
    assert!(!is_matrix4_affine(&near_one_low));
    assert_eq!(
        AffineRows::from_column_major(&near_one_low),
        Err(LayoutError::NonAffineMatrix)
    );
}

#[test]
fn regression_f64_source_eligibility_before_narrowing() {
    // In f64, small perspective terms must be verified BEFORE narrowing to f32.
    // If a term is non-zero in f64, narrowing or treating as affine would illegally drop it.
    let f64_elements_with_perspective = [
        1.0f64, 0.0, 0.0, 1e-15,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    assert!(!is_matrix4_f64_affine(&f64_elements_with_perspective));
    assert_eq!(
        AffineRows::from_column_major_f64(&f64_elements_with_perspective),
        Err(LayoutError::NonAffineMatrix)
    );

    // Valid f64 affine matrix transforms accurately when converted
    let f64_affine = [
        2.0f64, 0.0, 0.0, 0.0,
        0.0, 3.0, 0.0, 0.0,
        0.0, 0.0, 4.0, 0.0,
        10.0, 20.0, 30.0, 1.0,
    ];
    assert!(is_matrix4_f64_affine(&f64_affine));
    let affine = AffineRows::from_column_major_f64(&f64_affine).expect("valid f64 affine");
    assert_eq!(affine.r0, [2.0, 0.0, 0.0, 10.0]);
    assert_eq!(affine.r1, [0.0, 3.0, 0.0, 20.0]);
    assert_eq!(affine.r2, [0.0, 0.0, 4.0, 30.0]);

    // Retained ProjectiveMat4 from f64 elements
    let proj = ProjectiveMat4::from_elements_f64(&f64_elements_with_perspective);
    assert_eq!(proj.elements[0], 1.0f32);
}

#[test]
fn storage_array_stride_lawful_scalar_vec2_and_composite() {
    // Lawful scalar storage array: f32 (size 4, alignment 4)
    assert!(validate_storage_array_stride(4, 4, 4).is_ok());
    assert!(validate_storage_array_stride(8, 4, 4).is_ok());
    assert!(validate_storage_array_stride(16, 4, 4).is_ok());
    assert_eq!(
        validate_storage_array_stride(2, 4, 4),
        Err(LayoutError::BufferTooSmall { required: 4, provided: 2 })
    );
    assert_eq!(
        validate_storage_array_stride(6, 4, 4),
        Err(LayoutError::UnalignedOffset { offset: 6, required_alignment: 4 })
    );

    // Lawful vec2<f32> storage array: (size 8, alignment 8)
    assert!(validate_storage_array_stride(8, 8, 8).is_ok());
    assert!(validate_storage_array_stride(16, 8, 8).is_ok());
    assert!(validate_storage_array_stride(24, 8, 8).is_ok());
    assert_eq!(
        validate_storage_array_stride(4, 8, 8),
        Err(LayoutError::BufferTooSmall { required: 8, provided: 4 })
    );
    assert_eq!(
        validate_storage_array_stride(12, 8, 8),
        Err(LayoutError::UnalignedOffset { offset: 12, required_alignment: 8 })
    );

    // Composite 16-byte aligned record: AffineRows (size 48, alignment 16)
    assert!(validate_storage_array_stride(48, 48, 16).is_ok());
    assert!(validate_storage_array_stride(64, 48, 16).is_ok());
    assert!(validate_composite_storage_array_stride(48, 48).is_ok());
    assert!(validate_composite_storage_array_stride(64, 48).is_ok());
    assert_eq!(
        validate_composite_storage_array_stride(50, 48),
        Err(LayoutError::UnalignedOffset { offset: 50, required_alignment: 16 })
    );
}

#[test]
fn aligned_bytes_per_row_regression_tests() {
    // Regression for root 5571 item 5:
    // Readback row pitch for width 32 (128 bytes) must become 256 bytes per WebGPU COPY_BYTES_PER_ROW_ALIGNMENT.
    assert_eq!(aligned_bytes_per_row(32), Ok(256));

    // Naturally aligned width 64 (256 bytes) stays 256
    assert_eq!(aligned_bytes_per_row(64), Ok(256));

    // Width 65: 65 * 4 = 260 bytes -> rounded up to 512 bytes
    assert_eq!(aligned_bytes_per_row(65), Ok(512));

    // Validated by validate_copy_bytes_per_row
    assert!(validate_copy_bytes_per_row(aligned_bytes_per_row(32).unwrap()).is_ok());
    assert!(validate_copy_bytes_per_row(aligned_bytes_per_row(65).unwrap()).is_ok());

    // Overflowing width: u32::MAX * 4 overflows u32
    assert_eq!(aligned_bytes_per_row(u32::MAX), Err(LayoutError::CalculationOverflow));
    assert_eq!(aligned_bytes_per_row(u32::MAX / 4 + 1), Err(LayoutError::CalculationOverflow));

    // Alignment overflow: 1_073_741_823 * 4 = 4294967292, rounding up by 4 overflows u32
    assert_eq!(aligned_bytes_per_row(1_073_741_823), Err(LayoutError::CalculationOverflow));
}

#[test]
fn vertex_pos_uv_and_color_uniform_wire_invariants() {
    // 1. Color uniform: 16 bytes, 16-byte aligned
    assert_eq!(COLOR_UNIFORM_BYTES, 16);
    assert_eq!(COLOR_UNIFORM_ALIGNMENT, 16);

    // 2. VertexPosUv: 20 bytes, 4-byte aligned, stride 20
    assert_eq!(VERTEX_POS_UV_BYTES, 20);
    assert_eq!(VERTEX_POS_UV_STRIDE, 20);
    assert_eq!(VERTEX_POS_UV_ALIGNMENT, 4);
    assert_eq!(VertexPosUv::BYTE_SIZE, 20);
    assert_eq!(VertexPosUv::STRIDE, 20);

    // 3. Triangle 3-vertex buffer layout: 3 * 20 = 60 bytes
    let vertices = [
        VertexPosUv::new([0.0, 0.5, 0.0], [0.5, 1.0]),
        VertexPosUv::new([-0.5, -0.5, 0.0], [0.0, 0.0]),
        VertexPosUv::new([0.5, -0.5, 0.0], [1.0, 0.0]),
    ];

    let mut vertex_bytes = [0u8; 60];
    for (i, v) in vertices.iter().enumerate() {
        v.write_to_slice(&mut vertex_bytes[i * 20..(i + 1) * 20]).expect("write vertex");
    }

    // Verify first vertex position float bytes at offsets 0, 4, 8 and uv at 12, 16
    let v0_pos_x = f32::from_le_bytes([vertex_bytes[0], vertex_bytes[1], vertex_bytes[2], vertex_bytes[3]]);
    let v0_pos_y = f32::from_le_bytes([vertex_bytes[4], vertex_bytes[5], vertex_bytes[6], vertex_bytes[7]]);
    let v0_uv_u = f32::from_le_bytes([vertex_bytes[12], vertex_bytes[13], vertex_bytes[14], vertex_bytes[15]]);
    let v0_uv_v = f32::from_le_bytes([vertex_bytes[16], vertex_bytes[17], vertex_bytes[18], vertex_bytes[19]]);
    assert_eq!(v0_pos_x, 0.0);
    assert_eq!(v0_pos_y, 0.5);
    assert_eq!(v0_uv_u, 0.5);
    assert_eq!(v0_uv_v, 1.0);

    // Verify round-trip deserialization from buffer slice
    let restored_v0 = VertexPosUv::read_from_slice(&vertex_bytes[0..20]).expect("read vertex 0");
    assert_eq!(restored_v0, vertices[0]);
    let restored_v1 = VertexPosUv::read_from_slice(&vertex_bytes[20..40]).expect("read vertex 1");
    assert_eq!(restored_v1, vertices[1]);
    let restored_v2 = VertexPosUv::read_from_slice(&vertex_bytes[40..60]).expect("read vertex 2");
    assert_eq!(restored_v2, vertices[2]);
}

#[test]
fn vertex_pos_normal_uv_and_color_offsets_and_sizes() {
    // 1. VertexPosNormalUv: 32 bytes, 4-byte aligned, stride 32
    assert_eq!(core::mem::size_of::<VertexPosNormalUv>(), 32);
    assert_eq!(core::mem::align_of::<VertexPosNormalUv>(), 4);
    assert_eq!(core::mem::offset_of!(VertexPosNormalUv, position), 0);
    assert_eq!(core::mem::offset_of!(VertexPosNormalUv, normal), 12);
    assert_eq!(core::mem::offset_of!(VertexPosNormalUv, uv), 24);
    assert_eq!(VERTEX_POS_NORMAL_UV_BYTES, 32);
    assert_eq!(VERTEX_POS_NORMAL_UV_STRIDE, 32);
    assert_eq!(VERTEX_POS_NORMAL_UV_ALIGNMENT, 4);
    assert_eq!(VertexPosNormalUv::BYTE_SIZE, 32);
    assert_eq!(VertexPosNormalUv::STRIDE, 32);
    assert_eq!(VertexPosNormalUv::ALIGNMENT, 4);

    // 2. VertexPosColor: 28 bytes, 4-byte aligned, stride 28
    assert_eq!(core::mem::size_of::<VertexPosColor>(), 28);
    assert_eq!(core::mem::align_of::<VertexPosColor>(), 4);
    assert_eq!(core::mem::offset_of!(VertexPosColor, position), 0);
    assert_eq!(core::mem::offset_of!(VertexPosColor, color), 12);
    assert_eq!(VERTEX_POS_COLOR_BYTES, 28);
    assert_eq!(VERTEX_POS_COLOR_STRIDE, 28);
    assert_eq!(VERTEX_POS_COLOR_ALIGNMENT, 4);
    assert_eq!(VertexPosColor::BYTE_SIZE, 28);
    assert_eq!(VertexPosColor::STRIDE, 28);
    assert_eq!(VertexPosColor::ALIGNMENT, 4);

    // 3. Multi-vertex buffer packing test for VertexPosNormalUv
    let pnu_vertices = [
        VertexPosNormalUv::new([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 1.0]),
        VertexPosNormalUv::new([-1.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0]),
        VertexPosNormalUv::new([1.0, -1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0]),
    ];
    let mut pnu_bytes = [0u8; 96];
    for (i, v) in pnu_vertices.iter().enumerate() {
        v.write_to_slice(&mut pnu_bytes[i * 32..(i + 1) * 32]).expect("write pnu vertex");
    }
    for (i, v) in pnu_vertices.iter().enumerate() {
        let restored = VertexPosNormalUv::read_from_slice(&pnu_bytes[i * 32..(i + 1) * 32])
            .expect("read pnu vertex");
        assert_eq!(restored, *v);
    }

    // 4. Multi-vertex buffer packing test for VertexPosColor
    let pc_vertices = [
        VertexPosColor::new([0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0]),
        VertexPosColor::new([-1.0, -1.0, 0.0], [0.0, 1.0, 0.0, 1.0]),
        VertexPosColor::new([1.0, -1.0, 0.0], [0.0, 0.0, 1.0, 1.0]),
    ];
    let mut pc_bytes = [0u8; 84];
    for (i, v) in pc_vertices.iter().enumerate() {
        v.write_to_slice(&mut pc_bytes[i * 28..(i + 1) * 28]).expect("write pc vertex");
    }
    for (i, v) in pc_vertices.iter().enumerate() {
        let restored = VertexPosColor::read_from_slice(&pc_bytes[i * 28..(i + 1) * 28])
            .expect("read pc vertex");
        assert_eq!(restored, *v);
    }
}

// ---------------------------------------------------------------------------
// Seeded deterministic property tests (f3d-05-ids-layouts-epochs-transport-vqa.3)
// ---------------------------------------------------------------------------

/// Minimal 64-bit Linear Congruential Generator (LCG) for deterministic property testing.
///
/// Multiplier and increment are standard constants from Knuth / MMIX.
/// Provides a zero-dependency, reproducible pseudo-random stream across platforms.
#[derive(Clone, Copy, Debug)]
struct TestLcg {
    state: u64,
}

impl TestLcg {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Generates a well-behaved finite f32 float in the range `[-2000.0, 2000.0]`.
    fn next_f32(&mut self) -> f32 {
        let u = self.next_u32();
        let normalized = (u as f64) / (u32::MAX as f64);
        ((normalized * 4000.0) - 2000.0) as f32
    }

    /// Generates a well-behaved finite f64 float in the range `[-20000.0, 20000.0]`.
    fn next_f64(&mut self) -> f64 {
        let u = self.next_u64();
        let normalized = (u as f64) / (u64::MAX as f64);
        (normalized * 40000.0) - 20000.0
    }
}

#[test]
fn property_test_affine_rows_byte_roundtrip_and_conversions() {
    const SEED: u64 = 0xA110_C47E_0001_0001;
    const ITERATIONS: usize = 5_000;
    let mut rng = TestLcg::new(SEED);

    for i in 0..ITERATIONS {
        let r0 = [rng.next_f32(), rng.next_f32(), rng.next_f32(), rng.next_f32()];
        let r1 = [rng.next_f32(), rng.next_f32(), rng.next_f32(), rng.next_f32()];
        let r2 = [rng.next_f32(), rng.next_f32(), rng.next_f32(), rng.next_f32()];

        let affine = AffineRows::new(r0, r1, r2);

        // 1. Array byte serialization roundtrip
        let bytes = affine.to_bytes();
        assert_eq!(
            bytes.len(),
            AFFINE_ROWS_BYTES,
            "Byte length mismatch for seed {SEED:#018x} at iter {i}"
        );
        let restored = AffineRows::from_bytes(&bytes);
        assert_eq!(
            restored, affine,
            "AffineRows::from_bytes roundtrip failed for seed {SEED:#018x} at iter {i}"
        );
        assert_eq!(
            restored.to_bytes(),
            bytes,
            "AffineRows restored bytes mismatch for seed {SEED:#018x} at iter {i}"
        );

        // 2. Slice serialization roundtrip
        let mut slice_buf = [0u8; 48];
        affine
            .write_to_slice(&mut slice_buf)
            .unwrap_or_else(|e| panic!("write_to_slice failed for seed {SEED:#018x} at iter {i}: {e:?}"));
        assert_eq!(
            slice_buf, bytes,
            "write_to_slice output mismatch for seed {SEED:#018x} at iter {i}"
        );
        let from_slice = AffineRows::read_from_slice(&slice_buf)
            .unwrap_or_else(|e| panic!("read_from_slice failed for seed {SEED:#018x} at iter {i}: {e:?}"));
        assert_eq!(
            from_slice, affine,
            "read_from_slice roundtrip failed for seed {SEED:#018x} at iter {i}"
        );

        // 3. Column-major 4x4 matrix expansion and reconstruction
        let col_major = affine.to_column_major();
        assert!(
            is_matrix4_affine(&col_major),
            "to_column_major must produce valid affine matrix for seed {SEED:#018x} at iter {i}"
        );
        // Verify mapping: row 0 [e0, e4, e8, e12], row 1 [e1, e5, e9, e13], row 2 [e2, e6, e10, e14]
        assert_eq!(col_major[0], r0[0], "col_major[0] mapping failed for seed {SEED:#018x} at iter {i}");
        assert_eq!(col_major[4], r0[1], "col_major[4] mapping failed for seed {SEED:#018x} at iter {i}");
        assert_eq!(col_major[8], r0[2], "col_major[8] mapping failed for seed {SEED:#018x} at iter {i}");
        assert_eq!(col_major[12], r0[3], "col_major[12] mapping failed for seed {SEED:#018x} at iter {i}");

        assert_eq!(col_major[1], r1[0], "col_major[1] mapping failed for seed {SEED:#018x} at iter {i}");
        assert_eq!(col_major[5], r1[1], "col_major[5] mapping failed for seed {SEED:#018x} at iter {i}");
        assert_eq!(col_major[9], r1[2], "col_major[9] mapping failed for seed {SEED:#018x} at iter {i}");
        assert_eq!(col_major[13], r1[3], "col_major[13] mapping failed for seed {SEED:#018x} at iter {i}");

        assert_eq!(col_major[2], r2[0], "col_major[2] mapping failed for seed {SEED:#018x} at iter {i}");
        assert_eq!(col_major[6], r2[1], "col_major[6] mapping failed for seed {SEED:#018x} at iter {i}");
        assert_eq!(col_major[10], r2[2], "col_major[10] mapping failed for seed {SEED:#018x} at iter {i}");
        assert_eq!(col_major[14], r2[3], "col_major[14] mapping failed for seed {SEED:#018x} at iter {i}");

        assert_eq!(col_major[3], 0.0, "col_major[3] must be 0 for seed {SEED:#018x} at iter {i}");
        assert_eq!(col_major[7], 0.0, "col_major[7] must be 0 for seed {SEED:#018x} at iter {i}");
        assert_eq!(col_major[11], 0.0, "col_major[11] must be 0 for seed {SEED:#018x} at iter {i}");
        assert_eq!(col_major[15], 1.0, "col_major[15] must be 1 for seed {SEED:#018x} at iter {i}");

        let reconstructed = AffineRows::from_column_major(&col_major)
            .unwrap_or_else(|e| panic!("from_column_major failed for seed {SEED:#018x} at iter {i}: {e:?}"));
        assert_eq!(
            reconstructed, affine,
            "reconstructed affine matrix mismatch for seed {SEED:#018x} at iter {i}"
        );
    }
}

#[test]
fn property_test_affine_rows_rejection_of_every_short_length() {
    const SEED: u64 = 0xB00F_FEE1_0002_0002;
    const ITERATIONS: usize = 5_000;
    let mut rng = TestLcg::new(SEED);

    let mut buf = [0u8; 48];

    for i in 0..ITERATIONS {
        let r0 = [rng.next_f32(), rng.next_f32(), rng.next_f32(), rng.next_f32()];
        let r1 = [rng.next_f32(), rng.next_f32(), rng.next_f32(), rng.next_f32()];
        let r2 = [rng.next_f32(), rng.next_f32(), rng.next_f32(), rng.next_f32()];
        let affine = AffineRows::new(r0, r1, r2);

        // Randomize buffer content
        for b in buf.iter_mut() {
            *b = (rng.next_u32() & 0xFF) as u8;
        }

        // Test EVERY short length strictly below AFFINE_ROWS_BYTES (48)
        for len in 0..AFFINE_ROWS_BYTES {
            let write_res = affine.write_to_slice(&mut buf[..len]);
            assert_eq!(
                write_res,
                Err(LayoutError::BufferTooSmall {
                    required: AFFINE_ROWS_BYTES,
                    provided: len,
                }),
                "write_to_slice must reject short length {len} for seed {SEED:#018x} at iter {i}"
            );

            let read_res = AffineRows::read_from_slice(&buf[..len]);
            assert_eq!(
                read_res,
                Err(LayoutError::BufferTooSmall {
                    required: AFFINE_ROWS_BYTES,
                    provided: len,
                }),
                "read_from_slice must reject short length {len} for seed {SEED:#018x} at iter {i}"
            );
        }
    }
}

#[test]
fn property_test_affine_rows_rejection_of_non_affine_rows() {
    const SEED: u64 = 0xC0DE_D00D_0003_0003;
    const ITERATIONS: usize = 5_000;
    let mut rng = TestLcg::new(SEED);

    for i in 0..ITERATIONS {
        // Base affine matrix in f32
        let mut elements = [0.0f32; 16];
        for k in 0..16 {
            elements[k] = rng.next_f32();
        }
        // Canonical affine row 3: [0, 0, 0, 1]
        elements[3] = 0.0;
        elements[7] = 0.0;
        elements[11] = 0.0;
        elements[15] = 1.0;

        // Base matrix must be recognized as affine
        assert!(
            is_matrix4_affine(&elements),
            "Base matrix must be affine for seed {SEED:#018x} at iter {i}"
        );
        assert!(
            AffineRows::from_column_major(&elements).is_ok(),
            "from_column_major must succeed on affine matrix for seed {SEED:#018x} at iter {i}"
        );

        // Perturbation target: 0 = e[3], 1 = e[7], 2 = e[11], 3 = e[15]
        let target = (rng.next_u32() % 4) as usize;
        let mut perturbed = elements;
        let stride = i % 8;
        match target {
            0 => {
                // Perturb e[3]
                perturbed[3] = match stride {
                    0 => f32::from_bits(1),           // subnormal float
                    1 => -5e-7,                       // negative small perspective
                    2 => f32::MAX,                    // max finite float
                    3 => f32::MIN_POSITIVE,           // minimum positive normal float
                    4 => -f32::MAX,                   // negative max finite float
                    5 => 5e-7,                        // analytical counterexample value
                    6 => 1e-15,                       // tiny float
                    _ => {
                        let v = rng.next_f32();
                        if v == 0.0 { 0.5 } else { v }
                    }
                };
            }
            1 => {
                // Perturb e[7]
                perturbed[7] = match stride {
                    0 => f32::from_bits(1),           // subnormal float
                    1 => -1e-15,                      // tiny negative float
                    2 => f32::MAX,                    // max finite float
                    3 => f32::MIN_POSITIVE,           // minimum positive normal float
                    4 => -f32::MAX,                   // negative max finite float
                    5 => 1e-15,                       // analytical tiny value
                    6 => 5e-7,                        // analytical counterexample value
                    _ => {
                        let v = rng.next_f32();
                        if v == 0.0 { -0.5 } else { v }
                    }
                };
            }
            2 => {
                // Perturb e[11]
                perturbed[11] = match stride {
                    0 => f32::from_bits(1),           // subnormal float
                    1 => f32::from_bits(0x8000_0001), // negative subnormal float
                    2 => f32::MAX,                    // max finite float
                    3 => f32::MIN_POSITIVE,           // minimum positive normal float
                    4 => -1.0,                        // standard perspective depth value
                    5 => 5e-7,                        // analytical counterexample value
                    6 => 1e-15,                       // tiny float
                    _ => {
                        let v = rng.next_f32();
                        if v == 0.0 { 2.0 } else { v }
                    }
                };
            }
            _ => {
                // Perturb e[15] (affine requires exactly 1.0)
                perturbed[15] = match stride {
                    0 => 0.0,                         // standard zero
                    1 => -0.0,                        // negative zero (-0.0 != 1.0, must reject)
                    2 => f32::MAX,                    // max finite float
                    3 => f32::from_bits(1),           // subnormal float
                    4 => -1.0,                        // negative one
                    5 => 1.0 + 1e-7,                  // near-one high
                    6 => 0.9999999,                   // near-one low
                    _ => {
                        let delta = rng.next_f32();
                        1.0 + if delta == 0.0 { 0.5 } else { delta }
                    }
                };
            }
        }

        assert!(
            !is_matrix4_affine(&perturbed),
            "is_matrix4_affine must reject non-affine row 3 for seed {SEED:#018x} at iter {i}"
        );
        assert_eq!(
            AffineRows::from_column_major(&perturbed),
            Err(LayoutError::NonAffineMatrix),
            "from_column_major must return NonAffineMatrix for seed {SEED:#018x} at iter {i}"
        );

        // Property test for f64 structural affine check
        let mut elements_f64 = [0.0f64; 16];
        for k in 0..16 {
            elements_f64[k] = rng.next_f64();
        }
        elements_f64[3] = 0.0;
        elements_f64[7] = 0.0;
        elements_f64[11] = 0.0;
        elements_f64[15] = 1.0;

        assert!(
            is_matrix4_f64_affine(&elements_f64),
            "Base f64 matrix must be affine for seed {SEED:#018x} at iter {i}"
        );
        assert!(
            AffineRows::from_column_major_f64(&elements_f64).is_ok(),
            "from_column_major_f64 must succeed on affine matrix for seed {SEED:#018x} at iter {i}"
        );

        let mut perturbed_f64 = elements_f64;
        match target {
            0 => {
                // Perturb e[3] in f64
                perturbed_f64[3] = match stride {
                    0 => f64::from_bits(1),           // subnormal f64
                    1 => -1e-15,                      // small analytical negative perspective
                    2 => f64::MAX,                    // max finite f64
                    3 => f64::MIN_POSITIVE,           // minimum positive normal f64
                    4 => -f64::MAX,                   // negative max finite f64
                    5 => 1e-15,                       // analytical tiny value
                    6 => 1e-30,                       // ultra-tiny value
                    _ => {
                        let v = rng.next_f64();
                        if v == 0.0 { 0.5 } else { v }
                    }
                };
            }
            1 => {
                // Perturb e[7] in f64
                perturbed_f64[7] = match stride {
                    0 => f64::from_bits(1),           // subnormal f64
                    1 => -1e-30,                      // tiny negative f64
                    2 => f64::MAX,                    // max finite f64
                    3 => f64::MIN_POSITIVE,           // minimum positive normal f64
                    4 => -f64::MAX,                   // negative max finite f64
                    5 => 1e-30,                       // analytical tiny value
                    6 => 1e-15,                       // analytical tiny value
                    _ => {
                        let v = rng.next_f64();
                        if v == 0.0 { -0.5 } else { v }
                    }
                };
            }
            2 => {
                // Perturb e[11] in f64
                perturbed_f64[11] = match stride {
                    0 => f64::from_bits(1),                           // subnormal f64
                    1 => f64::from_bits(0x8000_0000_0000_0001),       // negative subnormal f64
                    2 => f64::MAX,                                    // max finite f64
                    3 => f64::MIN_POSITIVE,                           // minimum positive normal f64
                    4 => -1.0,                                        // standard perspective depth value
                    5 => 1e-15,                                       // analytical tiny value
                    6 => 1e-100,                                      // small normal f64 perspective term
                    _ => {
                        let v = rng.next_f64();
                        if v == 0.0 { 2.0 } else { v }
                    }
                };
            }
            _ => {
                // Perturb e[15] in f64 (affine requires exactly 1.0)
                perturbed_f64[15] = match stride {
                    0 => 0.0,                         // standard zero
                    1 => -0.0,                        // negative zero (-0.0 != 1.0, must reject)
                    2 => f64::MAX,                    // max finite f64
                    3 => f64::from_bits(1),           // subnormal f64
                    4 => -1.0,                        // negative one
                    5 => 1.0 + 1e-15,                 // near-one high in f64
                    6 => 0.999999999999999,           // near-one low in f64
                    _ => {
                        let delta = rng.next_f64();
                        1.0 + if delta == 0.0 { 0.5 } else { delta }
                    }
                };
            }
        }

        assert!(
            !is_matrix4_f64_affine(&perturbed_f64),
            "is_matrix4_f64_affine must reject non-affine f64 row 3 for seed {SEED:#018x} at iter {i}"
        );
        assert_eq!(
            AffineRows::from_column_major_f64(&perturbed_f64),
            Err(LayoutError::NonAffineMatrix),
            "from_column_major_f64 must return NonAffineMatrix for seed {SEED:#018x} at iter {i}"
        );
    }
}

#[test]
fn property_test_projective_mat4_byte_roundtrip_and_short_lengths() {
    const SEED: u64 = 0xD15C_05E4_0004_0004;
    const ITERATIONS: usize = 5_000;
    let mut rng = TestLcg::new(SEED);

    let mut buf = [0u8; 64];

    for i in 0..ITERATIONS {
        let mut elements = [0.0f32; 16];
        for k in 0..16 {
            elements[k] = rng.next_f32();
        }
        let proj = ProjectiveMat4::from_elements(elements);

        // 1. Array byte serialization roundtrip
        let bytes = proj.to_bytes();
        assert_eq!(
            bytes.len(),
            PROJECTIVE_MAT4_BYTES,
            "Byte length mismatch for seed {SEED:#018x} at iter {i}"
        );
        let restored = ProjectiveMat4::from_bytes(&bytes);
        assert_eq!(
            restored, proj,
            "ProjectiveMat4::from_bytes roundtrip failed for seed {SEED:#018x} at iter {i}"
        );
        assert_eq!(
            restored.to_bytes(),
            bytes,
            "ProjectiveMat4 restored bytes mismatch for seed {SEED:#018x} at iter {i}"
        );

        // 2. Slice serialization roundtrip
        proj.write_to_slice(&mut buf)
            .unwrap_or_else(|e| panic!("write_to_slice failed for seed {SEED:#018x} at iter {i}: {e:?}"));
        assert_eq!(
            buf, bytes,
            "write_to_slice output mismatch for seed {SEED:#018x} at iter {i}"
        );
        let from_slice = ProjectiveMat4::read_from_slice(&buf)
            .unwrap_or_else(|e| panic!("read_from_slice failed for seed {SEED:#018x} at iter {i}: {e:?}"));
        assert_eq!(
            from_slice, proj,
            "read_from_slice roundtrip failed for seed {SEED:#018x} at iter {i}"
        );

        // 3. Rejection of every short length (0..64)
        for len in 0..PROJECTIVE_MAT4_BYTES {
            let write_res = proj.write_to_slice(&mut buf[..len]);
            assert_eq!(
                write_res,
                Err(LayoutError::BufferTooSmall {
                    required: PROJECTIVE_MAT4_BYTES,
                    provided: len,
                }),
                "write_to_slice must reject short length {len} for seed {SEED:#018x} at iter {i}"
            );

            let read_res = ProjectiveMat4::read_from_slice(&buf[..len]);
            assert_eq!(
                read_res,
                Err(LayoutError::BufferTooSmall {
                    required: PROJECTIVE_MAT4_BYTES,
                    provided: len,
                }),
                "read_from_slice must reject short length {len} for seed {SEED:#018x} at iter {i}"
            );
        }
    }
}

#[test]
fn property_test_vertex_pos_uv_byte_roundtrip_and_short_lengths() {
    const SEED: u64 = 0xE1E1_7E57_0005_0005;
    const ITERATIONS: usize = 5_000;
    let mut rng = TestLcg::new(SEED);

    let mut buf = [0u8; 20];

    for i in 0..ITERATIONS {
        let position = [rng.next_f32(), rng.next_f32(), rng.next_f32()];
        let uv = [rng.next_f32(), rng.next_f32()];
        let v = VertexPosUv::new(position, uv);

        // 1. Array byte serialization roundtrip
        let bytes = v.to_bytes();
        assert_eq!(
            bytes.len(),
            VERTEX_POS_UV_BYTES,
            "Byte length mismatch for seed {SEED:#018x} at iter {i}"
        );
        let restored = VertexPosUv::from_bytes(&bytes);
        assert_eq!(
            restored, v,
            "VertexPosUv::from_bytes roundtrip failed for seed {SEED:#018x} at iter {i}"
        );
        assert_eq!(
            restored.to_bytes(),
            bytes,
            "VertexPosUv restored bytes mismatch for seed {SEED:#018x} at iter {i}"
        );

        // 2. Slice serialization roundtrip
        v.write_to_slice(&mut buf)
            .unwrap_or_else(|e| panic!("write_to_slice failed for seed {SEED:#018x} at iter {i}: {e:?}"));
        assert_eq!(
            buf, bytes,
            "write_to_slice output mismatch for seed {SEED:#018x} at iter {i}"
        );
        let from_slice = VertexPosUv::read_from_slice(&buf)
            .unwrap_or_else(|e| panic!("read_from_slice failed for seed {SEED:#018x} at iter {i}: {e:?}"));
        assert_eq!(
            from_slice, v,
            "read_from_slice roundtrip failed for seed {SEED:#018x} at iter {i}"
        );

        // 3. Rejection of every short length (0..20)
        for len in 0..VERTEX_POS_UV_BYTES {
            let write_res = v.write_to_slice(&mut buf[..len]);
            assert_eq!(
                write_res,
                Err(LayoutError::BufferTooSmall {
                    required: VERTEX_POS_UV_BYTES,
                    provided: len,
                }),
                "write_to_slice must reject short length {len} for seed {SEED:#018x} at iter {i}"
            );

            let read_res = VertexPosUv::read_from_slice(&buf[..len]);
            assert_eq!(
                read_res,
                Err(LayoutError::BufferTooSmall {
                    required: VERTEX_POS_UV_BYTES,
                    provided: len,
                }),
                "read_from_slice must reject short length {len} for seed {SEED:#018x} at iter {i}"
            );
        }
    }
}

#[test]
fn property_test_vertex_pos_normal_uv_byte_roundtrip_and_short_lengths() {
    const SEED: u64 = 0xE1E1_7E57_0007_0007;
    const ITERATIONS: usize = 5_000;
    let mut rng = TestLcg::new(SEED);

    let mut buf = [0u8; 32];

    for i in 0..ITERATIONS {
        let mut position = [rng.next_f32(), rng.next_f32(), rng.next_f32()];
        let mut normal = [rng.next_f32(), rng.next_f32(), rng.next_f32()];
        let mut uv = [rng.next_f32(), rng.next_f32()];

        // Fixed stride perturbations injecting explicit NaN payloads, negative zero, and infinities
        let stride = i % 8;
        match stride {
            0 => {
                // Explicit quiet NaN with specific non-standard payload (0x7FC0_1234) on position[0]
                position[0] = f32::from_bits(0x7FC0_1234);
            }
            1 => {
                // Explicit negative zero (-0.0, bit pattern 0x8000_0000) on normal[1] and uv[0]
                normal[1] = -0.0f32;
                uv[0] = -0.0f32;
            }
            2 => {
                // Explicit positive infinity (0x7F80_0000) and negative infinity (0xFF80_0000)
                position[1] = f32::INFINITY;
                normal[0] = f32::NEG_INFINITY;
            }
            3 => {
                // Explicit negative NaN with distinct payload (0xFFC0_BEEF) on uv[1]
                uv[1] = f32::from_bits(0xFFC0_BEEF);
            }
            4 => {
                // Complete vector of negative zeros across all fields
                position = [-0.0f32, -0.0f32, -0.0f32];
                normal = [-0.0f32, -0.0f32, -0.0f32];
                uv = [-0.0f32, -0.0f32];
            }
            5 => {
                // Mixed infinities and signaling NaN payloads across attributes
                position[2] = f32::from_bits(0x7F80_0001); // signaling NaN payload
                normal[2] = f32::INFINITY;
                uv[1] = f32::NEG_INFINITY;
            }
            6 => {
                // Arbitrary PRNG-derived NaN payload
                let payload = (rng.next_u32() & 0x003F_FFFF) | 1;
                normal[0] = f32::from_bits(0x7FC0_0000 | payload);
            }
            _ => {
                // Default pseudo-random finite normal floats from TestLcg
            }
        }

        let v = VertexPosNormalUv::new(position, normal, uv);

        // 1. Array byte serialization roundtrip
        let bytes = v.to_bytes();
        assert_eq!(
            bytes.len(),
            VERTEX_POS_NORMAL_UV_BYTES,
            "Byte length mismatch for seed {SEED:#018x} at iter {i} stride {stride}"
        );
        let restored = VertexPosNormalUv::from_bytes(&bytes);
        assert_eq!(
            restored.to_bytes(),
            v.to_bytes(),
            "VertexPosNormalUv::from_bytes roundtrip failed for seed {SEED:#018x} at iter {i} stride {stride}"
        );
        assert_eq!(
            restored.to_bytes(),
            bytes,
            "VertexPosNormalUv restored bytes mismatch for seed {SEED:#018x} at iter {i} stride {stride}"
        );

        // Bit pattern assertions: verify exact IEEE 754 bit-pattern preservation for every field
        assert_eq!(
            restored.position[0].to_bits(),
            position[0].to_bits(),
            "position[0] bit mismatch for seed {SEED:#018x} at iter {i} stride {stride}"
        );
        assert_eq!(
            restored.position[1].to_bits(),
            position[1].to_bits(),
            "position[1] bit mismatch for seed {SEED:#018x} at iter {i} stride {stride}"
        );
        assert_eq!(
            restored.position[2].to_bits(),
            position[2].to_bits(),
            "position[2] bit mismatch for seed {SEED:#018x} at iter {i} stride {stride}"
        );
        assert_eq!(
            restored.normal[0].to_bits(),
            normal[0].to_bits(),
            "normal[0] bit mismatch for seed {SEED:#018x} at iter {i} stride {stride}"
        );
        assert_eq!(
            restored.normal[1].to_bits(),
            normal[1].to_bits(),
            "normal[1] bit mismatch for seed {SEED:#018x} at iter {i} stride {stride}"
        );
        assert_eq!(
            restored.normal[2].to_bits(),
            normal[2].to_bits(),
            "normal[2] bit mismatch for seed {SEED:#018x} at iter {i} stride {stride}"
        );
        assert_eq!(
            restored.uv[0].to_bits(),
            uv[0].to_bits(),
            "uv[0] bit mismatch for seed {SEED:#018x} at iter {i} stride {stride}"
        );
        assert_eq!(
            restored.uv[1].to_bits(),
            uv[1].to_bits(),
            "uv[1] bit mismatch for seed {SEED:#018x} at iter {i} stride {stride}"
        );

        // 2. Slice serialization roundtrip
        v.write_to_slice(&mut buf)
            .unwrap_or_else(|e| panic!("write_to_slice failed for seed {SEED:#018x} at iter {i} stride {stride}: {e:?}"));
        assert_eq!(
            buf, bytes,
            "write_to_slice output mismatch for seed {SEED:#018x} at iter {i} stride {stride}"
        );
        let from_slice = VertexPosNormalUv::read_from_slice(&buf)
            .unwrap_or_else(|e| panic!("read_from_slice failed for seed {SEED:#018x} at iter {i} stride {stride}: {e:?}"));
        assert_eq!(
            from_slice.to_bytes(),
            v.to_bytes(),
            "read_from_slice roundtrip failed for seed {SEED:#018x} at iter {i} stride {stride}"
        );

        // Bit pattern assertions on from_slice as well
        assert_eq!(
            from_slice.position[0].to_bits(),
            position[0].to_bits(),
            "from_slice position[0] bit mismatch for seed {SEED:#018x} at iter {i} stride {stride}"
        );
        assert_eq!(
            from_slice.position[1].to_bits(),
            position[1].to_bits(),
            "from_slice position[1] bit mismatch for seed {SEED:#018x} at iter {i} stride {stride}"
        );
        assert_eq!(
            from_slice.position[2].to_bits(),
            position[2].to_bits(),
            "from_slice position[2] bit mismatch for seed {SEED:#018x} at iter {i} stride {stride}"
        );
        assert_eq!(
            from_slice.normal[0].to_bits(),
            normal[0].to_bits(),
            "from_slice normal[0] bit mismatch for seed {SEED:#018x} at iter {i} stride {stride}"
        );
        assert_eq!(
            from_slice.normal[1].to_bits(),
            normal[1].to_bits(),
            "from_slice normal[1] bit mismatch for seed {SEED:#018x} at iter {i} stride {stride}"
        );
        assert_eq!(
            from_slice.normal[2].to_bits(),
            normal[2].to_bits(),
            "from_slice normal[2] bit mismatch for seed {SEED:#018x} at iter {i} stride {stride}"
        );
        assert_eq!(
            from_slice.uv[0].to_bits(),
            uv[0].to_bits(),
            "from_slice uv[0] bit mismatch for seed {SEED:#018x} at iter {i} stride {stride}"
        );
        assert_eq!(
            from_slice.uv[1].to_bits(),
            uv[1].to_bits(),
            "from_slice uv[1] bit mismatch for seed {SEED:#018x} at iter {i} stride {stride}"
        );

        // 3. Rejection of every short length (0..32)
        for len in 0..VERTEX_POS_NORMAL_UV_BYTES {
            let write_res = v.write_to_slice(&mut buf[..len]);
            assert_eq!(
                write_res,
                Err(LayoutError::BufferTooSmall {
                    required: VERTEX_POS_NORMAL_UV_BYTES,
                    provided: len,
                }),
                "write_to_slice must reject short length {len} for seed {SEED:#018x} at iter {i}"
            );

            let read_res = VertexPosNormalUv::read_from_slice(&buf[..len]);
            assert_eq!(
                read_res,
                Err(LayoutError::BufferTooSmall {
                    required: VERTEX_POS_NORMAL_UV_BYTES,
                    provided: len,
                }),
                "read_from_slice must reject short length {len} for seed {SEED:#018x} at iter {i}"
            );
        }
    }
}

#[test]
fn property_test_vertex_pos_color_byte_roundtrip_and_short_lengths() {
    const SEED: u64 = 0xE1E1_7E57_0008_0008;
    const ITERATIONS: usize = 5_000;
    let mut rng = TestLcg::new(SEED);

    let mut buf = [0u8; 28];

    for i in 0..ITERATIONS {
        let mut position = [rng.next_f32(), rng.next_f32(), rng.next_f32()];
        let mut color = [rng.next_f32(), rng.next_f32(), rng.next_f32(), rng.next_f32()];

        // Fixed stride perturbations injecting explicit NaN payloads, negative zero, and infinities
        let stride = i % 8;
        match stride {
            0 => {
                // Explicit quiet NaN with specific non-standard payload (0x7FC0_ABCD) on color[3] (alpha)
                color[3] = f32::from_bits(0x7FC0_ABCD);
            }
            1 => {
                // Explicit negative zero (-0.0, bit pattern 0x8000_0000) on position[0] and color[1]
                position[0] = -0.0f32;
                color[1] = -0.0f32;
            }
            2 => {
                // Explicit positive infinity (0x7F80_0000) and negative infinity (0xFF80_0000)
                position[1] = f32::INFINITY;
                color[0] = f32::NEG_INFINITY;
            }
            3 => {
                // Explicit negative NaN with distinct payload (0xFFC0_DEAD) on position[2]
                position[2] = f32::from_bits(0xFFC0_DEAD);
            }
            4 => {
                // Complete vector of negative zeros across all fields
                position = [-0.0f32, -0.0f32, -0.0f32];
                color = [-0.0f32, -0.0f32, -0.0f32, -0.0f32];
            }
            5 => {
                // Mixed infinities and signaling NaN payloads across attributes
                position[0] = f32::from_bits(0x7F80_0001); // signaling NaN payload
                color[2] = f32::INFINITY;
                color[3] = f32::NEG_INFINITY;
            }
            6 => {
                // Arbitrary PRNG-derived NaN payload on color[0]
                let payload = (rng.next_u32() & 0x003F_FFFF) | 1;
                color[0] = f32::from_bits(0x7FC0_0000 | payload);
            }
            _ => {
                // Default pseudo-random finite normal floats from TestLcg
            }
        }

        let v = VertexPosColor::new(position, color);

        // 1. Array byte serialization roundtrip
        let bytes = v.to_bytes();
        assert_eq!(
            bytes.len(),
            VERTEX_POS_COLOR_BYTES,
            "Byte length mismatch for seed {SEED:#018x} at iter {i} stride {stride}"
        );
        let restored = VertexPosColor::from_bytes(&bytes);
        assert_eq!(
            restored.to_bytes(),
            v.to_bytes(),
            "VertexPosColor::from_bytes roundtrip failed for seed {SEED:#018x} at iter {i} stride {stride}"
        );
        assert_eq!(
            restored.to_bytes(),
            bytes,
            "VertexPosColor restored bytes mismatch for seed {SEED:#018x} at iter {i} stride {stride}"
        );

        // Bit pattern assertions: verify exact IEEE 754 bit-pattern preservation for every field
        assert_eq!(
            restored.position[0].to_bits(),
            position[0].to_bits(),
            "position[0] bit mismatch for seed {SEED:#018x} at iter {i} stride {stride}"
        );
        assert_eq!(
            restored.position[1].to_bits(),
            position[1].to_bits(),
            "position[1] bit mismatch for seed {SEED:#018x} at iter {i} stride {stride}"
        );
        assert_eq!(
            restored.position[2].to_bits(),
            position[2].to_bits(),
            "position[2] bit mismatch for seed {SEED:#018x} at iter {i} stride {stride}"
        );
        assert_eq!(
            restored.color[0].to_bits(),
            color[0].to_bits(),
            "color[0] bit mismatch for seed {SEED:#018x} at iter {i} stride {stride}"
        );
        assert_eq!(
            restored.color[1].to_bits(),
            color[1].to_bits(),
            "color[1] bit mismatch for seed {SEED:#018x} at iter {i} stride {stride}"
        );
        assert_eq!(
            restored.color[2].to_bits(),
            color[2].to_bits(),
            "color[2] bit mismatch for seed {SEED:#018x} at iter {i} stride {stride}"
        );
        assert_eq!(
            restored.color[3].to_bits(),
            color[3].to_bits(),
            "color[3] bit mismatch for seed {SEED:#018x} at iter {i} stride {stride}"
        );

        // 2. Slice serialization roundtrip
        v.write_to_slice(&mut buf)
            .unwrap_or_else(|e| panic!("write_to_slice failed for seed {SEED:#018x} at iter {i} stride {stride}: {e:?}"));
        assert_eq!(
            buf, bytes,
            "write_to_slice output mismatch for seed {SEED:#018x} at iter {i} stride {stride}"
        );
        let from_slice = VertexPosColor::read_from_slice(&buf)
            .unwrap_or_else(|e| panic!("read_from_slice failed for seed {SEED:#018x} at iter {i} stride {stride}: {e:?}"));
        assert_eq!(
            from_slice.to_bytes(),
            v.to_bytes(),
            "read_from_slice roundtrip failed for seed {SEED:#018x} at iter {i} stride {stride}"
        );

        // Bit pattern assertions on from_slice as well
        assert_eq!(
            from_slice.position[0].to_bits(),
            position[0].to_bits(),
            "from_slice position[0] bit mismatch for seed {SEED:#018x} at iter {i} stride {stride}"
        );
        assert_eq!(
            from_slice.position[1].to_bits(),
            position[1].to_bits(),
            "from_slice position[1] bit mismatch for seed {SEED:#018x} at iter {i} stride {stride}"
        );
        assert_eq!(
            from_slice.position[2].to_bits(),
            position[2].to_bits(),
            "from_slice position[2] bit mismatch for seed {SEED:#018x} at iter {i} stride {stride}"
        );
        assert_eq!(
            from_slice.color[0].to_bits(),
            color[0].to_bits(),
            "from_slice color[0] bit mismatch for seed {SEED:#018x} at iter {i} stride {stride}"
        );
        assert_eq!(
            from_slice.color[1].to_bits(),
            color[1].to_bits(),
            "from_slice color[1] bit mismatch for seed {SEED:#018x} at iter {i} stride {stride}"
        );
        assert_eq!(
            from_slice.color[2].to_bits(),
            color[2].to_bits(),
            "from_slice color[2] bit mismatch for seed {SEED:#018x} at iter {i} stride {stride}"
        );
        assert_eq!(
            from_slice.color[3].to_bits(),
            color[3].to_bits(),
            "from_slice color[3] bit mismatch for seed {SEED:#018x} at iter {i} stride {stride}"
        );

        // 3. Rejection of every short length (0..28)
        for len in 0..VERTEX_POS_COLOR_BYTES {
            let write_res = v.write_to_slice(&mut buf[..len]);
            assert_eq!(
                write_res,
                Err(LayoutError::BufferTooSmall {
                    required: VERTEX_POS_COLOR_BYTES,
                    provided: len,
                }),
                "write_to_slice must reject short length {len} for seed {SEED:#018x} at iter {i}"
            );

            let read_res = VertexPosColor::read_from_slice(&buf[..len]);
            assert_eq!(
                read_res,
                Err(LayoutError::BufferTooSmall {
                    required: VERTEX_POS_COLOR_BYTES,
                    provided: len,
                }),
                "read_from_slice must reject short length {len} for seed {SEED:#018x} at iter {i}"
            );
        }
    }
}

#[test]
fn property_test_u32_word_abi_epoch_and_data_version() {
    const SEED: u64 = 0xF00D_BEEF_0006_0006;
    const ITERATIONS: usize = 5_000;
    let mut rng = TestLcg::new(SEED);

    // Explicit edge cases tested first
    let edge_cases = [
        0u64,
        1u64,
        (u32::MAX as u64) - 1,
        u32::MAX as u64,
        (u32::MAX as u64) + 1,
        0x0000_0001_0000_0000u64,
        0x7FFF_FFFF_FFFF_FFFFu64,
        0x8000_0000_0000_0000u64,
        0xFFFF_FFFF_0000_0000u64,
        0x0000_0000_FFFF_FFFFu64,
        u64::MAX - 1,
        u64::MAX,
    ];

    for (k, &val) in edge_cases.iter().enumerate() {
        // Epoch
        let epoch = Epoch::new(val);
        let (high, low) = epoch.to_words();
        assert_eq!(high, (val >> 32) as u32, "Epoch edge case {k} high word mismatch");
        assert_eq!(low, val as u32, "Epoch edge case {k} low word mismatch");
        assert_eq!(epoch.high_u32(), high, "Epoch edge case {k} high_u32() mismatch");
        assert_eq!(epoch.low_u32(), low, "Epoch edge case {k} low_u32() mismatch");
        let restored_epoch = Epoch::from_words(high, low);
        assert_eq!(restored_epoch, epoch, "Epoch edge case {k} from_words roundtrip mismatch");
        assert_eq!(restored_epoch.get(), val, "Epoch edge case {k} get() mismatch");

        // DataVersion
        let dv = DataVersion::new(val);
        let (dv_high, dv_low) = dv.to_words();
        assert_eq!(dv_high, (val >> 32) as u32, "DataVersion edge case {k} high word mismatch");
        assert_eq!(dv_low, val as u32, "DataVersion edge case {k} low word mismatch");
        assert_eq!(dv.high_u32(), dv_high, "DataVersion edge case {k} high_u32() mismatch");
        assert_eq!(dv.low_u32(), dv_low, "DataVersion edge case {k} low_u32() mismatch");
        let restored_dv = DataVersion::from_words(dv_high, dv_low);
        assert_eq!(restored_dv, dv, "DataVersion edge case {k} from_words roundtrip mismatch");
        assert_eq!(restored_dv.get(), val, "DataVersion edge case {k} get() mismatch");
    }

    // Randomized cases
    for i in 0..ITERATIONS {
        let val = rng.next_u64();

        // 1. Epoch word roundtrip
        let epoch = Epoch::new(val);
        let (high, low) = epoch.to_words();
        assert_eq!(
            high,
            (val >> 32) as u32,
            "Epoch high word mismatch for val {val:#018x}, seed {SEED:#018x} at iter {i}"
        );
        assert_eq!(
            low,
            val as u32,
            "Epoch low word mismatch for val {val:#018x}, seed {SEED:#018x} at iter {i}"
        );
        assert_eq!(
            epoch.high_u32(),
            high,
            "Epoch high_u32() mismatch for val {val:#018x}, seed {SEED:#018x} at iter {i}"
        );
        assert_eq!(
            epoch.low_u32(),
            low,
            "Epoch low_u32() mismatch for val {val:#018x}, seed {SEED:#018x} at iter {i}"
        );
        let restored_epoch = Epoch::from_words(high, low);
        assert_eq!(
            restored_epoch,
            epoch,
            "Epoch from_words roundtrip mismatch for val {val:#018x}, seed {SEED:#018x} at iter {i}"
        );
        assert_eq!(
            restored_epoch.get(),
            val,
            "Epoch get() mismatch for val {val:#018x}, seed {SEED:#018x} at iter {i}"
        );

        // 2. DataVersion word roundtrip
        let dv = DataVersion::new(val);
        let (dv_high, dv_low) = dv.to_words();
        assert_eq!(
            dv_high,
            (val >> 32) as u32,
            "DataVersion high word mismatch for val {val:#018x}, seed {SEED:#018x} at iter {i}"
        );
        assert_eq!(
            dv_low,
            val as u32,
            "DataVersion low word mismatch for val {val:#018x}, seed {SEED:#018x} at iter {i}"
        );
        assert_eq!(
            dv.high_u32(),
            dv_high,
            "DataVersion high_u32() mismatch for val {val:#018x}, seed {SEED:#018x} at iter {i}"
        );
        assert_eq!(
            dv.low_u32(),
            dv_low,
            "DataVersion low_u32() mismatch for val {val:#018x}, seed {SEED:#018x} at iter {i}"
        );
        let restored_dv = DataVersion::from_words(dv_high, dv_low);
        assert_eq!(
            restored_dv,
            dv,
            "DataVersion from_words roundtrip mismatch for val {val:#018x}, seed {SEED:#018x} at iter {i}"
        );
        assert_eq!(
            restored_dv.get(),
            val,
            "DataVersion get() mismatch for val {val:#018x}, seed {SEED:#018x} at iter {i}"
        );
    }
}

#[test]
fn property_test_alignment_validators_on_random_offsets() {
    const SEED: u64 = 0x1234_5678_0007_0007;
    const ITERATIONS: usize = 5_000;
    let mut rng = TestLcg::new(SEED);

    let allowed_alignments = [256usize, 512, 1024, 2048, 4096];

    for i in 0..ITERATIONS {
        let align_idx = (rng.next_u32() as usize) % allowed_alignments.len();
        let align = allowed_alignments[align_idx];
        let k = (rng.next_u32() as usize) % 50_000;
        let aligned_offset = k * align;

        // Dynamic uniform and storage offset: aligned succeeds
        assert!(
            validate_dynamic_uniform_offset(aligned_offset, align).is_ok(),
            "Aligned dynamic uniform offset {aligned_offset} (align {align}) failed for seed {SEED:#018x} at iter {i}"
        );
        assert!(
            validate_dynamic_storage_offset(aligned_offset, align).is_ok(),
            "Aligned dynamic storage offset {aligned_offset} (align {align}) failed for seed {SEED:#018x} at iter {i}"
        );

        // Dynamic uniform and storage offset: unaligned fails
        let rem = (rng.next_u32() as usize % (align - 1)) + 1; // 1 <= rem < align
        let unaligned_offset = aligned_offset + rem;
        assert_eq!(
            validate_dynamic_uniform_offset(unaligned_offset, align),
            Err(LayoutError::UnalignedOffset {
                offset: unaligned_offset,
                required_alignment: align,
            }),
            "Unaligned dynamic uniform offset {unaligned_offset} (align {align}) must fail for seed {SEED:#018x} at iter {i}"
        );
        assert_eq!(
            validate_dynamic_storage_offset(unaligned_offset, align),
            Err(LayoutError::UnalignedOffset {
                offset: unaligned_offset,
                required_alignment: align,
            }),
            "Unaligned dynamic storage offset {unaligned_offset} (align {align}) must fail for seed {SEED:#018x} at iter {i}"
        );

        // writeBuffer 4-byte alignment
        let k_off = (rng.next_u32() as usize) % 50_000;
        let k_sz = (rng.next_u32() as usize) % 50_000;
        let aligned_off = k_off * 4;
        let aligned_sz = k_sz * 4;
        assert!(
            validate_write_buffer_alignment(aligned_off, aligned_sz).is_ok(),
            "Aligned writeBuffer offset {aligned_off}, size {aligned_sz} failed for seed {SEED:#018x} at iter {i}"
        );

        let rem_off = (rng.next_u32() as usize % 3) + 1; // 1, 2, 3
        let unaligned_off = aligned_off + rem_off;
        assert_eq!(
            validate_write_buffer_alignment(unaligned_off, aligned_sz),
            Err(LayoutError::UnalignedWriteBuffer { value: unaligned_off }),
            "Unaligned writeBuffer offset {unaligned_off} must fail for seed {SEED:#018x} at iter {i}"
        );

        let rem_sz = (rng.next_u32() as usize % 3) + 1; // 1, 2, 3
        let unaligned_sz = aligned_sz + rem_sz;
        assert_eq!(
            validate_write_buffer_alignment(aligned_off, unaligned_sz),
            Err(LayoutError::UnalignedWriteBuffer { value: unaligned_sz }),
            "Unaligned writeBuffer size {unaligned_sz} must fail for seed {SEED:#018x} at iter {i}"
        );

        // Texture copy bytesPerRow 256-byte alignment
        let k_bpr = rng.next_u32() % 10_000;
        let aligned_bpr = k_bpr * 256;
        assert!(
            validate_copy_bytes_per_row(aligned_bpr).is_ok(),
            "Aligned bytesPerRow {aligned_bpr} failed for seed {SEED:#018x} at iter {i}"
        );

        let rem_bpr = (rng.next_u32() % 255) + 1; // 1 <= rem_bpr < 256
        let unaligned_bpr = aligned_bpr + rem_bpr;
        assert_eq!(
            validate_copy_bytes_per_row(unaligned_bpr),
            Err(LayoutError::UnalignedBytesPerRow {
                bytes_per_row: unaligned_bpr,
                required_alignment: 256,
            }),
            "Unaligned bytesPerRow {unaligned_bpr} must fail for seed {SEED:#018x} at iter {i}"
        );

        // Checked row pitch calculation aligned_bytes_per_row(width)
        let safe_width = rng.next_u32() % (u32::MAX / 4 - 256);
        let calculated_pitch = aligned_bytes_per_row(safe_width)
            .unwrap_or_else(|e| panic!("aligned_bytes_per_row failed for safe width {safe_width}, seed {SEED:#018x} at iter {i}: {e:?}"));
        let unpadded_bytes = safe_width * 4;
        assert!(
            calculated_pitch >= unpadded_bytes,
            "calculated_pitch {calculated_pitch} < unpadded_bytes {unpadded_bytes} for seed {SEED:#018x} at iter {i}"
        );
        assert_eq!(
            calculated_pitch % 256,
            0,
            "calculated_pitch {calculated_pitch} must be multiple of 256 for seed {SEED:#018x} at iter {i}"
        );
        assert!(
            calculated_pitch < unpadded_bytes + 256,
            "calculated_pitch {calculated_pitch} exceeds minimum padding for seed {SEED:#018x} at iter {i}"
        );
        assert!(
            validate_copy_bytes_per_row(calculated_pitch).is_ok(),
            "calculated_pitch {calculated_pitch} must validate against copy_bytes_per_row for seed {SEED:#018x} at iter {i}"
        );

        // Calculation overflow on invalid width
        let overflow_width = (u32::MAX / 4) + 1 + (rng.next_u32() % 100_000);
        assert_eq!(
            aligned_bytes_per_row(overflow_width),
            Err(LayoutError::CalculationOverflow),
            "aligned_bytes_per_row must return CalculationOverflow for {overflow_width}, seed {SEED:#018x} at iter {i}"
        );
    }
}

#[test]
fn test_layout_table_cross_check_and_evidence() {
    let table = layout_table();
    assert_eq!(table.len(), 31, "Layout table must contain all 31 defined field and padding rows");

    for row in table {
        let (expected_offset, expected_size) = match (row.record, row.field) {
            ("AffineRows", "r0") => (core::mem::offset_of!(AffineRows, r0), core::mem::size_of::<[f32; 4]>()),
            ("AffineRows", "r1") => (core::mem::offset_of!(AffineRows, r1), core::mem::size_of::<[f32; 4]>()),
            ("AffineRows", "r2") => (core::mem::offset_of!(AffineRows, r2), core::mem::size_of::<[f32; 4]>()),
            ("ProjectiveMat4", "elements") => (core::mem::offset_of!(ProjectiveMat4, elements), core::mem::size_of::<[f32; 16]>()),
            ("VertexPosUv", "position") => (core::mem::offset_of!(VertexPosUv, position), core::mem::size_of::<[f32; 3]>()),
            ("VertexPosUv", "uv") => (core::mem::offset_of!(VertexPosUv, uv), core::mem::size_of::<[f32; 2]>()),
            ("VertexPosNormalUv", "position") => (core::mem::offset_of!(VertexPosNormalUv, position), core::mem::size_of::<[f32; 3]>()),
            ("VertexPosNormalUv", "normal") => (core::mem::offset_of!(VertexPosNormalUv, normal), core::mem::size_of::<[f32; 3]>()),
            ("VertexPosNormalUv", "uv") => (core::mem::offset_of!(VertexPosNormalUv, uv), core::mem::size_of::<[f32; 2]>()),
            ("VertexPosColor", "position") => (core::mem::offset_of!(VertexPosColor, position), core::mem::size_of::<[f32; 3]>()),
            ("VertexPosColor", "color") => (core::mem::offset_of!(VertexPosColor, color), core::mem::size_of::<[f32; 4]>()),
            ("InstanceRecord", "transform") => (core::mem::offset_of!(InstanceRecord, transform), core::mem::size_of::<AffineRows>()),
            ("InstanceRecord", "instance_id") => (core::mem::offset_of!(InstanceRecord, instance_id), core::mem::size_of::<u32>()),
            ("InstanceRecord", "_padding") => (52, 12),
            ("DrawIndirectArgs", "vertex_count") => (core::mem::offset_of!(DrawIndirectArgs, vertex_count), core::mem::size_of::<u32>()),
            ("DrawIndirectArgs", "instance_count") => (core::mem::offset_of!(DrawIndirectArgs, instance_count), core::mem::size_of::<u32>()),
            ("DrawIndirectArgs", "first_vertex") => (core::mem::offset_of!(DrawIndirectArgs, first_vertex), core::mem::size_of::<u32>()),
            ("DrawIndirectArgs", "first_instance") => (core::mem::offset_of!(DrawIndirectArgs, first_instance), core::mem::size_of::<u32>()),
            ("DrawIndexedIndirectArgs", "index_count") => (core::mem::offset_of!(DrawIndexedIndirectArgs, index_count), core::mem::size_of::<u32>()),
            ("DrawIndexedIndirectArgs", "instance_count") => (core::mem::offset_of!(DrawIndexedIndirectArgs, instance_count), core::mem::size_of::<u32>()),
            ("DrawIndexedIndirectArgs", "first_index") => (core::mem::offset_of!(DrawIndexedIndirectArgs, first_index), core::mem::size_of::<u32>()),
            ("DrawIndexedIndirectArgs", "base_vertex") => (core::mem::offset_of!(DrawIndexedIndirectArgs, base_vertex), core::mem::size_of::<i32>()),
            ("DrawIndexedIndirectArgs", "first_instance") => (core::mem::offset_of!(DrawIndexedIndirectArgs, first_instance), core::mem::size_of::<u32>()),
            ("ColorUniform", "rgba") => (0, COLOR_UNIFORM_BYTES),
            ("MaterialParams", "color") => (core::mem::offset_of!(MaterialParams, color), core::mem::size_of::<[f32; 4]>()),
            ("MaterialParams", "opacity") => (core::mem::offset_of!(MaterialParams, opacity), core::mem::size_of::<f32>()),
            ("MaterialParams", "alpha_test") => (core::mem::offset_of!(MaterialParams, alpha_test), core::mem::size_of::<f32>()),
            ("MaterialParams", "_pad0") => (core::mem::offset_of!(MaterialParams, _pad0), core::mem::size_of::<[u8; 8]>()),
            ("MaterialParams", "map_transform") => (core::mem::offset_of!(MaterialParams, map_transform), core::mem::size_of::<AffineRows>()),
            ("MaterialParams", "flags") => (core::mem::offset_of!(MaterialParams, flags), core::mem::size_of::<u32>()),
            ("MaterialParams", "_pad1") => (core::mem::offset_of!(MaterialParams, _pad1), core::mem::size_of::<[u8; 12]>()),
            (unknown_rec, unknown_field) => panic!("Unknown record/field in layout table: {unknown_rec}.{unknown_field}"),
        };

        assert_eq!(
            row.offset, expected_offset,
            "LayoutRow offset mismatch for field {}.{}: expected offset {}, actual offset {}",
            row.record, row.field, expected_offset, row.offset
        );
        assert_eq!(
            row.size, expected_size,
            "LayoutRow size mismatch for field {}.{}: expected size {}, actual size {}",
            row.record, row.field, expected_size, row.size
        );
    }

    // Tiling invariant: for each record type, the catalog rows must tile the full size_of with no gaps or overlaps.
    let record_sizes: &[(&str, usize)] = &[
        ("AffineRows", core::mem::size_of::<AffineRows>()),
        ("ProjectiveMat4", core::mem::size_of::<ProjectiveMat4>()),
        ("VertexPosUv", core::mem::size_of::<VertexPosUv>()),
        ("VertexPosNormalUv", core::mem::size_of::<VertexPosNormalUv>()),
        ("VertexPosColor", core::mem::size_of::<VertexPosColor>()),
        ("InstanceRecord", InstanceRecord::BYTE_SIZE),
        ("DrawIndirectArgs", core::mem::size_of::<DrawIndirectArgs>()),
        ("DrawIndexedIndirectArgs", core::mem::size_of::<DrawIndexedIndirectArgs>()),
        ("ColorUniform", COLOR_UNIFORM_BYTES),
        ("MaterialParams", core::mem::size_of::<MaterialParams>()),
    ];

    for &(record_name, expected_total_size) in record_sizes {
        let mut expected_next_offset = 0;
        let mut found_any = false;
        for row in table {
            if row.record == record_name {
                found_any = true;
                assert_eq!(
                    row.offset, expected_next_offset,
                    "Gap or overlap detected in record {}: field {} has offset {}, expected {}",
                    record_name, row.field, row.offset, expected_next_offset
                );
                expected_next_offset = row.offset + row.size;
            }
        }
        assert!(found_any, "No rows found for record {}", record_name);
        assert_eq!(
            expected_next_offset, expected_total_size,
            "Record {} rows do not tile the full size_of: tiled {} bytes, size_of is {}",
            record_name, expected_next_offset, expected_total_size
        );
    }

    // Verify Display formatting
    let table_view = dump_layouts();
    let display_str = format!("{table_view}");
    assert!(display_str.contains("RECORD"), "Display must contain column header RECORD");
    assert!(display_str.contains("WGSL_TYPE"), "Display must contain column header WGSL_TYPE");
    assert!(display_str.contains("AffineRows"), "Display must render AffineRows");
    assert!(display_str.contains("VertexPosColor"), "Display must render VertexPosColor");
    assert!(display_str.contains("ColorUniform"), "Display must render ColorUniform");
    assert!(display_str.contains("MaterialParams"), "Display must render MaterialParams");
    assert!(display_str.contains("_padding"), "Display must render _padding row");

    // When test-support feature is active and F3D_EVIDENCE_DIR is set, emit evidence under bead 05.2
    #[cfg(feature = "test-support")]
    {
        if let Ok(evidence_dir) = std::env::var("F3D_EVIDENCE_DIR") {
            let base = std::path::Path::new(&evidence_dir);
            let writer = f3d_core::test_evidence::EvidenceWriter::init(base, "05.2", "layout_table")
                .expect("Failed to initialize EvidenceWriter under 05.2");
            writer
                .write_json("layout_catalog.json", &table)
                .expect("Failed to write layout_catalog.json");
            writer
                .write_artifact("layout_table.txt", display_str.as_bytes())
                .expect("Failed to write layout_table.txt");
            let event = f3d_core::test_evidence::EvidenceEvent {
                ts_wall: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0),
                ts_app: None,
                lane: "unit".into(),
                bead: "05.2".into(),
                test: "test_layout_table_cross_check_and_evidence".into(),
                step: "layout_table_cross_check".into(),
                level: "info".into(),
                owner: "f3d-core".into(),
                route: None,
                browser: None,
                device_generation: None,
                scene_generation: None,
                msg: "Verified all 31 layout table rows against core::mem::offset_of, size_of, and gapless tiling".into(),
                data: Some(serde_json::to_value(table).expect("Failed to serialize layout table")),
            };
            writer.write_event(&event).expect("Failed to write evidence event");
            let summary = f3d_core::test_evidence::EvidenceSummary {
                commit: "HEAD".into(),
                upstream_commit: f3d_core::test_evidence::UPSTREAM_COMMIT.into(),
                bead: "05.2".into(),
                run_id: "layout_table".into(),
                pass: table.len(),
                fail: 0,
                first_failure: None,
            };
            writer.write_summary(&summary).expect("Failed to write evidence summary");
        }
    }
}

#[test]
fn test_material_params_exact_layout_and_roundtrip() {
    assert_eq!(MATERIAL_PARAMS_BYTES, 96);
    assert_eq!(MATERIAL_PARAMS_ALIGNMENT, 16);
    assert_eq!(core::mem::size_of::<MaterialParams>(), 96);
    assert_eq!(core::mem::offset_of!(MaterialParams, color), 0);
    assert_eq!(core::mem::offset_of!(MaterialParams, opacity), 16);
    assert_eq!(core::mem::offset_of!(MaterialParams, alpha_test), 20);
    assert_eq!(core::mem::offset_of!(MaterialParams, _pad0), 24);
    assert_eq!(core::mem::offset_of!(MaterialParams, map_transform), 32);
    assert_eq!(core::mem::offset_of!(MaterialParams, flags), 80);
    assert_eq!(core::mem::offset_of!(MaterialParams, _pad1), 84);

    let mut mat = MaterialParams::new(
        [1.0, 0.5, 0.25, 1.0],
        0.75,
        0.01,
        AffineRows::identity(),
        MATERIAL_FLAG_MAP | MATERIAL_FLAG_ALPHA_MAP | MATERIAL_FLAG_TRANSPARENT,
    );
    assert!(mat.has_flag(MATERIAL_FLAG_MAP));
    assert!(mat.has_flag(MATERIAL_FLAG_ALPHA_MAP));
    assert!(mat.has_flag(MATERIAL_FLAG_TRANSPARENT));
    assert!(!mat.has_flag(MATERIAL_FLAG_WIREFRAME));
    mat.set_flag(MATERIAL_FLAG_WIREFRAME, true);
    assert!(mat.has_flag(MATERIAL_FLAG_WIREFRAME));

    let bytes = mat.to_bytes();
    assert_eq!(bytes.len(), 96);
    let decoded = MaterialParams::from_bytes(&bytes);
    assert_eq!(decoded, mat);

    let mut buf = [0u8; 96];
    mat.write_to_slice(&mut buf).expect("write ok");
    let from_slice = MaterialParams::read_from_slice(&buf).expect("read ok");
    assert_eq!(from_slice, mat);

    // Buffer too small errors
    let mut short_buf = [0u8; 95];
    assert_eq!(
        mat.write_to_slice(&mut short_buf),
        Err(LayoutError::BufferTooSmall { required: 96, provided: 95 })
    );
    assert_eq!(
        MaterialParams::read_from_slice(&short_buf),
        Err(LayoutError::BufferTooSmall { required: 96, provided: 95 })
    );
}
