//! Unit tests for `Color` and `ColorManagement` matching Three.js r186.
//!
//! Validates:
//! 1. sRGB $\leftrightarrow$ Linear piecewise formula exactness at boundary thresholds (0.04045 and 0.0031308).
//! 2. Display P3 and Linear Display P3 conversions matching upstream Three.js r186 test fixtures.
//! 3. Hexadecimal round-trips (`0xRRGGBB`) across primary, secondary, and boundary colors.
//! 4. HSL hue wraparound, Euclidean modulo, pure hues, and achromatic lightness.
//! 5. CSS style string parsing (`#rgb`, `#rrggbb`, `rgb()`, `rgba()`, `hsl()`, `hsla()`).
//! 6. Linear RGB interpolation (`lerp`) and HSL interpolation (`lerp_hsl`).
//! 7. Matrix constants, mutual inversion, and `apply_matrix3`.
//! 8. Clamped subtraction and arithmetic operations.

use core::fmt::Write;
use f3d_math::color::{
    linear_to_srgb, srgb_to_linear, Color, ColorSpace, Hsl, StyleOutcome, COLOR_NAMES,
    LINEAR_DISPLAY_P3_TO_LINEAR_SRGB, LINEAR_DISPLAY_P3_TO_XYZ, LINEAR_REC709_TO_XYZ,
    LINEAR_SRGB_TO_LINEAR_DISPLAY_P3, XYZ_TO_LINEAR_DISPLAY_P3, XYZ_TO_LINEAR_REC709,
};

const EPS: f64 = 1e-9;

// ============================================================================
// 1. Piecewise sRGB Transfer Function Exactness
// ============================================================================

#[test]
fn test_srgb_piecewise_threshold_exactness() {
    // Condition: c < 0.04045
    // Below threshold: c * 0.0773993808
    let below_thresh = 0.04044;
    let expected_below = 0.04044 * 0.0773993808;
    assert_eq!(srgb_to_linear(below_thresh), expected_below);

    // At threshold: (c * 0.9478672986 + 0.0521327014)^2.4
    let at_thresh = 0.04045;
    let base_at: f64 = 0.04045 * 0.9478672986 + 0.0521327014;
    let expected_at = base_at.powf(2.4);
    assert_eq!(srgb_to_linear(at_thresh), expected_at);

    // Threshold meeting point continuity: both branches meet around 0.0031308
    let linear_from_below = 0.04045 * 0.0773993808;
    assert!((expected_at - linear_from_below).abs() < 1e-6);

    // Boundary values
    assert_eq!(srgb_to_linear(0.0), 0.0);
    assert!((srgb_to_linear(1.0) - 1.0).abs() < 1e-9);

    // Negative values stay on linear branch
    let neg = -0.1;
    assert_eq!(srgb_to_linear(neg), -0.1 * 0.0773993808);

    // Linear to sRGB: Condition c < 0.0031308
    // Below threshold: c * 12.92
    let lin_below = 0.0031307;
    assert_eq!(linear_to_srgb(lin_below), 0.0031307 * 12.92);

    // At threshold: 1.055 * c^0.41666 - 0.055
    let lin_at = 0.0031308;
    let expected_srgb_at = 1.055 * (0.0031308f64).powf(0.41666) - 0.055;
    assert_eq!(linear_to_srgb(lin_at), expected_srgb_at);

    // Continuity around threshold meeting point (approx 0.04045)
    let srgb_from_below = 0.0031308 * 12.92;
    assert!((expected_srgb_at - srgb_from_below).abs() < 1e-4);

    // Zero and unity
    assert_eq!(linear_to_srgb(0.0), 0.0);
    assert!((linear_to_srgb(1.0) - 1.0).abs() < 1e-9);

    // Negative linear values stay on linear branch
    assert_eq!(linear_to_srgb(-0.01), -0.01 * 12.92);
}

// ============================================================================
// 2. Color Spaces: Display P3 and Linear Display P3 Conversions
// ============================================================================

#[test]
fn test_display_p3_conversion_matching_upstream_fixtures() {
    // Replicates Three.js test fixture from ColorSpaces.tests.js lines 27-42:
    // const c = new Color().setRGB(0.3, 0.5, 0.7);
    // ColorManagement.convert(c, LinearSRGBColorSpace, DisplayP3ColorSpace);
    // assert.equal(c.r.toFixed(3), 0.614);
    // assert.equal(c.g.toFixed(3), 0.731);
    // assert.equal(c.b.toFixed(3), 0.843);
    let mut c = Color::new(0.3, 0.5, 0.7);
    c.convert(ColorSpace::LinearSRGB, ColorSpace::DisplayP3);

    assert!((c.r - 0.614).abs() < 1e-3, "r display-p3: {}", c.r);
    assert!((c.g - 0.731).abs() < 1e-3, "g display-p3: {}", c.g);
    assert!((c.b - 0.843).abs() < 1e-3, "b display-p3: {}", c.b);

    // c.setRGB(1.0, 0.5, 0.01, DisplayP3ColorSpace);
    // assert.equal(c.r.toFixed(3), 1.177);
    // assert.equal(c.g.toFixed(3), 0.181);
    // assert.equal(c.b.toFixed(3), -0.036);
    c.set_rgb(1.0, 0.5, 0.01, ColorSpace::DisplayP3);

    assert!((c.r - 1.177).abs() < 1e-3, "r srgb-linear: {}", c.r);
    assert!((c.g - 0.181).abs() < 1e-3, "g srgb-linear: {}", c.g);
    assert!((c.b - (-0.036)).abs() < 1e-3, "b srgb-linear: {}", c.b);

    // assert.equal(c.getStyle(DisplayP3ColorSpace), 'color(display-p3 1.000 0.500 0.010)');
    assert_eq!(
        c.get_style(ColorSpace::DisplayP3),
        "color(display-p3 1.000 0.500 0.010)"
    );
}

#[test]
fn test_linear_display_p3_conversion_matching_upstream_fixtures() {
    // Replicates Three.js test fixture from ColorSpaces.tests.js lines 45-62:
    // const c = new Color().setRGB(0.3, 0.5, 0.7);
    // ColorManagement.convert(c, LinearSRGBColorSpace, LinearDisplayP3ColorSpace);
    // assert.equal(c.r.toFixed(3), 0.336);
    // assert.equal(c.g.toFixed(3), 0.493);
    // assert.equal(c.b.toFixed(3), 0.679);
    let mut c = Color::new(0.3, 0.5, 0.7);
    c.convert(ColorSpace::LinearSRGB, ColorSpace::LinearDisplayP3);

    assert!((c.r - 0.336).abs() < 1e-3, "r display-p3-linear: {}", c.r);
    assert!((c.g - 0.493).abs() < 1e-3, "g display-p3-linear: {}", c.g);
    assert!((c.b - 0.679).abs() < 1e-3, "b display-p3-linear: {}", c.b);

    // c.setRGB(1.0, 0.5, 0.01, LinearDisplayP3ColorSpace);
    // assert.equal(c.r.toFixed(3), 1.112);
    // assert.equal(c.g.toFixed(3), 0.479);
    // assert.equal(c.b.toFixed(3), -0.048);
    c.set_rgb(1.0, 0.5, 0.01, ColorSpace::LinearDisplayP3);

    assert!((c.r - 1.112).abs() < 1e-3, "r srgb-linear: {}", c.r);
    assert!((c.g - 0.479).abs() < 1e-3, "g srgb-linear: {}", c.g);
    assert!((c.b - (-0.048)).abs() < 1e-3, "b srgb-linear: {}", c.b);

    assert_eq!(
        c.get_style(ColorSpace::LinearDisplayP3),
        "color(display-p3-linear 1.000 0.500 0.010)"
    );
}

// ============================================================================
// 3. Hexadecimal Round-Trips
// ============================================================================

#[test]
fn test_hex_round_trip_and_edge_values() {
    let test_hexes = [
        0x000000, // Black
        0xFFFFFF, // White
        0xFF0000, // Red
        0x00FF00, // Green
        0x0000FF, // Blue
        0xFFFF00, // Yellow
        0x00FFFF, // Cyan
        0xFF00FF, // Magenta
        0xFA8072, // Salmon
        0x80FFFF, // Pale cyan
        0x87CEEB, // Sky blue
        0x112233, // Low dark tones
        0x2F4F4F, // Dark slate gray
        0x808080, // Mid gray
    ];

    for &hex in &test_hexes {
        let mut c = Color::black();
        c.set_hex(hex, ColorSpace::SRGB);
        let recovered_hex = c.get_hex(ColorSpace::SRGB);
        assert_eq!(
            recovered_hex, hex,
            "Hex round-trip mismatch: expected 0x{:06x}, got 0x{:06x}",
            hex, recovered_hex
        );

        let hex_str = c.get_hex_string(ColorSpace::SRGB);
        let expected_str = format!("{:06x}", hex);
        assert_eq!(hex_str, expected_str);

        let hex_bytes = c.format_hex_string(ColorSpace::SRGB);
        assert_eq!(core::str::from_utf8(&hex_bytes).unwrap(), expected_str);
    }
}

// ============================================================================
// 4. HSL Hue Wraparound and Conversions
// ============================================================================

#[test]
fn test_hsl_hue_wraparound() {
    // Tests that h values wrapping around [0, 1] modulo 1.0 produce identical results
    let mut c1 = Color::black();
    let mut c2 = Color::black();
    let mut c3 = Color::black();
    let mut c4 = Color::black();

    c1.set_hsl_working(0.75, 1.0, 0.25);
    c2.set_hsl_working(1.75, 1.0, 0.25);
    c3.set_hsl_working(-0.25, 1.0, 0.25);
    c4.set_hsl_working(5.75, 1.0, 0.25);

    assert!((c1.r - c2.r).abs() < EPS);
    assert!((c1.g - c2.g).abs() < EPS);
    assert!((c1.b - c2.b).abs() < EPS);

    assert!((c1.r - c3.r).abs() < EPS);
    assert!((c1.g - c3.g).abs() < EPS);
    assert!((c1.b - c3.b).abs() < EPS);

    assert!((c1.r - c4.r).abs() < EPS);
    assert!((c1.g - c4.g).abs() < EPS);
    assert!((c1.b - c4.b).abs() < EPS);

    let hsl: Hsl = c1.get_hsl_working();
    assert!((hsl.h - 0.75).abs() < 1e-12, "h: {}", hsl.h);
    assert!((hsl.s - 1.00).abs() < 1e-12, "s: {}", hsl.s);
    assert!((hsl.l - 0.25).abs() < 1e-12, "l: {}", hsl.l);
}

#[test]
fn test_hsl_pure_hues_round_trip() {
    let pure_hues = [
        (0.0 / 6.0, 1.0, 0.0, 0.0), // Red
        (1.0 / 6.0, 1.0, 1.0, 0.0), // Yellow
        (2.0 / 6.0, 0.0, 1.0, 0.0), // Green
        (3.0 / 6.0, 0.0, 1.0, 1.0), // Cyan
        (4.0 / 6.0, 0.0, 0.0, 1.0), // Blue
        (5.0 / 6.0, 1.0, 0.0, 1.0), // Magenta
    ];

    for &(h, r, g, b) in &pure_hues {
        let mut c = Color::black();
        c.set_hsl_working(h, 1.0, 0.5);

        assert!((c.r - r).abs() < 1e-9, "hue {}: expected r {}, got {}", h, r, c.r);
        assert!((c.g - g).abs() < 1e-9, "hue {}: expected g {}, got {}", h, g, c.g);
        assert!((c.b - b).abs() < 1e-9, "hue {}: expected b {}, got {}", h, b, c.b);

        let hsl: Hsl = c.get_hsl_working();
        assert!((hsl.h - h).abs() < 1e-9, "hue recovery: expected {}, got {}", h, hsl.h);
        assert!((hsl.s - 1.0).abs() < 1e-9);
        assert!((hsl.l - 0.5).abs() < 1e-9);
    }

    // Achromatic (saturation = 0)
    let mut achrom = Color::black();
    achrom.set_hsl_working(0.35, 0.0, 0.4);
    assert_eq!(achrom.r, 0.4);
    assert_eq!(achrom.g, 0.4);
    assert_eq!(achrom.b, 0.4);

    let achrom_hsl = achrom.get_hsl_working();
    assert_eq!(achrom_hsl.h, 0.0);
    assert_eq!(achrom_hsl.s, 0.0);
    assert_eq!(achrom_hsl.l, 0.4);
}

// ============================================================================
// 5. CSS Style String Parsing
// ============================================================================

#[test]
fn test_style_parsing_hex_rgb_hsl() {
    let mut c = Color::black();

    // 3-digit hex: #rgb
    assert_eq!(c.set_style_srgb("#F00"), StyleOutcome::Applied);
    assert_eq!(c.get_hex_srgb(), 0xFF0000);

    assert_eq!(c.set_style_srgb("#f00"), StyleOutcome::Applied);
    assert_eq!(c.get_hex_srgb(), 0xFF0000);

    assert_eq!(c.set_style_srgb("#F8A"), StyleOutcome::Applied);
    assert_eq!(c.get_hex_srgb(), 0xFF88AA);

    // 6-digit hex: #rrggbb
    assert_eq!(c.set_style_srgb("#87CEEB"), StyleOutcome::Applied);
    assert_eq!(c.get_hex_srgb(), 0x87CEEB);

    assert_eq!(c.set_style_srgb("#87cEeB"), StyleOutcome::Applied);
    assert_eq!(c.get_hex_srgb(), 0x87CEEB);

    // Integer rgb / rgba
    assert_eq!(c.set_style_srgb("rgb(255, 0, 0)"), StyleOutcome::Applied);
    assert_eq!(c.get_hex_srgb(), 0xFF0000);

    assert_eq!(c.set_style_srgb("rgba(255, 0, 0, 0.5)"), StyleOutcome::AlphaIgnored);
    assert_eq!(c.get_hex_srgb(), 0xFF0000);

    assert_eq!(c.set_style_srgb("rgb( 255 , 0,   0 )"), StyleOutcome::Applied);
    assert_eq!(c.get_hex_srgb(), 0xFF0000);

    assert_eq!(c.set_style_srgb("rgba( 255,  0,  0  , 1 )"), StyleOutcome::Applied);
    assert_eq!(c.get_hex_srgb(), 0xFF0000);

    assert_eq!(c.set_style_srgb("rgba( 255,  0,  0  , 1.0 )"), StyleOutcome::Applied);
    assert_eq!(c.get_hex_srgb(), 0xFF0000);

    // Percentage rgb / rgba
    assert_eq!(c.set_style_srgb("rgb(100%, 50%, 10%)"), StyleOutcome::Applied);
    assert_eq!(c.get_hex_srgb(), 0xFF801A);

    assert_eq!(c.set_style_srgb("rgba( 100% ,50%  ,  10%, 0.5 )"), StyleOutcome::AlphaIgnored);
    assert_eq!(c.get_hex_srgb(), 0xFF801A);

    assert_eq!(c.set_style_srgb("rgba( 100% ,50%  ,  10%, 1.0 )"), StyleOutcome::Applied);
    assert_eq!(c.get_hex_srgb(), 0xFF801A);

    // HSL / HSLA
    assert_eq!(c.set_style_srgb("hsl(270, 50%, 75%)"), StyleOutcome::Applied);
    let hsl = c.get_hsl(ColorSpace::SRGB);
    assert!((hsl.h - 0.75).abs() < 1e-2);
    assert!((hsl.s - 0.50).abs() < 1e-2);
    assert!((hsl.l - 0.75).abs() < 1e-2);

    assert_eq!(c.set_style_srgb("hsla(360, 100.0%, 50.0%, 0.5)"), StyleOutcome::AlphaIgnored);
    assert_eq!(c.get_hex_srgb(), 0xFF0000);

    assert_eq!(c.set_style_srgb("hsla(360, 100.0%, 50.0%, 1.0)"), StyleOutcome::Applied);
    assert_eq!(c.get_hex_srgb(), 0xFF0000);

    // Invalid style strings return IgnoredInvalid
    assert_eq!(c.set_style_srgb(""), StyleOutcome::IgnoredInvalid);
    assert_eq!(c.set_style_srgb("not-a-color"), StyleOutcome::IgnoredInvalid);
    assert_eq!(c.set_style_srgb("#12"), StyleOutcome::IgnoredInvalid);
    assert_eq!(c.set_style_srgb("#1234"), StyleOutcome::IgnoredInvalid);
    assert_eq!(c.set_style_srgb("#12345"), StyleOutcome::IgnoredInvalid);
    assert_eq!(c.set_style_srgb("rgb(10, 20)"), StyleOutcome::IgnoredInvalid);
    assert_eq!(c.set_style_srgb("hsl(10, 20%)"), StyleOutcome::IgnoredInvalid);
}

// ============================================================================
// 6. Interpolation: lerp and lerp_hsl
// ============================================================================

#[test]
fn test_lerp_rgb() {
    let mut c1 = Color::new(0.0, 0.0, 0.0);
    let c2 = Color::new(1.0, 0.5, 0.2);

    c1.lerp(&c2, 0.2);
    assert!((c1.r - 0.2).abs() < EPS);
    assert!((c1.g - 0.1).abs() < EPS);
    assert!((c1.b - 0.04).abs() < EPS);

    // alpha = 0 retains c1
    let initial = c1;
    c1.lerp(&c2, 0.0);
    assert_eq!(c1, initial);

    // alpha = 1 becomes c2
    c1.lerp(&c2, 1.0);
    assert!((c1.r - c2.r).abs() < EPS);
    assert!((c1.g - c2.g).abs() < EPS);
    assert!((c1.b - c2.b).abs() < EPS);
}

#[test]
fn test_lerp_hsl() {
    // Red (h = 0.0, s = 1.0, l = 0.5) to Cyan (h = 0.5, s = 1.0, l = 0.5)
    let mut c1 = Color::black();
    c1.set_hsl_working(0.0, 1.0, 0.5);

    let mut c2 = Color::black();
    c2.set_hsl_working(0.5, 1.0, 0.5);

    // Midpoint at alpha = 0.5 should have h = 0.25 (chartreuse green)
    c1.lerp_hsl(&c2, 0.5);
    let hsl = c1.get_hsl_working();

    assert!((hsl.h - 0.25).abs() < 1e-4, "h: {}", hsl.h);
    assert!((hsl.s - 1.00).abs() < 1e-4, "s: {}", hsl.s);
    assert!((hsl.l - 0.50).abs() < 1e-4, "l: {}", hsl.l);
}

// ============================================================================
// 7. Matrix Constants and apply_matrix3
// ============================================================================

#[test]
fn test_matrix_constants_and_apply_matrix3() {
    let mut c = Color::new(0.3, 0.5, 0.7);

    // Applying direct sRGB -> Display P3 matrix
    let mut c_direct = c;
    c_direct.apply_matrix3(&LINEAR_SRGB_TO_LINEAR_DISPLAY_P3);

    // Applying via two-step XYZ intermediate
    let mut c_two_step = c;
    c_two_step.apply_matrix3(&LINEAR_REC709_TO_XYZ);
    c_two_step.apply_matrix3(&XYZ_TO_LINEAR_DISPLAY_P3);

    assert!(
        (c_direct.r - c_two_step.r).abs() < 1e-5,
        "direct r {} vs two-step r {}",
        c_direct.r,
        c_two_step.r
    );
    assert!(
        (c_direct.g - c_two_step.g).abs() < 1e-5,
        "direct g {} vs two-step g {}",
        c_direct.g,
        c_two_step.g
    );
    assert!(
        (c_direct.b - c_two_step.b).abs() < 1e-5,
        "direct b {} vs two-step b {}",
        c_direct.b,
        c_two_step.b
    );

    // Round-trip through inverse matrix
    let mut c_recovered = c_direct;
    c_recovered.apply_matrix3(&LINEAR_DISPLAY_P3_TO_LINEAR_SRGB);

    assert!((c_recovered.r - c.r).abs() < 1e-4);
    assert!((c_recovered.g - c.g).abs() < 1e-4);
    assert!((c_recovered.b - c.b).abs() < 1e-4);

    // Two-step reverse conversion through XYZ intermediate
    let mut c_two_step_rev = c_direct;
    c_two_step_rev.apply_matrix3(&LINEAR_DISPLAY_P3_TO_XYZ);
    c_two_step_rev.apply_matrix3(&XYZ_TO_LINEAR_REC709);

    assert!((c_two_step_rev.r - c.r).abs() < 1e-4);
    assert!((c_two_step_rev.g - c.g).abs() < 1e-4);
    assert!((c_two_step_rev.b - c.b).abs() < 1e-4);
}

// ============================================================================
// 8. Arithmetic Operations and Clamped Subtraction
// ============================================================================

#[test]
fn test_color_arithmetic_and_clamped_sub() {
    let mut a = Color::new(0.2, 0.5, 0.8);
    let b = Color::new(0.4, 0.6, 0.9);

    // Addition
    a.add(&b);
    assert!((a.r - 0.6).abs() < EPS);
    assert!((a.g - 1.1).abs() < EPS);
    assert!((a.b - 1.7).abs() < EPS);

    // Add scalar
    a.add_scalar(0.1);
    assert!((a.r - 0.7).abs() < EPS);
    assert!((a.g - 1.2).abs() < EPS);
    assert!((a.b - 1.8).abs() < EPS);

    // Subtraction clamps to 0.0 per Three.js r186 Math.max(0, this.r - color.r)
    let mut c = Color::new(0.5, 0.2, 0.1);
    let d = Color::new(0.8, 0.1, 0.3);
    c.sub(&d);
    assert_eq!(c.r, 0.0); // 0.5 - 0.8 -> clamped to 0.0
    assert!((c.g - 0.1).abs() < EPS); // 0.2 - 0.1 = 0.1
    assert_eq!(c.b, 0.0); // 0.1 - 0.3 -> clamped to 0.0

    // Multiply
    let mut m = Color::new(0.5, 0.4, 0.2);
    m.multiply(&Color::new(2.0, 0.5, 3.0));
    assert!((m.r - 1.0).abs() < EPS);
    assert!((m.g - 0.2).abs() < EPS);
    assert!((m.b - 0.6).abs() < EPS);

    // Multiply scalar
    m.multiply_scalar(2.0);
    assert!((m.r - 2.0).abs() < EPS);
    assert!((m.g - 0.4).abs() < EPS);
    assert!((m.b - 1.2).abs() < EPS);
}

#[test]
fn test_get_style_formatting() {
    let mut c = Color::black();
    c.set_hex(0xFF0000, ColorSpace::SRGB);
    assert_eq!(c.get_style_srgb(), "rgb(255,0,0)");
    assert_eq!(c.get_style(ColorSpace::SRGB), "rgb(255,0,0)");

    c.set_hex(0x00FF00, ColorSpace::SRGB);
    assert_eq!(c.get_style_srgb(), "rgb(0,255,0)");

    let p3_style = c.get_style(ColorSpace::DisplayP3);
    assert!(p3_style.starts_with("color(display-p3 "));
    assert!(p3_style.ends_with(')'));
}

// ============================================================================
// 9. Upstream Divergence Regressions (Ruby Review Fold)
// ============================================================================

#[test]
fn test_style_parsing_upstream_divergence_regressions() {
    // 1. Integer-only \d+ requirement for rgb() and rgb(%) components
    // Floats in integer rgb() must be rejected: Color.js:304 /^\s*(\d+)...$/
    let mut c = Color::new(0.1, 0.2, 0.3);
    let initial = c;

    assert_eq!(c.set_style_srgb("rgb(255.5, 0, 0)"), StyleOutcome::IgnoredInvalid);
    assert_eq!(c, initial, "color must remain unchanged on invalid style");

    assert_eq!(c.set_style_srgb("rgb(50.5%, 0%, 0%)"), StyleOutcome::IgnoredInvalid);
    assert_eq!(c, initial, "color must remain unchanged on invalid style");

    assert_eq!(c.set_style_srgb("rgb(-10, 0, 0)"), StyleOutcome::IgnoredInvalid);
    assert_eq!(c, initial);

    // Whitespace before % rejected per Color.js:331 /^\s*(\d+)\%\s*...$/
    assert_eq!(c.set_style_srgb("rgb(100 %, 50%, 0%)"), StyleOutcome::IgnoredInvalid);
    assert_eq!(c, initial);

    assert_eq!(c.set_style_srgb("hsl(0, 100 %, 50%)"), StyleOutcome::IgnoredInvalid);
    assert_eq!(c, initial);

    // 2. Case-sensitive function names: Color.js:292 switch (name)
    assert_eq!(c.set_style_srgb("RGB(255, 0, 0)"), StyleOutcome::IgnoredInvalid);
    assert_eq!(c, initial);

    assert_eq!(c.set_style_srgb("RGBA(255, 0, 0, 1)"), StyleOutcome::IgnoredInvalid);
    assert_eq!(c, initial);

    assert_eq!(c.set_style_srgb("Hsl(0, 100%, 50%)"), StyleOutcome::IgnoredInvalid);
    assert_eq!(c, initial);

    // 3. No outer trim: Color.js:290 /^(\w+)\(([^\)]*)\)/
    assert_eq!(c.set_style_srgb("  rgb(255, 0, 0)"), StyleOutcome::IgnoredInvalid);
    assert_eq!(c, initial);

    assert_eq!(c.set_style_srgb(" #ff0000"), StyleOutcome::IgnoredInvalid);
    assert_eq!(c, initial);

    // 4. Trailing content after ')' tolerated: Color.js:290 /^(\w+)\(([^\)]*)\)/ has no trailing $ anchor
    let mut red = Color::black();
    assert_eq!(red.set_style_srgb("rgb(255, 0, 0)trailing text"), StyleOutcome::Applied);
    assert_eq!(red.get_hex_srgb(), 0xFF0000);

    let mut green = Color::black();
    assert_eq!(green.set_style_srgb("rgba(0, 255, 0, 0.5); font-weight: bold;"), StyleOutcome::AlphaIgnored);
    assert_eq!(green.get_hex_srgb(), 0x00FF00);

    let mut blue = Color::black();
    assert_eq!(blue.set_style_srgb("hsl(240, 100%, 50%)!important"), StyleOutcome::Applied);
    assert_eq!(blue.get_hex_srgb(), 0x0000FF);

    // 5. Invariant: invalid input leaves color components completely unmodified
    let sentinel = Color::new(0.123, 0.456, 0.789);
    let mut test_color = sentinel;

    let invalid_inputs = [
        "",
        "   ",
        "not_a_color",
        "rgba(255, 0, 0, abc)",
        "rgba(255, 0, 0, 1.)",
        "rgb(255, 0, 0, 1, 2)",
        "hsl(100, 50)",
        "hsl(100, 50%, 50%, 0.5, extra)",
        "#12",
        "#1234",
        "#12345",
        "#1234567",
        "#gggggg",
    ];

    for &input in &invalid_inputs {
        let outcome = test_color.set_style_srgb(input);
        assert_eq!(outcome, StyleOutcome::IgnoredInvalid, "input '{}' should be IgnoredInvalid", input);
        assert_eq!(test_color, sentinel, "input '{}' must leave color untouched", input);
    }
}

// ============================================================================
// 10. CSS Named Colors and set_color_name
// ============================================================================

#[test]
fn test_css_named_colors_and_set_color_name() {
    // 1. Exactly 148 keywords in COLOR_NAMES matching Three.js r186 Color.js:6-29 (140 distinct colors plus 8 aliases)
    assert_eq!(COLOR_NAMES.len(), 148, "COLOR_NAMES must have exactly 148 entries");

    // All names must be lowercase ASCII and unique
    for i in 0..COLOR_NAMES.len() {
        let (name, hex) = COLOR_NAMES[i];
        assert!(!name.is_empty(), "color name cannot be empty");
        assert!(name.chars().all(|c| c.is_ascii_lowercase()), "name '{}' must be lowercase", name);
        assert!(hex <= 0xFFFFFF, "hex 0x{:x} out of 24-bit range", hex);

        for j in (i + 1)..COLOR_NAMES.len() {
            assert_ne!(COLOR_NAMES[i].0, COLOR_NAMES[j].0, "duplicate name in COLOR_NAMES: {}", name);
        }
    }

    // 2. set_color_name direct calls
    let mut c = Color::black();

    // Standard lowercase
    assert_eq!(c.set_color_name_srgb("aliceblue"), StyleOutcome::Applied);
    assert_eq!(c.get_hex_srgb(), 0xF0F8FF);

    assert_eq!(c.set_color_name_srgb("powderblue"), StyleOutcome::Applied);
    assert_eq!(c.get_hex_srgb(), 0xB0E0E6);

    assert_eq!(c.set_color_name_srgb("rebeccapurple"), StyleOutcome::Applied);
    assert_eq!(c.get_hex_srgb(), 0x663399);

    // Case-insensitivity matching Three.js Color.js:426 style.toLowerCase()
    assert_eq!(c.set_color_name_srgb("AliceBlue"), StyleOutcome::Applied);
    assert_eq!(c.get_hex_srgb(), 0xF0F8FF);

    assert_eq!(c.set_color_name_srgb("POWDERBLUE"), StyleOutcome::Applied);
    assert_eq!(c.get_hex_srgb(), 0xB0E0E6);

    assert_eq!(c.set_color_name_srgb("ReBeCcApUrPlE"), StyleOutcome::Applied);
    assert_eq!(c.get_hex_srgb(), 0x663399);

    // Unknown name leaves color unmodified
    let prev = c;
    assert_eq!(c.set_color_name_srgb("foobar_nonexistent"), StyleOutcome::IgnoredInvalid);
    assert_eq!(c, prev);

    // 3. Fallback through set_style
    let mut style_c = Color::black();
    assert_eq!(style_c.set_style_srgb("coral"), StyleOutcome::Applied);
    assert_eq!(style_c.get_hex_srgb(), 0xFF7F50);

    assert_eq!(style_c.set_style_srgb("DeepSkyBlue"), StyleOutcome::Applied);
    assert_eq!(style_c.get_hex_srgb(), 0x00BFFF);
}

// ============================================================================
// 11. Zero-Allocation Formatting and Buffers
// ============================================================================

#[test]
fn test_zero_allocation_style_and_hex_formatting() {
    let mut c = Color::black();
    c.set_hex(0x32CD32, ColorSpace::SRGB); // limegreen

    // Hex string into stack buffer
    let hex_bytes = c.format_hex_string(ColorSpace::SRGB);
    assert_eq!(&hex_bytes, b"32cd32");

    // Hex string into core::fmt::Write
    struct StackWriter {
        buf: [u8; 32],
        pos: usize,
    }
    impl Write for StackWriter {
        fn write_str(&mut self, s: &str) -> core::fmt::Result {
            let bytes = s.as_bytes();
            if self.pos + bytes.len() > self.buf.len() {
                return Err(core::fmt::Error);
            }
            self.buf[self.pos..self.pos + bytes.len()].copy_from_slice(bytes);
            self.pos += bytes.len();
            Ok(())
        }
    }

    let mut sw = StackWriter { buf: [0u8; 32], pos: 0 };
    c.write_hex_string(&mut sw, ColorSpace::SRGB).unwrap();
    assert_eq!(core::str::from_utf8(&sw.buf[..sw.pos]).unwrap(), "32cd32");

    // Style into caller buffer
    let mut style_buf = [0u8; 64];
    let style_str = c.format_style_srgb(&mut style_buf).unwrap();
    assert_eq!(style_str, "rgb(50,205,50)");

    // Style in Display P3
    let p3_str = c.format_style(&mut style_buf, ColorSpace::DisplayP3).unwrap();
    assert!(p3_str.starts_with("color(display-p3 "));
    assert!(p3_str.ends_with(')'));

    // Short buffer (10 bytes) returns Err(fmt::Error) without truncation or panic
    let mut short_buf_10 = [0u8; 10];
    assert!(c.format_style_srgb(&mut short_buf_10).is_err());
    assert!(c.format_style(&mut short_buf_10, ColorSpace::SRGB).is_err());

    // Upstream Color.js:650 does NOT clamp r,g,b in getStyle for sRGB (only getHex clamps)
    let mut out_of_gamut = Color::new(0.0, 0.0, 0.0);
    out_of_gamut.set_rgb(300.0 / 255.0, -20.0 / 255.0, 100.0 / 255.0, ColorSpace::SRGB);
    let mut oog_buf = [0u8; 32];
    let oog_style = out_of_gamut.format_style_srgb(&mut oog_buf).unwrap();
    assert_eq!(oog_style, "rgb(300,-20,100)");
    #[cfg(feature = "std")]
    {
        assert_eq!(out_of_gamut.get_style_srgb(), "rgb(300,-20,100)");
    }

    // Nonfinite channels format as JS strings ("NaN", "Infinity", "-Infinity") matching upstream getStyle
    let non_finite_color = Color::new(f64::NAN, f64::INFINITY, f64::NEG_INFINITY);
    let mut nf_buf = [0u8; 64];
    let nf_style = non_finite_color.format_style_srgb(&mut nf_buf).unwrap();
    assert_eq!(nf_style, "rgb(NaN,Infinity,-Infinity)");
    #[cfg(feature = "std")]
    {
        assert_eq!(non_finite_color.get_style_srgb(), "rgb(NaN,Infinity,-Infinity)");
    }

    // Large finite channels at 2^63:
    // Three.js r186 Color.getStyle(SRGBColorSpace) converts working linear space to sRGB via linear_to_srgb:
    // r = 2^63 / 255 -> linear_to_srgb(r) * 255 = 2133018191
    // g = -2^63 / 255 -> linear_to_srgb(g) * 255 = -2^63 * 12.92 -> "-119165966716163700000"
    // b = 100 / 255 -> linear_to_srgb(b) * 255 = 168
    let c_2_63 = Color::new(2.0_f64.powi(63) / 255.0, -2.0_f64.powi(63) / 255.0, 100.0 / 255.0);
    let mut buf_63 = [0u8; 64];
    let style_63 = c_2_63.format_style_srgb(&mut buf_63).unwrap();
    assert_eq!(style_63, "rgb(2133018191,-119165966716163700000,168)");
    #[cfg(feature = "std")]
    {
        assert_eq!(c_2_63.get_style_srgb(), "rgb(2133018191,-119165966716163700000,168)");
    }

    // Representable values near 1e18 beyond 2^53:
    // Under linear_to_srgb, (1e18/255)**(1/2.4) * 1.055 * 255 -> 845208004
    let c_1e18 = Color::new((1e18 + 1024.0) / 255.0, 1e18 / 255.0, 1.0);
    let mut buf_18 = [0u8; 64];
    let style_18 = c_1e18.format_style_srgb(&mut buf_18).unwrap();
    assert_eq!(style_18, "rgb(845208004,845208004,255)");
    #[cfg(feature = "std")]
    {
        assert_eq!(c_1e18.get_style_srgb(), "rgb(845208004,845208004,255)");
    }

    // Exponential formatting at >= 1e21 with signed exponent (e.g. 1e+21) matching ECMAScript:
    // Under linear_to_srgb:
    // r = 1e21/255 -> 15029468000
    // g = -1e21/255 -> -1e21 * 12.92 = -1.292e+22
    let c_1e21 = Color::new(1e21 / 255.0, -1e21 / 255.0, 0.0);
    let mut buf_21 = [0u8; 64];
    let style_21 = c_1e21.format_style_srgb(&mut buf_21).unwrap();
    assert_eq!(style_21, "rgb(15029468000,-1.292e+22,0)");
    #[cfg(feature = "std")]
    {
        assert_eq!(c_1e21.get_style_srgb(), "rgb(15029468000,-1.292e+22,0)");
    }

    // Large exponential (2.55e27) and decimal at 1e20 (< 1e21 threshold):
    // Under linear_to_srgb:
    // r = 2.55e27/255 -> 7019281183825
    // g = 1e20/255 -> 5758158397
    // b = -1e20/255 -> -1.292e+21
    let c_large = Color::new(2.55e27 / 255.0, 1e20 / 255.0, -1e20 / 255.0);
    let mut buf_large = [0u8; 96];
    let style_large = c_large.format_style_srgb(&mut buf_large).unwrap();
    assert_eq!(style_large, "rgb(7019281183825,5758158397,-1.292e+21)");
    #[cfg(feature = "std")]
    {
        assert_eq!(c_large.get_style_srgb(), "rgb(7019281183825,5758158397,-1.292e+21)");
    }
}

// ============================================================================
// 12. get_hex js_round Exactness
// ============================================================================

#[test]
fn test_get_hex_js_round_exactness() {
    // Three.js r186 getHex: Math.round(clamp(c * 255, 0, 255))
    // Uses js_round (half toward +Infinity)
    // 0.5 / 255.0 * 255.0 = 0.5 -> js_round(0.5) = 1.0
    let c_half = Color::new(0.5 / 255.0, 1.5 / 255.0, 2.5 / 255.0);
    // LinearSRGB: get_hex in LinearSRGB directly tests round
    let hex_lin = c_half.get_hex(ColorSpace::LinearSRGB);
    let r = (hex_lin >> 16) & 0xFF;
    let g = (hex_lin >> 8) & 0xFF;
    let b = hex_lin & 0xFF;

    assert_eq!(r, 1);
    assert_eq!(g, 2);
    assert_eq!(b, 3);
}

