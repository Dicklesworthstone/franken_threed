//! Three-component RGB color with `f64` semantics matching Three.js r186 `Color` and `ColorManagement`.
//!
//! # Storage and Representation
//! A [`Color`] instance stores its components `(r, g, b)` as 64-bit IEEE 754 floats in the
//! linear working color space ([`ColorSpace::LinearSRGB`]).
//! Inputs conventionally specified in standard sRGB (such as hexadecimals `0xRRGGBB` and CSS
//! color strings) are transformed to the linear working color space via exact piecewise transfer
//! functions matching Three.js r186 `ColorManagement.js`.
//!
//! # Piecewise sRGB Transfer Formulas
//! The piecewise sRGB $\leftrightarrow$ Linear transfer functions strictly adhere to the
//! numerical thresholds and coefficients of Three.js r186:
//! - **sRGB to Linear**:
//!   $$\text{linear}(c) = \begin{cases} c \times 0.0773993808, & c < 0.04045 \\ (c \times 0.9478672986 + 0.0521327014)^{2.4}, & c \ge 0.04045 \end{cases}$$
//! - **Linear to sRGB**:
//!   $$\text{sRGB}(c) = \begin{cases} c \times 12.92, & c < 0.0031308 \\ 1.055 \times c^{0.41666} - 0.055, & c \ge 0.0031308 \end{cases}$$
//!
//! # Display P3 Transformations
//! Primaries conversions between Rec. 709 (sRGB) and Display P3 evaluate through intermediate
//! CIE 1931 XYZ space using the canonical Three.js r186 3x3 transformation matrices.

use core::fmt::{self, Write};
use crate::matrix3::Matrix3;
use crate::vector3::Vector3;

#[cfg(feature = "std")]
use std::string::String;

// ============================================================================
// Upstream Transformation Matrices
// ============================================================================

/// CIE XYZ to Linear Rec. 709 (sRGB) 3x3 matrix from Three.js r186 `ColorManagement.js`.
pub const XYZ_TO_LINEAR_REC709: Matrix3 = Matrix3::new(
    3.2409699, -1.5373832, -0.4986108,
    -0.9692436, 1.8759675, 0.0415551,
    0.0556301, -0.2039770, 1.0569715,
);

/// Linear Rec. 709 (sRGB) to CIE XYZ 3x3 matrix from Three.js r186 `ColorManagement.js`.
pub const LINEAR_REC709_TO_XYZ: Matrix3 = Matrix3::new(
    0.4123908, 0.3575843, 0.1804808,
    0.2126390, 0.7151687, 0.0721923,
    0.0193308, 0.1191948, 0.9505322,
);

/// CIE XYZ to Linear Display P3 3x3 matrix from Three.js r186 `ColorSpaces.js`.
pub const XYZ_TO_LINEAR_DISPLAY_P3: Matrix3 = Matrix3::new(
    2.4934969, -0.9313836, -0.4027108,
    -0.8294890, 1.7626641, 0.0236247,
    0.0358458, -0.0761724, 0.9568845,
);

/// Linear Display P3 to CIE XYZ 3x3 matrix from Three.js r186 `ColorSpaces.js`.
pub const LINEAR_DISPLAY_P3_TO_XYZ: Matrix3 = Matrix3::new(
    0.4865709, 0.2656677, 0.1982173,
    0.2289746, 0.6917385, 0.0792869,
    0.0000000, 0.0451134, 1.0439444,
);

/// Direct Linear sRGB (Rec. 709) to Linear Display P3 3x3 matrix.
///
/// Evaluated as `XYZ_TO_LINEAR_DISPLAY_P3 * LINEAR_REC709_TO_XYZ`.
pub const LINEAR_SRGB_TO_LINEAR_DISPLAY_P3: Matrix3 = Matrix3::new(
    0.8224620, 0.1775380, 0.0000000,
    0.0331942, 0.9668058, 0.0000000,
    0.0170826, 0.0723974, 0.9105200,
);

/// Direct Linear Display P3 to Linear sRGB (Rec. 709) 3x3 matrix.
///
/// Evaluated as `XYZ_TO_LINEAR_REC709 * LINEAR_DISPLAY_P3_TO_XYZ`.
pub const LINEAR_DISPLAY_P3_TO_LINEAR_SRGB: Matrix3 = Matrix3::new(
    1.2249401, -0.2249404, 0.0000000,
    -0.0420569, 1.0420571, 0.0000000,
    -0.0196376, -0.0786361, 1.0982735,
);

// ============================================================================
// 140 CSS Named Colors Table (Color.js:6-29)
// ============================================================================

/// The 148 standard CSS color keyword entries (140 distinct colors plus 8 grey/gray and synonym aliases) matching Three.js r186 `_colorKeywords`.
pub const COLOR_NAMES: [(&str, u32); 148] = [
    ("aliceblue", 0xF0F8FF),
    ("antiquewhite", 0xFAEBD7),
    ("aqua", 0x00FFFF),
    ("aquamarine", 0x7FFFD4),
    ("azure", 0xF0FFFF),
    ("beige", 0xF5F5DC),
    ("bisque", 0xFFE4C4),
    ("black", 0x000000),
    ("blanchedalmond", 0xFFEBCD),
    ("blue", 0x0000FF),
    ("blueviolet", 0x8A2BE2),
    ("brown", 0xA52A2A),
    ("burlywood", 0xDEB887),
    ("cadetblue", 0x5F9EA0),
    ("chartreuse", 0x7FFF00),
    ("chocolate", 0xD2691E),
    ("coral", 0xFF7F50),
    ("cornflowerblue", 0x6495ED),
    ("cornsilk", 0xFFF8DC),
    ("crimson", 0xDC143C),
    ("cyan", 0x00FFFF),
    ("darkblue", 0x00008B),
    ("darkcyan", 0x008B8B),
    ("darkgoldenrod", 0xB8860B),
    ("darkgray", 0xA9A9A9),
    ("darkgreen", 0x006400),
    ("darkgrey", 0xA9A9A9),
    ("darkkhaki", 0xBDB76B),
    ("darkmagenta", 0x8B008B),
    ("darkolivegreen", 0x556B2F),
    ("darkorange", 0xFF8C00),
    ("darkorchid", 0x9932CC),
    ("darkred", 0x8B0000),
    ("darksalmon", 0xE9967A),
    ("darkseagreen", 0x8FBC8F),
    ("darkslateblue", 0x483D8B),
    ("darkslategray", 0x2F4F4F),
    ("darkslategrey", 0x2F4F4F),
    ("darkturquoise", 0x00CED1),
    ("darkviolet", 0x9400D3),
    ("deeppink", 0xFF1493),
    ("deepskyblue", 0x00BFFF),
    ("dimgray", 0x696969),
    ("dimgrey", 0x696969),
    ("dodgerblue", 0x1E90FF),
    ("firebrick", 0xB22222),
    ("floralwhite", 0xFFFAF0),
    ("forestgreen", 0x228B22),
    ("fuchsia", 0xFF00FF),
    ("gainsboro", 0xDCDCDC),
    ("ghostwhite", 0xF8F8FF),
    ("gold", 0xFFD700),
    ("goldenrod", 0xDAA520),
    ("gray", 0x808080),
    ("green", 0x008000),
    ("greenyellow", 0xADFF2F),
    ("grey", 0x808080),
    ("honeydew", 0xF0FFF0),
    ("hotpink", 0xFF69B4),
    ("indianred", 0xCD5C5C),
    ("indigo", 0x4B0082),
    ("ivory", 0xFFFFF0),
    ("khaki", 0xF0E68C),
    ("lavender", 0xE6E6FA),
    ("lavenderblush", 0xFFF0F5),
    ("lawngreen", 0x7CFC00),
    ("lemonchiffon", 0xFFFACD),
    ("lightblue", 0xADD8E6),
    ("lightcoral", 0xF08080),
    ("lightcyan", 0xE0FFFF),
    ("lightgoldenrodyellow", 0xFAFAD2),
    ("lightgray", 0xD3D3D3),
    ("lightgreen", 0x90EE90),
    ("lightgrey", 0xD3D3D3),
    ("lightpink", 0xFFB6C1),
    ("lightsalmon", 0xFFA07A),
    ("lightseagreen", 0x20B2AA),
    ("lightskyblue", 0x87CEFA),
    ("lightslategray", 0x778899),
    ("lightslategrey", 0x778899),
    ("lightsteelblue", 0xB0C4DE),
    ("lightyellow", 0xFFFFE0),
    ("lime", 0x00FF00),
    ("limegreen", 0x32CD32),
    ("linen", 0xFAF0E6),
    ("magenta", 0xFF00FF),
    ("maroon", 0x800000),
    ("mediumaquamarine", 0x66CDAA),
    ("mediumblue", 0x0000CD),
    ("mediumorchid", 0xBA55D3),
    ("mediumpurple", 0x9370DB),
    ("mediumseagreen", 0x3CB371),
    ("mediumslateblue", 0x7B68EE),
    ("mediumspringgreen", 0x00FA9A),
    ("mediumturquoise", 0x48D1CC),
    ("mediumvioletred", 0xC71585),
    ("midnightblue", 0x191970),
    ("mintcream", 0xF5FFFA),
    ("mistyrose", 0xFFE4E1),
    ("moccasin", 0xFFE4B5),
    ("navajowhite", 0xFFDEAD),
    ("navy", 0x000080),
    ("oldlace", 0xFDF5E6),
    ("olive", 0x808000),
    ("olivedrab", 0x6B8E23),
    ("orange", 0xFFA500),
    ("orangered", 0xFF4500),
    ("orchid", 0xDA70D6),
    ("palegoldenrod", 0xEEE8AA),
    ("palegreen", 0x98FB98),
    ("paleturquoise", 0xAFEEEE),
    ("palevioletred", 0xDB7093),
    ("papayawhip", 0xFFEFD5),
    ("peachpuff", 0xFFDAB9),
    ("peru", 0xCD853F),
    ("pink", 0xFFC0CB),
    ("plum", 0xDDA0DD),
    ("powderblue", 0xB0E0E6),
    ("purple", 0x800080),
    ("rebeccapurple", 0x663399),
    ("red", 0xFF0000),
    ("rosybrown", 0xBC8F8F),
    ("royalblue", 0x4169E1),
    ("saddlebrown", 0x8B4513),
    ("salmon", 0xFA8072),
    ("sandybrown", 0xF4A460),
    ("seagreen", 0x2E8B57),
    ("seashell", 0xFFF5EE),
    ("sienna", 0xA0522D),
    ("silver", 0xC0C0C0),
    ("skyblue", 0x87CEEB),
    ("slateblue", 0x6A5ACD),
    ("slategray", 0x708090),
    ("slategrey", 0x708090),
    ("snow", 0xFFFAFA),
    ("springgreen", 0x00FF7F),
    ("steelblue", 0x4682B4),
    ("tan", 0xD2B48C),
    ("teal", 0x008080),
    ("thistle", 0xD8BFD8),
    ("tomato", 0xFF6347),
    ("turquoise", 0x40E0D0),
    ("violet", 0xEE82EE),
    ("wheat", 0xF5DEB3),
    ("white", 0xFFFFFF),
    ("whitesmoke", 0xF5F5F5),
    ("yellow", 0xFFFF00),
    ("yellowgreen", 0x9ACD32),
];

// ============================================================================
// StyleOutcome Enum
// ============================================================================

/// Outcome of a `set_style` or `set_color_name` operation matching Three.js r186 behavior.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StyleOutcome {
    /// Style was parsed successfully and applied without alpha warning.
    Applied,
    /// Style was applied, but alpha component < 1 was ignored (matching Three.js warning).
    AlphaIgnored,
    /// Style input was invalid or unrecognized; color components were left unchanged.
    IgnoredInvalid,
}

impl StyleOutcome {
    /// Returns `true` if the color was modified (`Applied` or `AlphaIgnored`).
    #[inline]
    pub const fn is_applied(&self) -> bool {
        matches!(self, Self::Applied | Self::AlphaIgnored)
    }

    /// Returns `true` if the input was invalid and the color was left unchanged.
    #[inline]
    pub const fn is_ignored_invalid(&self) -> bool {
        matches!(self, Self::IgnoredInvalid)
    }
}

// ============================================================================
// Piecewise Transfer Functions
// ============================================================================

/// Converts a single scalar color component from standard sRGB to Linear sRGB.
///
/// Matches Three.js r186 `ColorManagement.SRGBToLinear`:
/// ```text
/// (c < 0.04045) ? c * 0.0773993808 : Math.pow(c * 0.9478672986 + 0.0521327014, 2.4)
/// ```
#[inline]
pub fn srgb_to_linear(c: f64) -> f64 {
    if c < 0.04045 {
        c * 0.0773993808
    } else {
        (c * 0.9478672986 + 0.0521327014).powf(2.4)
    }
}

/// Converts a single scalar color component from Linear sRGB to standard sRGB.
///
/// Matches Three.js r186 `ColorManagement.LinearToSRGB`:
/// ```text
/// (c < 0.0031308) ? c * 12.92 : 1.055 * (Math.pow(c, 0.41666)) - 0.055
/// ```
#[inline]
pub fn linear_to_srgb(c: f64) -> f64 {
    if c < 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(0.41666) - 0.055
    }
}

/// Clamps value between `min` and `max` matching Three.js `MathUtils.clamp`.
#[inline]
fn clamp(value: f64, min: f64, max: f64) -> f64 {
    value.max(min).min(max)
}

/// Euclidean modulo `((n % m) + m) % m` matching Three.js `MathUtils.euclideanModulo`.
#[inline]
fn euclidean_modulo(n: f64, m: f64) -> f64 {
    ((n % m) + m) % m
}

/// Converts HSL hue channel to RGB component matching Three.js r186 `hue2rgb`.
#[inline]
pub fn hue2rgb(p: f64, q: f64, mut t: f64) -> f64 {
    if t < 0.0 {
        t += 1.0;
    }
    if t > 1.0 {
        t -= 1.0;
    }
    if t < 1.0 / 6.0 {
        p + (q - p) * 6.0 * t
    } else if t < 1.0 / 2.0 {
        q
    } else if t < 2.0 / 3.0 {
        p + (q - p) * 6.0 * (2.0 / 3.0 - t)
    } else {
        p
    }
}

// ============================================================================
// String Parsing Helper Functions (Zero-Allocation)
// ============================================================================

/// Checks if string consists strictly of ASCII digits `0-9` (matching `\d+`).
#[inline]
fn is_digits_only(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_digit())
}

/// Checks if string matches the float pattern `\d*\.?\d+` (digits with optional decimal point, no signs).
fn is_float_digits_only(s: &str) -> bool {
    if s.is_empty() || !s.ends_with(|c: char| c.is_ascii_digit()) {
        return false;
    }
    let mut has_digit = false;
    let mut has_dot = false;
    for c in s.chars() {
        if c.is_ascii_digit() {
            has_digit = true;
        } else if c == '.' && !has_dot {
            has_dot = true;
        } else {
            return false;
        }
    }
    has_digit
}

/// Non-allocating buffer writer adapter for `core::fmt::Write`.
struct BufWriter<'a> {
    buf: &'a mut [u8],
    len: usize,
}

impl<'a> BufWriter<'a> {
    #[inline]
    fn new(buf: &'a mut [u8]) -> Self {
        Self { buf, len: 0 }
    }

    #[inline]
    fn as_str(&self) -> Result<&str, core::fmt::Error> {
        core::str::from_utf8(&self.buf[..self.len]).map_err(|_| core::fmt::Error)
    }
}

impl<'a> Write for BufWriter<'a> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let bytes = s.as_bytes();
        if self.len + bytes.len() > self.buf.len() {
            return Err(fmt::Error);
        }
        self.buf[self.len..self.len + bytes.len()].copy_from_slice(bytes);
        self.len += bytes.len();
        Ok(())
    }
}

// ============================================================================
// ColorSpace Enum
// ============================================================================

/// Supported color spaces matching Three.js r186 and `ColorSpaces.js`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ColorSpace {
    /// Linear sRGB (Rec. 709 primaries with linear transfer function).
    /// Default working color space in Three.js.
    #[default]
    LinearSRGB,
    /// Standard sRGB (Rec. 709 primaries with sRGB transfer function).
    SRGB,
    /// Display P3 (P3 primaries with sRGB transfer function).
    DisplayP3,
    /// Linear Display P3 (P3 primaries with linear transfer function).
    LinearDisplayP3,
}

impl ColorSpace {
    /// Returns the canonical CSS / Three.js string identifier for this color space.
    #[inline]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::LinearSRGB => "srgb-linear",
            Self::SRGB => "srgb",
            Self::DisplayP3 => "display-p3",
            Self::LinearDisplayP3 => "display-p3-linear",
        }
    }
}

// ============================================================================
// Hsl Struct
// ============================================================================

/// HSL (Hue, Saturation, Lightness) color representation.
///
/// Component values are normalized in the range `[0.0, 1.0]`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Hsl {
    /// Hue in `[0.0, 1.0]` ($0^\circ \dots 360^\circ$).
    pub h: f64,
    /// Saturation in `[0.0, 1.0]`.
    pub s: f64,
    /// Lightness in `[0.0, 1.0]`.
    pub l: f64,
}

impl Hsl {
    /// Constructs a new `Hsl` instance.
    #[inline]
    pub const fn new(h: f64, s: f64, l: f64) -> Self {
        Self { h, s, l }
    }
}

// ============================================================================
// Color Struct
// ============================================================================

/// Three-component RGB color with `f64` semantics matching Three.js r186 `Color`.
///
/// Components `(r, g, b)` are stored in the linear working color space ([`ColorSpace::LinearSRGB`]).
#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Color {
    /// Red channel in linear working color space.
    pub r: f64,
    /// Green channel in linear working color space.
    pub g: f64,
    /// Blue channel in linear working color space.
    pub b: f64,
}

impl fmt::Debug for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Color")
            .field("r", &self.r)
            .field("g", &self.g)
            .field("b", &self.b)
            .finish()
    }
}

impl Default for Color {
    /// Constructs white `(1.0, 1.0, 1.0)` matching Three.js default constructor `new Color()`.
    #[inline]
    fn default() -> Self {
        Self::white()
    }
}

impl From<[f64; 3]> for Color {
    #[inline]
    fn from(arr: [f64; 3]) -> Self {
        Self::new(arr[0], arr[1], arr[2])
    }
}

impl From<(f64, f64, f64)> for Color {
    #[inline]
    fn from(tuple: (f64, f64, f64)) -> Self {
        Self::new(tuple.0, tuple.1, tuple.2)
    }
}

impl Color {
    /// Pure white constant `(1.0, 1.0, 1.0)`.
    pub const WHITE: Self = Self { r: 1.0, g: 1.0, b: 1.0 };
    /// Pure black constant `(0.0, 0.0, 0.0)`.
    pub const BLACK: Self = Self { r: 0.0, g: 0.0, b: 0.0 };

    /// Constructs a new color from raw components in linear working space.
    #[inline]
    pub const fn new(r: f64, g: f64, b: f64) -> Self {
        Self { r, g, b }
    }

    /// Constructs white `(1.0, 1.0, 1.0)`.
    #[inline]
    pub const fn white() -> Self {
        Self::WHITE
    }

    /// Constructs black `(0.0, 0.0, 0.0)`.
    #[inline]
    pub const fn black() -> Self {
        Self::BLACK
    }

    /// Sets the components of this color directly in the linear working color space.
    ///
    /// Matches Three.js r186 `Color.set(r, g, b)`.
    #[inline]
    pub fn set(&mut self, r: f64, g: f64, b: f64) -> &mut Self {
        self.r = r;
        self.g = g;
        self.b = b;
        self
    }

    /// Sets all three components of this color to a scalar value.
    ///
    /// Matches Three.js r186 `Color.setScalar(scalar)`.
    #[inline]
    pub fn set_scalar(&mut self, scalar: f64) -> &mut Self {
        self.r = scalar;
        self.g = scalar;
        self.b = scalar;
        self
    }

    /// Copies components from another color instance into this one.
    #[inline]
    pub fn copy(&mut self, color: &Self) -> &mut Self {
        self.r = color.r;
        self.g = color.g;
        self.b = color.b;
        self
    }

    /// Returns a new clone of this color.
    #[inline]
    pub fn clone(&self) -> Self {
        *self
    }

    // ------------------------------------------------------------------------
    // Color Space Conversions
    // ------------------------------------------------------------------------

    /// Converts this color from `source` color space to `target` color space in place.
    ///
    /// Matches Three.js r186 `ColorManagement.convert(color, source, target)`:
    /// 1. Converts source transfer to linear if necessary ([`srgb_to_linear`]).
    /// 2. Converts primaries between Rec. 709 and Display P3 via CIE XYZ if necessary.
    /// 3. Converts linear to target transfer if necessary ([`linear_to_srgb`]).
    pub fn convert(&mut self, source: ColorSpace, target: ColorSpace) -> &mut Self {
        if source == target {
            return self;
        }

        // 1. Source transfer to linear
        if source == ColorSpace::SRGB || source == ColorSpace::DisplayP3 {
            self.r = srgb_to_linear(self.r);
            self.g = srgb_to_linear(self.g);
            self.b = srgb_to_linear(self.b);
        }

        // 2. Primaries conversion
        let source_is_p3 = source == ColorSpace::DisplayP3 || source == ColorSpace::LinearDisplayP3;
        let target_is_p3 = target == ColorSpace::DisplayP3 || target == ColorSpace::LinearDisplayP3;

        if source_is_p3 != target_is_p3 {
            if source_is_p3 {
                // Linear Display P3 -> Linear sRGB (Rec. 709)
                self.apply_matrix3(&LINEAR_DISPLAY_P3_TO_XYZ);
                self.apply_matrix3(&XYZ_TO_LINEAR_REC709);
            } else {
                // Linear sRGB (Rec. 709) -> Linear Display P3
                self.apply_matrix3(&LINEAR_REC709_TO_XYZ);
                self.apply_matrix3(&XYZ_TO_LINEAR_DISPLAY_P3);
            }
        }

        // 3. Linear to target transfer
        if target == ColorSpace::SRGB || target == ColorSpace::DisplayP3 {
            self.r = linear_to_srgb(self.r);
            self.g = linear_to_srgb(self.g);
            self.b = linear_to_srgb(self.b);
        }

        self
    }

    /// Converts this color from an external `source` color space to the linear working color space.
    #[inline]
    pub fn color_space_to_working(&mut self, source: ColorSpace) -> &mut Self {
        self.convert(source, ColorSpace::LinearSRGB)
    }

    /// Converts this color from the linear working color space to an external `target` color space.
    #[inline]
    pub fn working_to_color_space(&mut self, target: ColorSpace) -> &mut Self {
        self.convert(ColorSpace::LinearSRGB, target)
    }

    /// Sets this color from RGB values interpreted in the specified `color_space`.
    ///
    /// Matches Three.js r186 `Color.setRGB(r, g, b, colorSpace)`.
    pub fn set_rgb(&mut self, r: f64, g: f64, b: f64, color_space: ColorSpace) -> &mut Self {
        self.r = r;
        self.g = g;
        self.b = b;
        self.color_space_to_working(color_space);
        self
    }

    /// Sets this color from RGB values in the default linear working color space ([`ColorSpace::LinearSRGB`]).
    #[inline]
    pub fn set_rgb_working(&mut self, r: f64, g: f64, b: f64) -> &mut Self {
        self.set_rgb(r, g, b, ColorSpace::LinearSRGB)
    }

    /// Sets this color from an integer hexadecimal value `0xRRGGBB` interpreted in `color_space`.
    ///
    /// Matches Three.js r186 `Color.setHex(hex, colorSpace)`.
    pub fn set_hex(&mut self, hex: u32, color_space: ColorSpace) -> &mut Self {
        let r = ((hex >> 16) & 0xFF) as f64 / 255.0;
        let g = ((hex >> 8) & 0xFF) as f64 / 255.0;
        let b = (hex & 0xFF) as f64 / 255.0;
        self.set_rgb(r, g, b, color_space)
    }

    /// Sets this color from an integer hexadecimal value `0xRRGGBB` in standard sRGB ([`ColorSpace::SRGB`]).
    #[inline]
    pub fn set_hex_srgb(&mut self, hex: u32) -> &mut Self {
        self.set_hex(hex, ColorSpace::SRGB)
    }

    /// Returns the hexadecimal representation of this color converted to `color_space`.
    ///
    /// Uses [`crate::jsnum::js_round`] matching Three.js r186 `Color.getHex(colorSpace)`:
    /// `Math.round(clamp(c * 255, 0, 255))`
    pub fn get_hex(&self, color_space: ColorSpace) -> u32 {
        let mut copy = *self;
        copy.working_to_color_space(color_space);
        let r = crate::jsnum::js_round(clamp(copy.r * 255.0, 0.0, 255.0)) as u32;
        let g = crate::jsnum::js_round(clamp(copy.g * 255.0, 0.0, 255.0)) as u32;
        let b = crate::jsnum::js_round(clamp(copy.b * 255.0, 0.0, 255.0)) as u32;
        (r << 16) | (g << 8) | b
    }

    /// Returns the hexadecimal representation in standard sRGB ([`ColorSpace::SRGB`]).
    #[inline]
    pub fn get_hex_srgb(&self) -> u32 {
        self.get_hex(ColorSpace::SRGB)
    }

    /// Formats the 6-character lowercase hexadecimal string into a caller-supplied `Write` destination.
    pub fn write_hex_string<W: Write>(&self, writer: &mut W, color_space: ColorSpace) -> fmt::Result {
        write!(writer, "{:06x}", self.get_hex(color_space))
    }

    /// Formats the 6-character lowercase hexadecimal string in standard sRGB into a caller-supplied `Write` destination.
    #[inline]
    pub fn write_hex_string_srgb<W: Write>(&self, writer: &mut W) -> fmt::Result {
        self.write_hex_string(writer, ColorSpace::SRGB)
    }

    /// Formats the 6-character lowercase hexadecimal string into a fixed stack array of bytes `[u8; 6]`.
    pub fn format_hex_string(&self, color_space: ColorSpace) -> [u8; 6] {
        let hex = self.get_hex(color_space);
        let mut buf = [0u8; 6];
        for i in 0..6 {
            let shift = (5 - i) * 4;
            let nibble = ((hex >> shift) & 0xF) as u8;
            buf[i] = if nibble < 10 {
                b'0' + nibble
            } else {
                b'a' + (nibble - 10)
            };
        }
        buf
    }

    /// Formats the 6-character lowercase hexadecimal string in standard sRGB into `[u8; 6]`.
    #[inline]
    pub fn format_hex_string_srgb(&self) -> [u8; 6] {
        self.format_hex_string(ColorSpace::SRGB)
    }

    /// Returns the 6-character lowercase hexadecimal string (available with feature `std`).
    #[cfg(feature = "std")]
    pub fn get_hex_string(&self, color_space: ColorSpace) -> String {
        let bytes = self.format_hex_string(color_space);
        // ASCII hex is guaranteed valid UTF-8
        String::from(core::str::from_utf8(&bytes).unwrap_or("000000"))
    }

    /// Returns the 6-character lowercase hexadecimal string in standard sRGB (available with feature `std`).
    #[cfg(feature = "std")]
    #[inline]
    pub fn get_hex_string_srgb(&self) -> String {
        self.get_hex_string(ColorSpace::SRGB)
    }

    // ------------------------------------------------------------------------
    // HSL Operations
    // ------------------------------------------------------------------------

    /// Sets this color from HSL values interpreted in `color_space`.
    ///
    /// Matches Three.js r186 `Color.setHSL(h, s, l, colorSpace)`.
    pub fn set_hsl(&mut self, h: f64, s: f64, l: f64, color_space: ColorSpace) -> &mut Self {
        let h = euclidean_modulo(h, 1.0);
        let s = clamp(s, 0.0, 1.0);
        let l = clamp(l, 0.0, 1.0);

        if s == 0.0 {
            self.r = l;
            self.g = l;
            self.b = l;
        } else {
            let p = if l <= 0.5 {
                l * (1.0 + s)
            } else {
                l + s - (l * s)
            };
            let q = 2.0 * l - p;

            self.r = hue2rgb(q, p, h + 1.0 / 3.0);
            self.g = hue2rgb(q, p, h);
            self.b = hue2rgb(q, p, h - 1.0 / 3.0);
        }

        self.color_space_to_working(color_space);
        self
    }

    /// Sets this color from HSL values in the working color space ([`ColorSpace::LinearSRGB`]).
    #[inline]
    pub fn set_hsl_working(&mut self, h: f64, s: f64, l: f64) -> &mut Self {
        self.set_hsl(h, s, l, ColorSpace::LinearSRGB)
    }

    /// Extracts HSL values from this color converted to `color_space`.
    ///
    /// Matches Three.js r186 `Color.getHSL(target, colorSpace)`.
    pub fn get_hsl(&self, color_space: ColorSpace) -> Hsl {
        let mut copy = *self;
        copy.working_to_color_space(color_space);

        let r = copy.r;
        let g = copy.g;
        let b = copy.b;

        let max = r.max(g).max(b);
        let min = r.min(g).min(b);

        let lightness = (min + max) / 2.0;

        if min == max {
            Hsl {
                h: 0.0,
                s: 0.0,
                l: lightness,
            }
        } else {
            let delta = max - min;
            let saturation = if lightness <= 0.5 {
                delta / (max + min)
            } else {
                delta / (2.0 - max - min)
            };

            let hue = if max == r {
                (g - b) / delta + if g < b { 6.0 } else { 0.0 }
            } else if max == g {
                (b - r) / delta + 2.0
            } else {
                (r - g) / delta + 4.0
            } / 6.0;

            Hsl {
                h: hue,
                s: saturation,
                l: lightness,
            }
        }
    }

    /// Extracts HSL values in the working color space ([`ColorSpace::LinearSRGB`]).
    #[inline]
    pub fn get_hsl_working(&self) -> Hsl {
        self.get_hsl(ColorSpace::LinearSRGB)
    }

    /// Adds the given HSL deltas to this color's HSL values in working color space.
    ///
    /// Matches Three.js r186 `Color.offsetHSL(h, s, l)`.
    pub fn offset_hsl(&mut self, h: f64, s: f64, l: f64) -> &mut Self {
        let current = self.get_hsl_working();
        self.set_hsl_working(current.h + h, current.s + s, current.l + l);
        self
    }

    /// Converts this color from `ColorSpace::SRGB` to `ColorSpace::LinearSRGB` in place.
    ///
    /// Matches Three.js r186 `Color.convertSRGBToLinear()`.
    #[inline]
    pub fn convert_srgb_to_linear(&mut self) -> &mut Self {
        self.r = srgb_to_linear(self.r);
        self.g = srgb_to_linear(self.g);
        self.b = srgb_to_linear(self.b);
        self
    }

    /// Converts this color from `ColorSpace::LinearSRGB` to `ColorSpace::SRGB` in place.
    ///
    /// Matches Three.js r186 `Color.convertLinearToSRGB()`.
    #[inline]
    pub fn convert_linear_to_srgb(&mut self) -> &mut Self {
        self.r = linear_to_srgb(self.r);
        self.g = linear_to_srgb(self.g);
        self.b = linear_to_srgb(self.b);
        self
    }

    // ------------------------------------------------------------------------
    // Named Color Keywords (Color.js:425-445)
    // ------------------------------------------------------------------------

    /// Sets this color from a CSS color name keyword (e.g. `'aliceblue'`, `'red'`).
    ///
    /// Case-insensitive search matches Three.js r186 `Color.setColorName(style, colorSpace)`.
    /// Returns [`StyleOutcome::Applied`] on success, or [`StyleOutcome::IgnoredInvalid`]
    /// leaving color unmodified if the name is not in the 140-color keyword dictionary.
    pub fn set_color_name(&mut self, name: &str, color_space: ColorSpace) -> StyleOutcome {
        for &(color_name, hex) in &COLOR_NAMES {
            if color_name.eq_ignore_ascii_case(name) {
                self.set_hex(hex, color_space);
                return StyleOutcome::Applied;
            }
        }
        StyleOutcome::IgnoredInvalid
    }

    /// Sets this color from a CSS color name keyword in standard sRGB.
    #[inline]
    pub fn set_color_name_srgb(&mut self, name: &str) -> StyleOutcome {
        self.set_color_name(name, ColorSpace::SRGB)
    }

    // ------------------------------------------------------------------------
    // CSS String Parsing and Formatting (Zero-Allocation)
    // ------------------------------------------------------------------------

    /// Sets this color from a CSS-style string matching Three.js r186 `Color.setStyle(style, colorSpace)`.
    ///
    /// Implements exact upstream regex and parsing semantics:
    /// - No outer trim: leading whitespace fails functional syntax `/^(\w+)\(([^\)]*)\)/`.
    /// - Functional names are case-sensitive: only lowercase `'rgb'`, `'rgba'`, `'hsl'`, `'hsla'` are accepted.
    /// - Trailing content after closing parenthesis `)` is tolerated matching `/^(\w+)\(([^\)]*)\)/`.
    /// - Integer `rgb()` and `rgb(%)` strictly require `\d+` digits (floating-point components are rejected).
    /// - Hex colors: `#rgb` (divided by 15.0) and `#rrggbb` (base 16 integer) require exact full-string match `/^\#([A-Fa-f\d]+)$/`.
    /// - Named colors: non-empty strings not matching functional or hex fall through to `set_color_name`.
    /// - Unrecognized or malformed strings leave the color components unchanged and return [`StyleOutcome::IgnoredInvalid`].
    /// - Discarded alpha < 1.0 returns [`StyleOutcome::AlphaIgnored`] matching upstream warning.
    pub fn set_style(&mut self, style: &str, color_space: ColorSpace) -> StyleOutcome {
        // 1. Functional matching: /^(\w+)\(([^\)]*)\)/
        // Must start with word characters [a-zA-Z0-9_]+ followed immediately by '('
        if let Some(open_paren) = style.find('(') {
            let prefix = &style[..open_paren];
            let is_word = !prefix.is_empty() && prefix.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
            if is_word {
                if let Some(close_paren) = style[open_paren + 1..].find(')') {
                    let close_idx = open_paren + 1 + close_paren;
                    let name = prefix;
                    let components = &style[open_paren + 1..close_idx];

                    return self.parse_functional_style(name, components, color_space);
                }
            }
        }

        // 2. Hex matching: /^\#([A-Fa-f\d]+)$/
        if let Some(hex_str) = style.strip_prefix('#') {
            if !hex_str.is_empty() && hex_str.chars().all(|c| c.is_ascii_hexdigit()) {
                if hex_str.len() == 3 {
                    let mut chars = hex_str.chars();
                    if let (Some(c0), Some(c1), Some(c2)) = (chars.next(), chars.next(), chars.next()) {
                        let d0 = c0.to_digit(16).unwrap_or(0);
                        let d1 = c1.to_digit(16).unwrap_or(0);
                        let d2 = c2.to_digit(16).unwrap_or(0);
                        self.set_rgb(d0 as f64 / 15.0, d1 as f64 / 15.0, d2 as f64 / 15.0, color_space);
                        return StyleOutcome::Applied;
                    }
                } else if hex_str.len() == 6 {
                    if let Ok(hex) = u32::from_str_radix(hex_str, 16) {
                        self.set_hex(hex, color_space);
                        return StyleOutcome::Applied;
                    }
                }
            }
            // Upstream: invalid hex warns and leaves color unchanged
            return StyleOutcome::IgnoredInvalid;
        }

        // 3. Fallback to named color keyword (Color.js:404)
        if !style.is_empty() {
            return self.set_color_name(style, color_space);
        }

        StyleOutcome::IgnoredInvalid
    }

    /// Internal helper parsing functional `rgb(...)`, `rgba(...)`, `hsl(...)`, `hsla(...)` components.
    fn parse_functional_style(&mut self, name: &str, components: &str, color_space: ColorSpace) -> StyleOutcome {
        // Parse up to 4 comma-separated tokens on the stack
        let mut parts = [""; 4];
        let mut count = 0;
        for part in components.split(',') {
            if count >= 4 {
                return StyleOutcome::IgnoredInvalid;
            }
            parts[count] = part.trim();
            count += 1;
        }
        if count != 3 && count != 4 {
            return StyleOutcome::IgnoredInvalid;
        }

        // Case-sensitive function name matching: Color.js switch (name)
        match name {
            "rgb" | "rgba" => {
                // Check integer form: /^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*(\d*\.?\d+)\s*)?$/
                let p0 = parts[0];
                let p1 = parts[1];
                let p2 = parts[2];

                if is_digits_only(p0) && is_digits_only(p1) && is_digits_only(p2) {
                    let mut outcome = StyleOutcome::Applied;
                    if count == 4 {
                        let alpha_str = parts[3];
                        if !is_float_digits_only(alpha_str) {
                            return StyleOutcome::IgnoredInvalid;
                        }
                        if let Ok(alpha) = alpha_str.parse::<f64>() {
                            if alpha < 1.0 {
                                outcome = StyleOutcome::AlphaIgnored;
                            }
                        } else {
                            return StyleOutcome::IgnoredInvalid;
                        }
                    }

                    let r_int = u32::from_str_radix(p0, 10).unwrap_or(0);
                    let g_int = u32::from_str_radix(p1, 10).unwrap_or(0);
                    let b_int = u32::from_str_radix(p2, 10).unwrap_or(0);

                    let r = r_int.min(255) as f64 / 255.0;
                    let g = g_int.min(255) as f64 / 255.0;
                    let b = b_int.min(255) as f64 / 255.0;

                    self.set_rgb(r, g, b, color_space);
                    return outcome;
                }

                // Check percentage form: /^\s*(\d+)\%\s*,\s*(\d+)\%\s*,\s*(\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/
                if let (Some(s0), Some(s1), Some(s2)) = (
                    p0.strip_suffix('%'),
                    p1.strip_suffix('%'),
                    p2.strip_suffix('%'),
                ) {
                    if is_digits_only(s0) && is_digits_only(s1) && is_digits_only(s2) {
                        let mut outcome = StyleOutcome::Applied;
                        if count == 4 {
                            let alpha_str = parts[3];
                            if !is_float_digits_only(alpha_str) {
                                return StyleOutcome::IgnoredInvalid;
                            }
                            if let Ok(alpha) = alpha_str.parse::<f64>() {
                                if alpha < 1.0 {
                                    outcome = StyleOutcome::AlphaIgnored;
                                }
                            } else {
                                return StyleOutcome::IgnoredInvalid;
                            }
                        }

                        let r_int = u32::from_str_radix(s0, 10).unwrap_or(0);
                        let g_int = u32::from_str_radix(s1, 10).unwrap_or(0);
                        let b_int = u32::from_str_radix(s2, 10).unwrap_or(0);

                        let r = r_int.min(100) as f64 / 100.0;
                        let g = g_int.min(100) as f64 / 100.0;
                        let b = b_int.min(100) as f64 / 100.0;

                        self.set_rgb(r, g, b, color_space);
                        return outcome;
                    }
                }

                StyleOutcome::IgnoredInvalid
            }
            "hsl" | "hsla" => {
                // Check HSL form: /^\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\%\s*,\s*(\d*\.?\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/
                let h_str = parts[0];
                let s_str = parts[1];
                let l_str = parts[2];

                if !is_float_digits_only(h_str) {
                    return StyleOutcome::IgnoredInvalid;
                }

                let s_val_str = match s_str.strip_suffix('%') {
                    Some(s) => s,
                    None => return StyleOutcome::IgnoredInvalid,
                };
                if !is_float_digits_only(s_val_str) {
                    return StyleOutcome::IgnoredInvalid;
                }

                let l_val_str = match l_str.strip_suffix('%') {
                    Some(s) => s,
                    None => return StyleOutcome::IgnoredInvalid,
                };
                if !is_float_digits_only(l_val_str) {
                    return StyleOutcome::IgnoredInvalid;
                }

                let mut outcome = StyleOutcome::Applied;
                if count == 4 {
                    let alpha_str = parts[3];
                    if !is_float_digits_only(alpha_str) {
                        return StyleOutcome::IgnoredInvalid;
                    }
                    if let Ok(alpha) = alpha_str.parse::<f64>() {
                        if alpha < 1.0 {
                            outcome = StyleOutcome::AlphaIgnored;
                        }
                    } else {
                        return StyleOutcome::IgnoredInvalid;
                    }
                }

                if let (Ok(h_deg), Ok(s_pct), Ok(l_pct)) = (
                    h_str.parse::<f64>(),
                    s_val_str.parse::<f64>(),
                    l_val_str.parse::<f64>(),
                ) {
                    let h = h_deg / 360.0;
                    let s = s_pct / 100.0;
                    let l = l_pct / 100.0;
                    self.set_hsl(h, s, l, color_space);
                    outcome
                } else {
                    StyleOutcome::IgnoredInvalid
                }
            }
            // Upstream default: warn and return this
            _ => StyleOutcome::IgnoredInvalid,
        }
    }

    /// Sets this color from a CSS-style string in standard sRGB.
    #[inline]
    pub fn set_style_srgb(&mut self, style: &str) -> StyleOutcome {
        self.set_style(style, ColorSpace::SRGB)
    }

    /// Formats this color as a CSS style string writing to a caller-supplied `Write` destination.
    ///
    /// For standard sRGB: writes `rgb(r,g,b)` using [`crate::jsnum::js_round`].
    /// For other color spaces: writes `color(<colorSpace> r.rrr g.ggg b.bbb)`.
    pub fn write_style<W: Write>(&self, writer: &mut W, color_space: ColorSpace) -> fmt::Result {
        let mut copy = *self;
        copy.working_to_color_space(color_space);

        if color_space != ColorSpace::SRGB {
            write!(
                writer,
                "color({} {:.3} {:.3} {:.3})",
                color_space.as_str(),
                copy.r,
                copy.g,
                copy.b
            )
        } else {
            let r = crate::jsnum::js_round(copy.r * 255.0) as i64;
            let g = crate::jsnum::js_round(copy.g * 255.0) as i64;
            let b = crate::jsnum::js_round(copy.b * 255.0) as i64;
            write!(writer, "rgb({},{},{})", r, g, b)
        }
    }

    /// Formats this color as a CSS style string in standard sRGB into a `Write` destination.
    #[inline]
    pub fn write_style_srgb<W: Write>(&self, writer: &mut W) -> fmt::Result {
        self.write_style(writer, ColorSpace::SRGB)
    }

    /// Formats this color as a CSS style string into a caller-supplied byte buffer without heap allocation.
    pub fn format_style<'a>(&self, buf: &'a mut [u8], color_space: ColorSpace) -> Result<&'a str, fmt::Error> {
        let mut writer = BufWriter::new(buf);
        self.write_style(&mut writer, color_space)?;
        writer.as_str()
    }

    /// Formats this color as a CSS style string in standard sRGB into a caller-supplied byte buffer.
    #[inline]
    pub fn format_style_srgb<'a>(&self, buf: &'a mut [u8]) -> Result<&'a str, fmt::Error> {
        self.format_style(buf, ColorSpace::SRGB)
    }

    /// Returns the CSS style string (available with feature `std`).
    #[cfg(feature = "std")]
    pub fn get_style(&self, color_space: ColorSpace) -> String {
        let mut s = String::with_capacity(48);
        let _ = self.write_style(&mut s, color_space);
        s
    }

    /// Returns the CSS style string in standard sRGB (available with feature `std`).
    #[cfg(feature = "std")]
    #[inline]
    pub fn get_style_srgb(&self) -> String {
        self.get_style(ColorSpace::SRGB)
    }

    // ------------------------------------------------------------------------
    // Interpolation (lerp, lerp_hsl)
    // ------------------------------------------------------------------------

    /// Linearly interpolates this color's RGB values toward `color` in working color space.
    ///
    /// Matches Three.js r186 `Color.lerp(color, alpha)`:
    /// `self.r += (color.r - self.r) * alpha`.
    pub fn lerp(&mut self, color: &Self, alpha: f64) -> &mut Self {
        self.r += (color.r - self.r) * alpha;
        self.g += (color.g - self.g) * alpha;
        self.b += (color.b - self.b) * alpha;
        self
    }

    /// Linearly interpolates between two colors and stores the result in `self`.
    ///
    /// Matches Three.js r186 `Color.lerpColors(color1, color2, alpha)`.
    pub fn lerp_colors(&mut self, color1: &Self, color2: &Self, alpha: f64) -> &mut Self {
        self.r = color1.r + (color2.r - color1.r) * alpha;
        self.g = color1.g + (color2.g - color1.g) * alpha;
        self.b = color1.b + (color2.b - color1.b) * alpha;
        self
    }

    /// Linearly interpolates this color toward `color` in HSL space in working color space.
    ///
    /// Matches Three.js r186 `Color.lerpHSL(color, alpha)`.
    pub fn lerp_hsl(&mut self, color: &Self, alpha: f64) -> &mut Self {
        let hsl_a = self.get_hsl_working();
        let hsl_b = color.get_hsl_working();

        let h = (1.0 - alpha) * hsl_a.h + alpha * hsl_b.h;
        let s = (1.0 - alpha) * hsl_a.s + alpha * hsl_b.s;
        let l = (1.0 - alpha) * hsl_a.l + alpha * hsl_b.l;

        self.set_hsl_working(h, s, l);
        self
    }

    // ------------------------------------------------------------------------
    // Arithmetic Operations
    // ------------------------------------------------------------------------

    /// Adds the RGB components of `color` to this color.
    pub fn add(&mut self, color: &Self) -> &mut Self {
        self.r += color.r;
        self.g += color.g;
        self.b += color.b;
        self
    }

    /// Adds `a` and `b` and stores the result in this color (`self = a + b`).
    pub fn add_colors(&mut self, a: &Self, b: &Self) -> &mut Self {
        self.r = a.r + b.r;
        self.g = a.g + b.g;
        self.b = a.b + b.b;
        self
    }

    /// Adds a scalar value `s` to all components of this color.
    pub fn add_scalar(&mut self, s: f64) -> &mut Self {
        self.r += s;
        self.g += s;
        self.b += s;
        self
    }

    /// Subtracts `color` from this color, clamping negative values to zero.
    ///
    /// Matches Three.js r186 `Color.sub`:
    /// `this.r = Math.max(0, this.r - color.r)`
    pub fn sub(&mut self, color: &Self) -> &mut Self {
        self.r = (self.r - color.r).max(0.0);
        self.g = (self.g - color.g).max(0.0);
        self.b = (self.b - color.b).max(0.0);
        self
    }

    /// Multiplies the RGB components of this color with `color`.
    pub fn multiply(&mut self, color: &Self) -> &mut Self {
        self.r *= color.r;
        self.g *= color.g;
        self.b *= color.b;
        self
    }

    /// Multiplies each component of this color by a scalar `s`.
    pub fn multiply_scalar(&mut self, s: f64) -> &mut Self {
        self.r *= s;
        self.g *= s;
        self.b *= s;
        self
    }

    /// Transforms this color by a 3x3 matrix in column-vector form.
    ///
    /// Matches Three.js r186 `Color.applyMatrix3`:
    /// ```text
    /// r' = e[0]*r + e[3]*g + e[6]*b
    /// g' = e[1]*r + e[4]*g + e[7]*b
    /// b' = e[2]*r + e[5]*g + e[8]*b
    /// ```
    pub fn apply_matrix3(&mut self, m: &Matrix3) -> &mut Self {
        let r = self.r;
        let g = self.g;
        let b = self.b;
        let e = &m.elements;

        self.r = e[0] * r + e[3] * g + e[6] * b;
        self.g = e[1] * r + e[4] * g + e[7] * b;
        self.b = e[2] * r + e[5] * g + e[8] * b;

        self
    }

    /// Sets this color's RGB components from a 3D vector.
    pub fn set_from_vector3(&mut self, v: &Vector3) -> &mut Self {
        self.r = v.x;
        self.g = v.y;
        self.b = v.z;
        self
    }

    /// Returns `true` if all components match `other` exactly.
    #[inline]
    pub fn equals(&self, other: &Self) -> bool {
        self.r == other.r && self.g == other.g && self.b == other.b
    }

    /// Writes components to an array `[r, g, b]`.
    #[inline]
    pub const fn to_array(&self) -> [f64; 3] {
        [self.r, self.g, self.b]
    }

    /// Sets components from a slice at the given offset.
    pub fn from_array(&mut self, array: &[f64], offset: usize) -> &mut Self {
        self.r = array[offset];
        self.g = array[offset + 1];
        self.b = array[offset + 2];
        self
    }
}
