/** Still WebP header preflight for EXT_texture_webp. No codec, DOM or GPU work.
 * https://developers.google.com/speed/webp/docs/riff_container
 * https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Vendor/EXT_texture_webp
 * Accept VP8, VP8L and VP8X with optional alpha/XMP. ICCP, EXIF and animation
 * need the retained source route; never substitute one animation frame or apply
 * implicit color/orientation changes. This checks extents and dimensions, not
 * entropy-coded pixel data. A real decoder must still validate the bitstream.
 */
import { GltfTextureError } from "./gltf_image_decode.mjs";
const fail = (message) => { throw new GltfTextureError("GLTF_TEXTURE_IMAGE", message); };

/** Signature only, for encoded asset caching; not a container validator. */
export function isGltfWebp(bytes) {
  return bytes instanceof Uint8Array && bytes.length >= 12 &&
    [82, 73, 70, 70].every((v, i) => bytes[i] === v) &&
    [87, 69, 66, 80].every((v, i) => bytes[i + 8] === v);
}

export function gltfWebpDimensions(bytes) {
  if (!(bytes instanceof Uint8Array) || !(bytes.buffer instanceof ArrayBuffer) ||
      bytes.buffer.resizable) fail("Expected fixed, unshared WebP bytes");
  let view;
  try { view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength); }
  catch { fail("Detached WebP bytes"); }
  if (!isGltfWebp(bytes) || bytes.length < 20 ||
      view.getUint32(4, true) + 8 !== bytes.length || bytes.length % 2)
    fail("Invalid WebP RIFF header or file extent");
  const u24 = (at) => bytes[at] + bytes[at + 1] * 256 + bytes[at + 2] * 65536;
  let canvas = null, picture = null, flags = 0, alpha = null, xmp = false, chunks = 0;
  for (let at = 12; at < bytes.length;) {
    if (++chunks > 65536)
      throw new GltfTextureError("GLTF_TEXTURE_LIMIT", "Too many WebP chunks");
    if (bytes.length - at < 8) fail("Truncated WebP chunk header");
    const tag = view.getUint32(at, true), size = view.getUint32(at + 4, true), start = at + 8;
    const padded = size + (size % 2);
    if (padded > bytes.length - start || (size % 2 && bytes[start + size] !== 0))
      fail("Invalid WebP chunk extent or padding");
    if (at === 12 && ![0x58385056, 0x20385056, 0x4c385056].includes(tag))
      fail("WebP must start with VP8, VP8L or VP8X");
    if (tag === 0x58385056) { // VP8X
      if (at !== 12 || size !== 10) fail("Invalid or duplicate WebP extended header");
      flags = bytes[start];
      if ((flags & 0xc1) || bytes[start + 1] || bytes[start + 2] || bytes[start + 3])
        fail("Reserved WebP extended header bits are nonzero");
      if (flags & 0x2a) fail("Animated, ICC or EXIF WebP needs the source route");
      canvas = { width: u24(start + 4) + 1, height: u24(start + 7) + 1 };
      if (canvas.width * canvas.height > 0xffffffff) fail("WebP canvas exceeds uint32 pixels");
    } else if (tag === 0x20385056 || tag === 0x4c385056) { // VP8 / VP8L
      if (picture) fail("Multiple WebP image bitstreams");
      if (tag === 0x20385056) {
        if (size < 10 || (bytes[start] & 1) || ((bytes[start] >> 1) & 7) > 3 ||
            !(bytes[start] & 16) || bytes[start + 3] !== 0x9d ||
            bytes[start + 4] !== 1 || bytes[start + 5] !== 0x2a)
          fail("Invalid WebP VP8 key-frame header");
        picture = { width: view.getUint16(start + 6, true) & 0x3fff,
          height: view.getUint16(start + 8, true) & 0x3fff };
        if (!picture.width || !picture.height) fail("Empty WebP dimensions");
        if (canvas && Boolean(flags & 16) !== Boolean(alpha))
          fail("WebP alpha flag disagrees with ALPH chunk");
        if (alpha && (alpha.control & 3) === 0 &&
            alpha.size !== 1 + picture.width * picture.height)
          fail("Raw WebP alpha does not fill the image");
      } else {
        if (size < 5 || bytes[start] !== 0x2f || alpha)
          fail("Invalid WebP lossless header or separate alpha");
        const word = view.getUint32(start + 1, true);
        if (word >>> 29) fail("Unsupported WebP lossless version");
        picture = { width: (word & 0x3fff) + 1, height: ((word >>> 14) & 0x3fff) + 1 };
      }
      if (canvas && (canvas.width !== picture.width || canvas.height !== picture.height))
        fail("WebP bitstream dimensions disagree with canvas");
    } else if (tag === 0x48504c41) { // ALPH
      if (!canvas || !(flags & 16) || alpha || picture || size < 1)
        fail("Invalid WebP alpha chunk order or declaration");
      const control = bytes[start];
      if ((control & 0xc0) || (control & 3) > 1 || ((control >> 4) & 3) > 1)
        fail("Unsupported WebP alpha encoding");
      alpha = { size, control };
    } else if ([0x4d494e41, 0x464d4e41, 0x50434349, 0x46495845].includes(tag)) {
      // Check actual chunks too; a false VP8X flag must not disguise animation.
      fail("Animated, ICC or EXIF WebP needs the source route");
    } else if (tag === 0x20504d58) { // XMP, not interpreted
      if (!canvas || !(flags & 4) || xmp) fail("Invalid WebP XMP declaration");
      xmp = true;
    }
    at = start + padded;
  }
  if (!picture || (Boolean(flags & 4) !== xmp)) fail("Incomplete WebP image");
  return picture;
}
