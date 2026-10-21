/** Native libwebp 1.6.0 / Pillow fixtures: 2x2 lossy, lossless, alpha and XMP.
 * RGBA values are independently decoded native pixels; lossy RGB is not guessed. */
export const webpFixtures = [
  {
    "name": "lossless",
    "base64": "UklGRiwAAABXRUJQVlA4TCAAAAAvAUAAEB8gEEjaH3qN+d9/gQAhjf8oQWYX4gcIEf0PAQ==",
    "rgba": [
      255,
      0,
      0,
      128,
      0,
      255,
      0,
      255,
      0,
      0,
      255,
      0,
      255,
      255,
      255,
      64
    ]
  },
  {
    "name": "lossy",
    "base64": "UklGRkAAAABXRUJQVlA4IDQAAACQAQCdASoCAAIAAMASJaQAAudPjMAA/vz+/+8b4H+m94/2hH/5Fec85tasNAuetb5HtgAA",
    "rgba": [
      82,
      81,
      82,
      255,
      149,
      149,
      149,
      255,
      29,
      29,
      29,
      255,
      253,
      253,
      253,
      255
    ]
  },
  {
    "name": "alpha",
    "base64": "UklGRnYAAABXRUJQVlA4WAoAAAAQAAAAAQAAAQAAQUxQSAUAAAAAgP8AQABWUDggSgAAADACAJ0BKgIAAgAAwBIloAJ0ugH4AAMg8dfAAP7/Gj+J/qP+Cr/xi/9E38O6oyGGKuO1LH/5rt8wA+YAf812/+jD+GH8MP8PYAAA",
    "rgba": [
      37,
      116,
      0,
      128,
      113,
      191,
      46,
      255,
      124,
      202,
      57,
      0,
      214,
      255,
      148,
      64
    ]
  },
  {
    "name": "xmp",
    "base64": "UklGRkoAAABXRUJQVlA4WAoAAAAUAAAAAQAAAQAAVlA4TCAAAAAvAUAAEB8gEEjaH3qN+d9/gQAhjf8oQWYX4gcIEf0PAVhNUCAEAAAAPHgvPg==",
    "rgba": [
      255,
      0,
      0,
      128,
      0,
      255,
      0,
      255,
      0,
      0,
      255,
      0,
      255,
      255,
      255,
      64
    ]
  }
];
export const webpBytes = fixture => Uint8Array.from(atob(fixture.base64), c => c.charCodeAt(0));
