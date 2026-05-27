/**
 * Decode the env's `features_b64` payload into a Float32Array.
 *
 * Wire format: base64(float16, row-major, shape [16, 12]).
 * float16 is IEEE 754 half-precision. Browsers as of late 2024 / early
 * 2025 ship Float16Array (Chrome 118+, Firefox 122+). For older targets
 * we manually convert each uint16 to float32.
 */
const HAS_F16 =
  typeof (globalThis as Record<string, unknown>).Float16Array !== "undefined";

export type FeatureMatrix = {
  rows: number;
  cols: number;
  data: Float32Array; // length = rows * cols, row-major
};

export function decodeFeaturesB64(
  b64: string,
  rows: number,
  cols: number,
): FeatureMatrix {
  if (!b64) {
    return { rows, cols, data: new Float32Array(rows * cols) };
  }
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const view = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) view[i] = bin.charCodeAt(i);
  const u16 = new Uint16Array(buf);

  let f32: Float32Array;
  if (HAS_F16) {
    // The Float16Array constructor reads the buffer as half-precision.
    const F16 = (globalThis as unknown as {
      Float16Array: { new (b: ArrayBuffer): ArrayLike<number> };
    }).Float16Array;
    f32 = Float32Array.from(new F16(buf));
  } else {
    f32 = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) f32[i] = halfToFloat(u16[i]!);
  }
  return { rows, cols, data: f32 };
}

/**
 * Convert a uint16 IEEE 754 half-precision bit pattern to a JS number.
 * Reference: https://en.wikipedia.org/wiki/Half-precision_floating-point_format
 */
function halfToFloat(h: number): number {
  const sign = (h >> 15) & 0x1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;
  const sgn = sign === 0 ? 1 : -1;
  if (exp === 0) {
    if (frac === 0) return sgn * 0;
    // subnormal
    return sgn * Math.pow(2, -14) * (frac / 1024);
  }
  if (exp === 31) {
    if (frac === 0) return sign === 0 ? Infinity : -Infinity;
    return NaN;
  }
  return sgn * Math.pow(2, exp - 15) * (1 + frac / 1024);
}

