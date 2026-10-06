/** GLSL ES 3.00 (ESSL) -> WGSL compiler for source ShaderMaterial programs.
 *
 * compileEsslProgram(vertexSource, fragmentSource, options) preprocesses, parses
 * and type-checks both stages, then emits two WGSL modules with a shared
 * resource interface and a reflection record:
 *
 *   group(0) binding(0)   one uniform buffer: every non-sampler uniform of both
 *                         stages, in a generated layout (reflection.uniforms
 *                         gives each leaf's byte offset; bool -> u32, scalar and
 *                         vec2 arrays -> vec4 elements, structs @align(16)).
 *   group(1) 2k / 2k+1    texture / sampler pair for the k-th sampler uniform
 *                         (arrays expand per element; indices must be constant,
 *                         as ESSL 3.00 already requires).
 *
 * Semantics preserved explicitly (not by regex substitution):
 * - GLSL global stage I/O becomes private variables copied by generated entry
 *   points, so helper functions keep reading/writing them as globals.
 * - Clip depth: gl_Position.z is remapped from GL [-w, w] to WebGPU [0, w]
 *   (option clipDepth:'gl', the default); gl_FragCoord is reported in GL window
 *   coordinates (bottom-left origin) using the internal f3d_target uniform
 *   (framebuffer width/height), and dFdy keeps GL's upward sign.
 * - Vector == / != reduce with all()/any() as in GLSL; `mod` uses GLSL's
 *   floor definition; out/inout parameters copy in/out through temporaries at
 *   the call, in source evaluation order; side-effecting ?:, &&, || and ++/--
 *   inside expressions are lowered to statements; swizzle stores are split per
 *   component; combined samplers split into texture + sampler, including as
 *   function parameters.
 * - Implicit-derivative sampling inside non-uniform control flow keeps the GL
 *   behavior (derivatives undefined in divergent quads, identical to ESSL) by
 *   disabling WGSL's derivative_uniformity diagnostic. This is the same
 *   semantics, not a hidden approximation.
 * Unsupported constructs (gl_PointCoord, integer-texture filtering, true switch
 * fallthrough, struct/array varyings, ...) throw EsslError; nothing is dropped.
 * No performance claim.
 */
import {preprocess, EsslError, essError} from './essl_preprocess.mjs';
import {parse} from './essl_parse.mjs';
export {EsslError};

// ---------------------------------------------------------------------------
// Types
const S = s => ({k: 's', s}), V = (s, n) => ({k: 'v', s, n}), M = (c, r) => ({k: 'm', c, r});
const VOID = {k: 'void'}, FLOAT = S('float'), INT = S('int'), UINT = S('uint'), BOOL = S('bool');
const SCALAR_PREFIX = {float: '', int: 'i', uint: 'u', bool: 'b'};
const WGSL_SCALAR = {float: 'f32', int: 'i32', uint: 'u32', bool: 'bool'};
const SAMPLERS = {
  sampler2D: {dim: '2d', s: 'float'}, sampler3D: {dim: '3d', s: 'float'}, samplerCube: {dim: 'cube', s: 'float'},
  sampler2DArray: {dim: '2d-array', s: 'float'}, sampler2DShadow: {dim: '2d', s: 'float', shadow: true},
  samplerCubeShadow: {dim: 'cube', s: 'float', shadow: true}, sampler2DArrayShadow: {dim: '2d-array', s: 'float', shadow: true},
  isampler2D: {dim: '2d', s: 'int'}, isampler3D: {dim: '3d', s: 'int'}, isamplerCube: {dim: 'cube', s: 'int'}, isampler2DArray: {dim: '2d-array', s: 'int'},
  usampler2D: {dim: '2d', s: 'uint'}, usampler3D: {dim: '3d', s: 'uint'}, usamplerCube: {dim: 'cube', s: 'uint'}, usampler2DArray: {dim: '2d-array', s: 'uint'},
};
function basicType(name) {
  if (name === 'void') return VOID;
  if (WGSL_SCALAR[name]) return S(name);
  let m = /^(b|i|u)?vec([234])$/.exec(name);
  if (m) return V({b: 'bool', i: 'int', u: 'uint'}[m[1]] ?? 'float', Number(m[2]));
  m = /^mat([234])(?:x([234]))?$/.exec(name);
  if (m) return M(Number(m[1]), Number(m[2] ?? m[1]));
  if (SAMPLERS[name]) return {k: 'smp', name, ...SAMPLERS[name]};
  return null;
}
const isScalar = t => t.k === 's', isVec = t => t.k === 'v', isMat = t => t.k === 'm';
const isNumeric = t => (t.k === 's' || t.k === 'v') && t.s !== 'bool';
const comps = t => t.k === 's' ? 1 : t.k === 'v' ? t.n : t.k === 'm' ? t.c * t.r : 0;
const scalarOf = t => t.k === 's' || t.k === 'v' ? t.s : t.k === 'm' ? 'float' : t.k === 'a' ? scalarOf(t.of) : null;
/** Inter-stage slots of one I/O variable: matrices by column, arrays by element. */
function ioSlots(t) {
  const plain = x => x.k === 's' || x.k === 'v';
  if (plain(t)) return [{suffix: '', access: '', type: t}];
  if (t.k === 'm') return Array.from({length: t.c}, (_, i) => ({suffix: `_c${i}`, access: `[${i}]`, type: V('float', t.r)}));
  if (t.k === 'a' && plain(t.of)) return Array.from({length: t.n}, (_, i) => ({suffix: `_e${i}`, access: `[${i}]`, type: t.of}));
  if (t.k === 'a' && t.of.k === 'm') return Array.from({length: t.n * t.of.c}, (_, k) => ({suffix: `_e${Math.floor(k / t.of.c)}_c${k % t.of.c}`, access: `[${Math.floor(k / t.of.c)}][${k % t.of.c}]`, type: V('float', t.of.r)}));
  essError(`Struct stage I/O is not supported (${glslName(t)})`);
}
const withScalar = (t, s) => t.k === 's' ? S(s) : t.k === 'v' ? V(s, t.n) : t;
const vecOf = (s, n) => n === 1 ? S(s) : V(s, n);
function same(a, b) {
  if (a.k !== b.k) return false;
  switch (a.k) {
    case 's': return a.s === b.s; case 'v': return a.s === b.s && a.n === b.n; case 'm': return a.c === b.c && a.r === b.r;
    case 'st': return a.name === b.name; case 'a': return a.n === b.n && same(a.of, b.of); case 'smp': return a.name === b.name;
    default: return true;
  }
}
function glslName(t) {
  switch (t.k) {
    case 's': return t.s; case 'v': return SCALAR_PREFIX[t.s] + 'vec' + t.n;
    case 'm': return t.c === t.r ? `mat${t.c}` : `mat${t.c}x${t.r}`;
    case 'st': return t.name; case 'a': return `${glslName(t.of)}[${t.n}]`; case 'smp': return t.name; default: return 'void';
  }
}
const typeKey = glslName;

// Every WGSL keyword, reserved word, predeclared type and builtin function this
// compiler may emit. User identifiers matching one gain a trailing underscore.
const RESERVED = new Set(`alias break case const const_assert continue continuing default diagnostic discard else enable false fn for if let loop override requires return struct switch true var while
NULL Self abstract active alignas alignof as asm asm_fragment async attribute auto await become binding_array cast catch class co_await co_return co_yield coherent column_major common compile compile_fragment concept const_cast consteval constexpr constinit crate debugger decltype delete demote demote_to_helper do dynamic_cast enum explicit export extends extern external fallthrough filter final finally friend from fxgroup get goto groupshared highp impl implements import inline instanceof interface layout lowp macro macro_rules match mediump meta mod module move mut mutable namespace new nil noexcept noinline nointerpolation noperspective null nullptr of operator package packoffset partition pass patch pixelfragment precise precision premerge priv protected pub public readonly ref regardless register reinterpret_cast require resource restrict self set shared sizeof smooth snorm static static_assert static_cast std subroutine super target template this thread_local throw trait try type typedef typeid typename typeof union unless unorm unsafe unsized use using varying virtual volatile wgsl where with writeonly yield
bool f16 f32 i32 u32 vec2 vec3 vec4 mat2x2 mat2x3 mat2x4 mat3x2 mat3x3 mat3x4 mat4x2 mat4x3 mat4x4 array atomic ptr sampler sampler_comparison texture_1d texture_2d texture_2d_array texture_3d texture_cube texture_cube_array texture_depth_2d texture_depth_cube texture_depth_2d_array texture_multisampled_2d texture_storage_2d
vec2f vec3f vec4f vec2i vec3i vec4i vec2u vec3u vec4u vec2h vec3h vec4h mat2x2f mat3x3f mat4x4f
abs acos acosh all any arrayLength asin asinh atan atan2 atanh bitcast ceil clamp cos cosh countLeadingZeros countOneBits countTrailingZeros cross degrees determinant distance dot dpdx dpdxCoarse dpdxFine dpdy dpdyCoarse dpdyFine exp exp2 extractBits faceForward firstLeadingBit firstTrailingBit floor fma fract frexp fwidth fwidthCoarse fwidthFine insertBits inverseSqrt ldexp length log log2 max min mix modf normalize pack2x16float pack2x16snorm pack2x16unorm pack4x8snorm pack4x8unorm pow quantizeToF16 radians reflect refract reverseBits round saturate select sign sin sinh smoothstep sqrt step storageBarrier tan tanh textureDimensions textureGather textureGatherCompare textureLoad textureNumLayers textureNumLevels textureNumSamples textureSample textureSampleBias textureSampleCompare textureSampleCompareLevel textureSampleGrad textureSampleLevel textureSampleBaseClampToEdge transpose trunc unpack2x16float unpack2x16snorm unpack2x16unorm unpack4x8snorm unpack4x8unorm workgroupBarrier main`.split(/\s+/));
const safe = name => (RESERVED.has(name) || name.startsWith('__') || name.startsWith('f3d_') ? name + '_' : name);

// ---------------------------------------------------------------------------
// Uniform-buffer representation and layout (WGSL uniform address space rules)
const roundUp = (k, n) => Math.ceil(n / k) * k;

/** Compile one program. Returns {vertex, fragment, reflection}. */
export function compileEsslProgram(vertexSource, fragmentSource, {defines = {}, clipDepth = 'gl'} = {}) {
  if (clipDepth !== 'gl' && clipDepth !== 'webgpu') essError('clipDepth must be gl or webgpu');
  const shared = {
    uniforms: new Map(),      // name -> {type, field}
    uniformOrder: [],
    samplers: new Map(),      // name -> {type, count, bindings: [k...]}
    samplerCount: 0,
    structs: new Map(),       // name -> {fields:[{name, type}], wname}
    needsTarget: false,
  };
  const units = {};
  for (const stage of ['vertex', 'fragment']) {
    const source = stage === 'vertex' ? vertexSource : fragmentSource;
    const pre = preprocess(source, {defines});
    units[stage] = new Unit(stage, parse(pre.tokens), shared, esslVersion(pre.version));
    units[stage].declare();
  }
  // Link stage interfaces before emitting either entry point.
  const varyings = [];
  for (const out of units.vertex.outputs) {
    if (out.builtin) continue;
    varyings.push({name: out.name, type: out.type, flat: out.flat, location: null});
  }
  let location = 0;
  for (const v of varyings) {
    const input = units.fragment.inputs.find(i => i.name === v.name);
    if (input && !same(input.type, v.type)) essError(`Varying ${v.name} type differs between stages`);
    v.flat = v.flat || input?.flat || scalarOf(v.type) !== 'float';
    v.location = location;
    location += ioSlots(v.type).length;
  }
  for (const input of units.fragment.inputs) {
    if (input.builtin) continue;
    if (!varyings.some(v => v.name === input.name)) essError(`Fragment input ${input.name} is not written by the vertex stage`);
  }
  const vertex = units.vertex.emit({varyings, clipDepth});
  const fragment = units.fragment.emit({varyings, clipDepth});
  return {vertex, fragment, reflection: reflect(shared, units, varyings)};
}

/** Compile one stage without a partner (corpus validation and diagnostics):
 * stage I/O locations follow declaration order. */
export function compileEsslStage(stage, source, {defines = {}, clipDepth = 'gl'} = {}) {
  const shared = {uniforms: new Map(), uniformOrder: [], samplers: new Map(), samplerCount: 0, structs: new Map(), needsTarget: false};
  const pre = preprocess(source, {defines});
  const unit = new Unit(stage, parse(pre.tokens), shared, esslVersion(pre.version));
  unit.declare();
  const io = stage === 'vertex' ? unit.outputs : unit.inputs;
  let location = 0;
  const varyings = io.map(v => { const r = {name: v.name, type: v.type, flat: v.flat || scalarOf(v.type) !== 'float', location}; location += ioSlots(v.type).length; return r; });
  return unit.emit({varyings, clipDepth});
}

function esslVersion(v) {
  if (v === null || /^100\b/.test(v)) return 100;
  if (/^300\s+es\b/.test(v)) return 300;
  essError(`Unsupported #version ${v}`);
}
// ESSL 1.00 texture builtins and their ESSL 3.00 equivalents.
const ESSL1_TEXTURE = {texture2D: 'texture', textureCube: 'texture', texture2DProj: 'textureProj', texture2DLod: 'textureLod',
  textureCubeLod: 'textureLod', texture2DProjLod: 'textureProjLod', texture2DLodEXT: 'textureLod', textureCubeLodEXT: 'textureLod',
  texture2DProjLodEXT: 'textureProjLod', texture2DGradEXT: 'textureGrad', textureCubeGradEXT: 'textureGrad'};

function reflect(shared, units, varyings) {
  // Layout must match Unit.uniformStruct(); computed once here.
  const fields = [];
  let offset = 0, align = 16;
  for (const name of shared.uniformOrder) {
    const u = shared.uniforms.get(name);
    const lay = layoutOf(u.repr, shared);
    offset = roundUp(lay.align, offset);
    fields.push({name, glslType: glslName(u.type), offset, size: lay.size, node: describe(u.type, u.repr, offset, shared)});
    offset += lay.size; align = Math.max(align, lay.align);
  }
  let targetOffset = null;
  if (shared.needsTarget) { offset = roundUp(16, offset); targetOffset = offset; offset += 16; }
  const size = Math.max(16, roundUp(16, offset));
  const textures = [];
  for (const [name, s] of shared.samplers) s.bindings.forEach((binding, element) => textures.push({
    name, element: s.count === null ? null : element, glslType: s.type.name, dimension: s.type.dim,
    sampleType: s.type.shadow ? 'depth' : s.type.s === 'float' ? 'float' : s.type.s === 'int' ? 'sint' : 'uint',
    comparison: !!s.type.shadow, textureBinding: binding * 2, samplerBinding: binding * 2 + 1}));
  return {
    uniformBufferSize: size, uniforms: fields, targetOffset,
    textures,
    attributes: units.vertex.inputs.filter(i => !i.builtin).map(i => ({name: i.name, glslType: glslName(i.type), location: i.location,
      locations: i.type.k === 'm' ? i.type.c : 1, components: i.type.k === 'm' ? i.type.r : comps(i.type), scalar: scalarOf(i.type)})),
    varyings: varyings.map(v => ({name: v.name, glslType: glslName(v.type), location: v.location, flat: v.flat})),
    outputs: units.fragment.outputs.filter(o => !o.builtin).map(o => ({name: o.name, glslType: glslName(o.type), location: o.location})),
    writesPointSize: units.vertex.used.has('gl_PointSize'),
    usesFragCoord: units.fragment.used.has('gl_FragCoord'),
    writesFragDepth: units.fragment.used.has('gl_FragDepth'),
    usesFrontFacing: units.fragment.used.has('gl_FrontFacing'),
    discards: units.fragment.discards,
    usesVertexIndex: units.vertex.used.has('gl_VertexID'), usesInstanceIndex: units.vertex.used.has('gl_InstanceID'),
  };
}

/** Uniform-address-space representation of a GLSL type. */
function uniformRepr(t, shared) {
  switch (t.k) {
    case 's': return t.s === 'bool' ? {k: 's', s: 'uint', fromBool: true} : t;
    case 'v': return t.s === 'bool' ? {k: 'v', s: 'uint', n: t.n, fromBool: true} : t;
    case 'm': return t;
    case 'st': return {k: 'st', name: t.name, uniform: needsStructConversion(t.name, shared)};
    case 'a': {
      const of = t.of;
      if ((of.k === 's') || (of.k === 'v' && of.n <= 2)) return {k: 'a', n: t.n, of: V(of.s === 'bool' ? 'uint' : of.s, 4), wrapped: of};
      if (of.k === 'v' && of.s === 'bool') return {k: 'a', n: t.n, of: V('uint', 4), wrapped: of};
      if (of.k === 'm' && roundUp(16, layoutOf(of, shared).size) !== layoutOf(of, shared).size)
        essError(`Uniform arrays of ${glslName(of)} are not supported`);
      if (of.k === 'a') essError('Arrays of arrays are not ESSL 3.00');
      return {k: 'a', n: t.n, of: uniformRepr(of, shared)};
    }
    default: essError(`Type ${glslName(t)} cannot be a uniform`);
  }
}
function needsStructConversion(name, shared) {
  const s = shared.structs.get(name);
  return s.fields.some(f => { const r = uniformRepr(f.type, shared); return r.fromBool || r.wrapped || r.uniform || (r.k === 'a' && r.of.uniform); });
}
function layoutOf(t, shared) {
  switch (t.k) {
    case 's': return {align: 4, size: 4};
    case 'v': return {align: t.n === 2 ? 8 : 16, size: t.n * 4};
    case 'm': { const col = layoutOf(V('float', t.r), shared); const stride = roundUp(col.align, col.size); return {align: col.align, size: stride * t.c, colStride: stride}; }
    case 'a': { const e = layoutOf(t.of, shared); const stride = roundUp(e.align, e.size); return {align: e.align, size: stride * t.n, stride}; }
    case 'st': {
      const s = shared.structs.get(t.name);
      let offset = 0, align = 16;
      for (const f of s.fields) {
        const r = t.uniform !== undefined || true ? uniformRepr(f.type, shared) : f.type;
        const l = layoutOf(r, shared);
        offset = roundUp(l.align, offset) + l.size; align = Math.max(align, l.align);
      }
      return {align, size: roundUp(align, offset)};
    }
    default: essError('Unsized layout');
  }
}
/** CPU packing plan node for a GLSL-typed value at a byte offset. */
function describe(t, repr, offset, shared) {
  switch (t.k) {
    case 's': case 'v': return {k: 'num', s: t.s, n: comps(t), offset};
    case 'm': { const l = layoutOf(t, shared); return {k: 'mat', c: t.c, r: t.r, offset, colStride: l.colStride}; }
    case 'a': {
      const l = layoutOf(repr, shared);
      return {k: 'array', n: t.n, stride: l.stride, offset, elem: describe(t.of, repr.wrapped ? repr.of : repr.of, 0, shared)};
    }
    case 'st': {
      const s = shared.structs.get(t.name), fields = [];
      let o = 0;
      for (const f of s.fields) {
        const r = uniformRepr(f.type, shared), l = layoutOf(r, shared);
        o = roundUp(l.align, o);
        fields.push({name: f.name, node: describe(f.type, r, offset + o, shared)});
        o += l.size;
      }
      return {k: 'struct', name: t.name, offset, fields};
    }
  }
}

// ---------------------------------------------------------------------------
const BUILTIN_VARS = {
  vertex: {gl_Position: {type: V('float', 4), w: 'f3d_Position', out: true}, gl_PointSize: {type: FLOAT, w: 'f3d_PointSize', out: true},
    gl_VertexID: {type: INT, w: 'f3d_VertexID'}, gl_InstanceID: {type: INT, w: 'f3d_InstanceID'}},
  fragment: {gl_FragCoord: {type: V('float', 4), w: 'f3d_FragCoord'}, gl_FrontFacing: {type: BOOL, w: 'f3d_FrontFacing'},
    gl_FragDepth: {type: FLOAT, w: 'f3d_FragDepth', out: true}},
};

class Unit {
  constructor(stage, decls, shared, version = 300) {
    Object.assign(this, {stage, decls, shared, version});
    this.globals = new Map();     // name -> symbol
    this.functions = new Map();   // name -> [{params, ret, wname, decl}]
    this.inputs = []; this.outputs = [];
    this.privates = [];           // module-scope var<private> lines
    this.consts = [];             // module-scope const lines
    this.deferredInit = [];       // global initializers evaluated at entry
    this.used = new Set();
    this.helpers = new Map();
    this.localStructs = [];
    this.temp = 0;
    this.discards = false;
  }
  // ---- declarations -------------------------------------------------------
  resolveType(spec, arraySpec = null, scope = null) {
    let t;
    if (spec.struct) { this.defineStruct(spec.struct, scope); t = {k: 'st', name: spec.struct.name}; }
    else {
      t = basicType(spec.name);
      if (!t) {
        if (!this.shared.structs.has(spec.name)) essError(`Unknown type ${spec.name}`);
        t = {k: 'st', name: spec.name};
      }
    }
    for (const a of [spec.array, arraySpec]) {
      if (a === null || a === undefined) continue;
      if (a === 'unsized') { t = {k: 'a', of: t, n: null}; continue; }
      const n = this.constInt(a, scope);
      if (!(n > 0)) essError('Array size must be a positive constant', a.line);
      t = {k: 'a', of: t, n};
    }
    return t;
  }
  defineStruct(st) {
    const existing = this.shared.structs.get(st.name);
    const fields = st.fields.map(f => ({name: f.name, type: this.resolveType(f.type, f.array)}));
    if (existing) {
      if (existing.fields.length !== fields.length || existing.fields.some((f, i) => f.name !== fields[i].name || !same(f.type, fields[i].type)))
        essError(`Struct ${st.name} differs between declarations`, st.line);
      return;
    }
    this.shared.structs.set(st.name, {fields, wname: safe(st.name)});
  }
  constInt(e, scope) {
    const v = this.constValue(e, scope);
    if (typeof v !== 'number' || !Number.isFinite(v)) essError('Expected a constant integer expression', e.line);
    return Math.trunc(v);
  }
  constValue(e, scope) {
    switch (e.k) {
      case 'num': return parseNumber(e.v).value;
      case 'bool': return e.v;
      case 'paren': return this.constValue(e.e, scope);
      case 'id': { const s = this.lookup(e.name, scope); if (s?.constValue !== undefined) return s.constValue; return undefined; }
      case 'unary': { const a = this.constValue(e.a, scope); if (typeof a !== 'number') return undefined; return e.op === '-' ? -a : e.op === '+' ? a : e.op === '~' ? ~a : undefined; }
      case 'bin': {
        const a = this.constValue(e.a, scope), b = this.constValue(e.b, scope);
        if (typeof a !== 'number' || typeof b !== 'number') return undefined;
        switch (e.op) { case '+': return a + b; case '-': return a - b; case '*': return a * b; case '/': return Number.isInteger(a) && Number.isInteger(b) ? Math.trunc(a / b) : a / b; case '%': return a % b; case '<<': return a << b; case '>>': return a >> b; }
        return undefined;
      }
      case 'ctor': if (e.args.length === 1 && ['int', 'uint', 'float'].includes(e.type.name)) { const a = this.constValue(e.args[0], scope); return typeof a === 'number' ? (e.type.name === 'float' ? a : Math.trunc(a)) : undefined; } return undefined;
      default: return undefined;
    }
  }
  lookup(name, scope) {
    for (let s = scope; s; s = s.parent) if (s.vars.has(name)) return s.vars.get(name);
    return this.globals.get(name);
  }
  declare() {
    for (const d of this.decls) {
      if (d.k === 'struct') { this.defineStruct(d.struct); continue; }
      if (d.k === 'ublock') { this.declareBlock(d); continue; }
      if (d.k === 'proto') { this.signature(d); continue; }
      if (d.k === 'func') { this.signature(d); continue; }
      if (d.k === 'var') this.declareGlobal(d);
    }
  }
  signature(d) {
    const params = d.params.map(p => ({q: p.q, name: p.name, type: this.resolveType(p.type, p.array)}));
    const ret = this.resolveType(d.ret);
    const list = this.functions.get(d.name) ?? [];
    let f = list.find(x => x.params.length === params.length && x.params.every((p, i) => same(p.type, params[i].type)));
    if (!f) { f = {name: d.name, params, ret, decl: null}; list.push(f); this.functions.set(d.name, list); }
    if (d.k === 'func') {
      if (f.decl) essError(`Function ${d.name} redefined`, d.line);
      f.decl = d; f.params = params;
    }
  }
  declareBlock(d) {
    if (this.stage && d.instance) {
      const sym = {kind: 'ublock', fields: new Map()};
      for (const f of d.fields) {
        const t = this.resolveType(f.type, f.array), name = `${d.instance}_${f.name}`;
        this.addUniform(name, t, d.line); sym.fields.set(f.name, name);
      }
      this.globals.set(d.instance, sym);
    } else for (const f of d.fields) {
      const t = this.resolveType(f.type, f.array);
      this.addUniform(f.name, t, d.line);
      this.globals.set(f.name, {kind: 'uniform', type: t, name: f.name});
    }
  }
  addUniform(name, type, line) {
    const existing = this.shared.uniforms.get(name);
    if (existing) { if (!same(existing.type, type)) essError(`Uniform ${name} has different types in the two stages`, line); return; }
    this.shared.uniforms.set(name, {type, repr: uniformRepr(type, this.shared), field: safe(name)});
    this.shared.uniformOrder.push(name);
  }
  declareGlobal(d) {
    const storage = d.q.storage === 'varying' ? (this.stage === 'vertex' ? 'out' : 'in') : d.q.storage;
    for (const v of d.list) {
      let type = this.resolveType(d.type, v.array);
      if (type.k === 'a' && type.n === null) {
        if (!v.init) essError('Unsized array needs an initializer', v.line);
        const n = v.init.k === 'ctor' ? v.init.args.length : null;
        if (!n) essError('Cannot size array', v.line);
        type = {k: 'a', of: type.of, n};
      }
      if (storage === 'uniform') {
        if (type.k === 'smp' || (type.k === 'a' && type.of.k === 'smp')) {
          const smp = type.k === 'smp' ? type : type.of;
          let s = this.shared.samplers.get(v.name);
          if (s && (s.type.name !== smp.name || s.count !== (type.k === 'a' ? type.n : null))) essError(`Sampler ${v.name} differs between stages`, v.line);
          if (!s) {
            const count = type.k === 'a' ? type.n : null;
            s = {type: smp, count, bindings: Array.from({length: count ?? 1}, () => this.shared.samplerCount++)};
            this.shared.samplers.set(v.name, s);
          }
          this.globals.set(v.name, {kind: 'sampler', type, name: v.name, sampler: s});
          continue;
        }
        this.addUniform(v.name, type, v.line);
        this.globals.set(v.name, {kind: 'uniform', type, name: v.name});
        continue;
      }
      if (type.k === 'smp' || (type.k === 'a' && type.of.k === 'smp')) essError('Samplers must be uniforms', v.line);
      const w = safe(v.name);
      if (storage === 'in' || storage === 'out') {
        if (type.k === 'st' || (type.k === 'a' && type.of.k === 'st')) essError(`Struct ${storage} variables are not supported (${v.name})`, v.line);
        if (storage === 'in' && this.stage === 'vertex' && type.k === 'a') essError('Vertex inputs cannot be arrays', v.line);
        if (type.k === 's' && type.s === 'bool' || type.k === 'v' && type.s === 'bool') essError('Boolean stage I/O is not ESSL', v.line);
        const io = {name: v.name, type, w, flat: d.q.interp === 'flat', location: d.q.layout?.location ?? null};
        (storage === 'in' ? this.inputs : this.outputs).push(io);
        this.privates.push(`var<private> ${w}: ${this.wgsl(type)};`);
        this.globals.set(v.name, {kind: 'global', type, w});
        continue;
      }
      if (storage === 'const') {
        if (!v.init) essError('const needs an initializer', v.line);
        const value = this.constValue(v.init, null);
        const sym = {kind: 'global', type, w, constValue: typeof value === 'number' ? value : undefined, isConst: true};
        this.globals.set(v.name, sym);
        sym.pendingInit = {decl: v, type, w};
        continue;
      }
      const sym = {kind: 'global', type, w};
      this.globals.set(v.name, sym);
      this.privates.push(`var<private> ${w}: ${this.wgsl(type)};`);
      if (v.init) this.deferredInit.push({decl: v, type, w});
    }
  }
  // ---- type strings -------------------------------------------------------
  wgsl(t, uniform = false) {
    if (uniform) return this.wgslUniform(uniformRepr(t, this.shared));
    switch (t.k) {
      case 's': return WGSL_SCALAR[t.s];
      case 'v': return `vec${t.n}<${WGSL_SCALAR[t.s]}>`;
      case 'm': return `mat${t.c}x${t.r}<f32>`;
      case 'st': return this.shared.structs.get(t.name).wname;
      case 'a': return `array<${this.wgsl(t.of)}, ${t.n}>`;
      case 'void': return '';
      default: essError(`No WGSL value type for ${glslName(t)}`);
    }
  }
  wgslUniform(r) {
    switch (r.k) {
      case 's': return WGSL_SCALAR[r.s];
      case 'v': return `vec${r.n}<${WGSL_SCALAR[r.s]}>`;
      case 'm': return `mat${r.c}x${r.r}<f32>`;
      case 'st': return this.shared.structs.get(r.name).wname + (r.uniform ? '_u' : '');
      case 'a': return `array<${this.wgslUniform(r.of)}, ${r.n}>`;
    }
  }
  // ---- emission -----------------------------------------------------------
  emit({varyings, clipDepth}) {
    const body = [];
    // Functions (user definitions only; prototypes merge into definitions).
    const groups = new Map();
    for (const [name, list] of this.functions) {
      const defined = list.filter(f => f.decl);
      defined.forEach((f, i) => { f.wname = defined.length > 1 ? `${safe(name)}_${f.params.map(p => typeKey(p.type).replace(/\W/g, '')).join('_') || 'v'}` : safe(name); });
      groups.set(name, defined);
    }
    // Global const initializers: WGSL module const when const-evaluable, else
    // a private initialized at entry before main(), in declaration order.
    const deferredConsts = [];
    for (const sym of this.globals.values()) {
      if (!sym.pendingInit) continue;
      const {decl, type, w} = sym.pendingInit;
      const ctx = this.context(null, null);
      const value = this.coerce(this.expr(decl.init, ctx), type, decl.line);
      if (!ctx.pre.length && this.constEvaluable(decl.init)) { sym.wgslConst = true; this.consts.push(`const ${w}: ${this.wgsl(type)} = ${value.c};`); }
      else { this.privates.push(`var<private> ${w}: ${this.wgsl(type)};`); deferredConsts.push({decl, type, w}); }
    }
    this.deferredInit = [...deferredConsts, ...this.deferredInit];
    const fnLines = [];
    for (const list of groups.values()) for (const f of list) fnLines.push(...this.func(f));
    const initLines = [];
    for (const {decl, type, w} of this.deferredInit) {
      const ctx = this.context(null, null);
      const value = this.coerce(this.expr(decl.init, ctx), type, decl.line);
      initLines.push(...ctx.pre, `${w} = ${value.c};`);
    }
    const mainList = groups.get('main');
    if (!mainList?.length) essError(`${this.stage} shader has no main()`);
    const entry = this.stage === 'vertex' ? this.vertexEntry(varyings, initLines, clipDepth) : this.fragmentEntry(varyings, initLines);
    // Structs used by this unit (all known structs; unused ones are harmless).
    for (const [name, s] of this.shared.structs) {
      const fields = s.fields.map((f, i) => `${i === 0 ? '@align(16) ' : ''}${safe(f.name)}: ${this.wgsl(f.type)},`);
      body.push(`struct ${s.wname} {`, ...fields.map(x => '  ' + x), '};');
      if (needsStructConversion(name, this.shared)) {
        body.push(`struct ${s.wname}_u {`, ...s.fields.map((f, i) => `  ${i === 0 ? '@align(16) ' : ''}${safe(f.name)}: ${this.wgsl(f.type, true)},`), '};');
        body.push(`fn f3d_from_u_${s.wname}(v: ${s.wname}_u) -> ${s.wname} {`, `  var r: ${s.wname};`,
          ...s.fields.map(f => `  r.${safe(f.name)} = ${this.fromUniform(`v.${safe(f.name)}`, f.type)};`), '  return r;', '}');
      }
    }
    const uniformFields = this.shared.uniformOrder.map(n => { const u = this.shared.uniforms.get(n); return `  ${u.field}: ${this.wgslUniform(u.repr)},`; });
    if (this.shared.needsTarget) uniformFields.push('  f3d_target: vec4<f32>,');
    if (!uniformFields.length) uniformFields.push('  f3d_pad: vec4<f32>,');
    body.push('struct F3DUniforms {', ...uniformFields, '};', '@group(0) @binding(0) var<uniform> f3d_u: F3DUniforms;');
    for (const [, s] of this.shared.samplers) {
      // Declared in both modules so binding numbers are identical everywhere.
    }
    for (const [name, s] of this.shared.samplers) s.bindings.forEach((b, i) => {
      const base = s.count === null ? safe(name) : `${safe(name)}_${i}`;
      body.push(`@group(1) @binding(${b * 2}) var ${base}_texture: ${textureType(s.type)};`,
        `@group(1) @binding(${b * 2 + 1}) var ${base}_sampler: ${s.type.shadow ? 'sampler_comparison' : 'sampler'};`);
    });
    const out = ['diagnostic(off, derivative_uniformity);', ...body, ...this.consts, ...this.privates, ...this.localStructs,
      ...[...this.helpers.values()], ...fnLines, ...entry];
    return out.join('\n') + '\n';
  }
  constEvaluable(e) {
    switch (e.k) {
      case 'num': case 'bool': return true;
      case 'paren': return this.constEvaluable(e.e);
      case 'id': return !!this.globals.get(e.name)?.wgslConst;
      case 'unary': return this.constEvaluable(e.a);
      case 'bin': return this.constEvaluable(e.a) && this.constEvaluable(e.b);
      case 'ctor': return e.args.every(a => this.constEvaluable(a));
      case 'field': return this.constEvaluable(e.a);
      case 'index': return this.constEvaluable(e.a) && this.constEvaluable(e.index);
      case 'call': return ['sin', 'cos', 'sqrt', 'pow', 'normalize', 'abs', 'min', 'max', 'clamp', 'exp', 'log', 'exp2', 'log2', 'radians', 'degrees', 'tan', 'floor', 'ceil', 'fract', 'length', 'dot', 'cross'].includes(e.name) && e.args.every(a => this.constEvaluable(a));
      default: return false;
    }
  }
  vertexEntry(varyings, initLines, clipDepth) {
    const fields = [], copy = [];
    let location = 0;
    for (const i of this.inputs) {
      if (i.location !== null) location = i.location;
      i.location = location;
      if (i.type.k === 'm') {
        for (let c = 0; c < i.type.c; c++) fields.push(`@location(${location + c}) ${i.w}_c${c}: vec${i.type.r}<f32>,`);
        copy.push(`${i.w} = mat${i.type.c}x${i.type.r}<f32>(${Array.from({length: i.type.c}, (_, c) => `input.${i.w}_c${c}`).join(', ')});`);
        location += i.type.c;
      } else { fields.push(`@location(${location}) ${i.w}: ${this.wgsl(i.type)},`); copy.push(`${i.w} = input.${i.w};`); location++; }
    }
    if (this.used.has('gl_VertexID')) { fields.push('@builtin(vertex_index) f3d_vertex_index: u32,'); copy.push('f3d_VertexID = i32(input.f3d_vertex_index);'); }
    if (this.used.has('gl_InstanceID')) { fields.push('@builtin(instance_index) f3d_instance_index: u32,'); copy.push('f3d_InstanceID = i32(input.f3d_instance_index);'); }
    const outFields = ['@builtin(position) f3d_position: vec4<f32>,'], outCopy = [];
    for (const v of varyings) {
      const o = this.outputs.find(x => x.name === v.name);
      const interp = v.flat ? ' @interpolate(flat)' : '';
      ioSlots(v.type).forEach((slot, k) => {
        outFields.push(`@location(${v.location + k})${interp} ${o.w}${slot.suffix}: ${this.wgsl(slot.type)},`);
        outCopy.push(`out.${o.w}${slot.suffix} = ${o.w}${slot.access};`);
      });
    }
    const lines = [];
    if (fields.length) lines.push('struct F3DVertexIn {', ...fields.map(f => '  ' + f), '};');
    lines.push('struct F3DVertexOut {', ...outFields.map(f => '  ' + f), '};');
    lines.push(`@vertex fn f3d_vertex(${fields.length ? 'input: F3DVertexIn' : ''}) -> F3DVertexOut {`,
      ...copy.map(x => '  ' + x), ...initLines.map(x => '  ' + x), `  ${this.mainName()}();`, '  var out: F3DVertexOut;',
      clipDepth === 'gl' ? '  out.f3d_position = vec4<f32>(f3d_Position.xy, (f3d_Position.z + f3d_Position.w) * 0.5, f3d_Position.w);'
        : '  out.f3d_position = f3d_Position;',
      ...outCopy.map(x => '  ' + x), '  return out;', '}');
    return lines;
  }
  fragmentEntry(varyings, initLines) {
    const fields = [], copy = [];
    for (const i of this.inputs) {
      const v = varyings.find(x => x.name === i.name);
      i.location = v.location;
      const interp = v.flat ? ' @interpolate(flat)' : '';
      ioSlots(i.type).forEach((slot, k) => {
        fields.push(`@location(${v.location + k})${interp} ${i.w}${slot.suffix}: ${this.wgsl(slot.type)},`);
        copy.push(`${i.w}${slot.access} = input.${i.w}${slot.suffix};`);
      });
    }
    if (this.used.has('gl_FragCoord')) {
      fields.push('@builtin(position) f3d_frag_position: vec4<f32>,');
      // GL window coordinates: bottom-left origin of the framebuffer.
      copy.push('f3d_FragCoord = vec4<f32>(input.f3d_frag_position.x, f3d_u.f3d_target.y - input.f3d_frag_position.y, input.f3d_frag_position.z, input.f3d_frag_position.w);');
    }
    if (this.used.has('gl_FrontFacing')) { fields.push('@builtin(front_facing) f3d_front_facing: bool,'); copy.push('f3d_FrontFacing = input.f3d_front_facing;'); }
    const outFields = [], outCopy = [];
    let location = 0;
    for (const o of this.outputs) {
      if (o.location === null) o.location = location;
      location = o.location + 1;
      outFields.push(`@location(${o.location}) ${o.w}: ${this.wgsl(o.type)},`); outCopy.push(`out.${o.w} = ${o.w};`);
    }
    if (this.used.has('gl_FragDepth')) { outFields.push('@builtin(frag_depth) f3d_frag_depth: f32,'); outCopy.push('out.f3d_frag_depth = f3d_FragDepth;'); }
    const lines = [];
    if (fields.length) lines.push('struct F3DFragmentIn {', ...fields.map(f => '  ' + f), '};');
    if (outFields.length) lines.push('struct F3DFragmentOut {', ...outFields.map(f => '  ' + f), '};');
    lines.push(`@fragment fn f3d_fragment(${fields.length ? 'input: F3DFragmentIn' : ''})${outFields.length ? ' -> F3DFragmentOut' : ''} {`,
      ...copy.map(x => '  ' + x), ...initLines.map(x => '  ' + x), `  ${this.mainName()}();`,
      ...(outFields.length ? ['  var out: F3DFragmentOut;', ...outCopy.map(x => '  ' + x), '  return out;'] : []), '}');
    return lines;
  }
  mainName() { return this.functions.get('main').find(f => f.decl).wname; }

  // ---- functions & statements --------------------------------------------
  context(fn, scope) { return {fn, scope, pre: []}; }
  func(f) {
    const scope = {vars: new Map(), parent: null};
    const params = [];
    for (const p of f.params) {
      if (p.name === null) continue;
      const w = safe(p.name);
      if (p.type.k === 'smp') {
        params.push(`${w}_texture: ${textureType(p.type)}`, `${w}_sampler: ${p.type.shadow ? 'sampler_comparison' : 'sampler'}`);
        scope.vars.set(p.name, {kind: 'samplerParam', type: p.type, w});
      } else if (p.q === 'out' || p.q === 'inout') {
        params.push(`${w}: ptr<function, ${this.wgsl(p.type)}>`);
        scope.vars.set(p.name, {kind: 'pparam', type: p.type, w});
      } else {
        // Parameters are mutable locals in GLSL: copy into a var.
        params.push(`${w}_in: ${this.wgsl(p.type)}`);
        scope.vars.set(p.name, {kind: 'local', type: p.type, w, paramCopy: true});
      }
    }
    const ret = f.ret.k === 'void' ? '' : ` -> ${this.wgsl(f.ret)}`;
    const ctx = this.context(f, scope);
    const copies = [...scope.vars.values()].filter(v => v.paramCopy).map(v => `var ${v.w}: ${this.wgsl(v.type)} = ${v.w}_in;`);
    const body = this.statements(f.decl.body.list, ctx, scope);
    // WGSL requires a return at the end of non-void functions reached by flow analysis.
    const tail = f.ret.k !== 'void' && !endsWithReturn(body) ? [`return ${zero(this, f.ret)};`] : [];
    return [`fn ${f.wname}(${params.join(', ')})${ret} {`, ...[...copies, ...body, ...tail].map(x => '  ' + x), '}'];
  }
  statements(list, ctx, scope) {
    const out = [];
    for (const s of list) out.push(...this.statement(s, ctx, scope));
    return out;
  }
  block(s, ctx, scope) {
    const inner = {vars: new Map(), parent: scope};
    if (s.k === 'block') return this.statements(s.list, ctx, s.scoped ? inner : scope);
    return this.statement(s, ctx, inner);
  }
  hoisted(ctx, fn) {
    const saved = ctx.pre; ctx.pre = [];
    const result = fn();
    const pre = ctx.pre; ctx.pre = saved;
    return {pre, result};
  }
  statement(s, ctx, scope) {
    ctx.scope = scope;
    switch (s.k) {
      case 'empty': return [];
      case 'struct': {
        this.defineStruct(s.struct);
        return [];
      }
      case 'block': {
        if (!s.scoped) return this.statements(s.list, ctx, scope);
        const inner = this.statements(s.list, ctx, {vars: new Map(), parent: scope});
        return ['{', ...inner.map(x => '  ' + x), '}'];
      }
      case 'var': return this.localVar(s, ctx, scope);
      case 'expr': {
        const {pre, result} = this.hoisted(ctx, () => this.exprStatement(s.e, ctx));
        return [...pre, ...result];
      }
      case 'if': {
        const {pre, result} = this.hoisted(ctx, () => this.coerceBool(this.expr(s.c, ctx), s.line));
        const a = this.block(s.a, ctx, scope), b = s.b ? this.block(s.b, ctx, scope) : null;
        return [...pre, `if (${result.c}) {`, ...a.map(x => '  ' + x), ...(b ? ['} else {', ...b.map(x => '  ' + x)] : []), '}'];
      }
      case 'while': {
        const inner = {vars: new Map(), parent: scope};
        const {pre, result} = this.hoisted(ctx, () => this.condition(s.c, ctx, inner));
        const body = this.block(s.body, ctx, inner);
        return ['loop {', ...[...pre, ...result.decl, `if (!(${result.c})) { break; }`, ...body].map(x => '  ' + x), '}'];
      }
      case 'do': {
        const body = this.block(s.body, ctx, scope);
        const {pre, result} = this.hoisted(ctx, () => this.coerceBool(this.expr(s.c, ctx), s.line));
        return ['loop {', ...body.map(x => '  ' + x), '  continuing {', ...[...pre, `break if !(${result.c});`].map(x => '    ' + x), '  }', '}'];
      }
      case 'for': {
        const inner = {vars: new Map(), parent: scope};
        ctx.scope = inner;
        const init = s.init ? this.statement(s.init, ctx, inner) : [];
        let cond = {pre: [], result: {c: 'true', decl: []}};
        if (s.cond) cond = this.hoisted(ctx, () => this.condition(s.cond, ctx, inner));
        const step = s.step ? this.hoisted(ctx, () => this.exprStatement(s.step, ctx)) : {pre: [], result: []};
        const body = this.block(s.body, ctx, inner);
        const stepLines = [...step.pre, ...step.result];
        return ['{', ...init.map(x => '  ' + x), '  loop {',
          ...[...cond.pre, ...cond.result.decl, `if (!(${cond.result.c})) { break; }`, ...body].map(x => '    ' + x),
          ...(stepLines.length ? ['    continuing {', ...stepLines.map(x => '      ' + x), '    }'] : []), '  }', '}'];
      }
      case 'switch': return this.switchStatement(s, ctx, scope);
      case 'break': return ['break;'];
      case 'continue': return ['continue;'];
      case 'discard': this.discards = true; return ['discard;'];
      case 'return': {
        if (!s.e) return ['return;'];
        const {pre, result} = this.hoisted(ctx, () => this.coerce(this.expr(s.e, ctx), ctx.fn.ret, s.line));
        return [...pre, `return ${result.c};`];
      }
      default: essError(`Unsupported statement ${s.k}`, s.line);
    }
  }
  condition(c, ctx, scope) {
    if (c.k === 'condDecl') {
      const type = this.resolveType(c.type, null, scope), w = this.local(c.name, type, scope);
      const v = this.coerce(this.expr(c.init, ctx), type, c.line);
      return {c: w, decl: [`var ${w}: ${this.wgsl(type)} = ${v.c};`]};
    }
    return {...this.coerceBool(this.expr(c, ctx), c.line), decl: []};
  }
  switchStatement(s, ctx, scope) {
    const {pre, result} = this.hoisted(ctx, () => this.expr(s.e, ctx));
    if (!(result.t.k === 's' && (result.t.s === 'int' || result.t.s === 'uint'))) essError('switch needs an integer selector', s.line);
    // Merge empty labels; require each non-empty group to end in a jump.
    const groups = [];
    let labels = [];
    s.cases.forEach((c, i) => {
      labels.push(c.value);
      if (!c.body.length && i < s.cases.length - 1) return;
      groups.push({labels, body: c.body}); labels = [];
    });
    const lines = [...pre, `switch (${result.c}) {`];
    let hasDefault = false;
    groups.forEach((g, i) => {
      let body = g.body;
      const last = body.at(-1);
      if (i < groups.length - 1 && !(last && ['break', 'return', 'discard', 'continue'].includes(last.k)))
        essError('switch fallthrough between non-empty cases is not supported', s.line);
      if (last?.k === 'break') body = body.slice(0, -1);
      const values = g.labels.filter(v => v !== null).map(v => this.expr(v, ctx).c);
      const isDefault = g.labels.includes(null); hasDefault ||= isDefault;
      const label = isDefault ? (values.length ? `case ${values.join(', ')}, default` : 'default') : `case ${values.join(', ')}`;
      const inner = this.statements(body, ctx, {vars: new Map(), parent: scope});
      lines.push(`  ${label}: {`, ...inner.map(x => '    ' + x), '  }');
    });
    if (!hasDefault) lines.push('  default: {}');
    lines.push('}');
    return lines;
  }
  local(name, type, scope) {
    let w = safe(name);
    // Avoid WGSL redeclaration in the same scope after GLSL shadowing a parameter.
    scope.vars.set(name, {kind: 'local', type, w});
    return w;
  }
  localVar(s, ctx, scope) {
    const out = [];
    for (const v of s.list) {
      if (s.type.struct) this.defineStruct(s.type.struct);
      let type = this.resolveType(s.type, v.array, scope);
      if (type.k === 'a' && type.n === null) {
        if (v.init?.k !== 'ctor') essError('Cannot size array', v.line);
        type = {k: 'a', of: type.of, n: v.init.args.length};
      }
      let init = null;
      if (v.init) {
        const {pre, result} = this.hoisted(ctx, () => this.coerce(this.expr(v.init, ctx), type, v.line));
        out.push(...pre); init = result.c;
      }
      const w = this.local(v.name, type, scope);
      const sym = scope.vars.get(v.name);
      if (s.q.storage === 'const') {
        const value = this.constValue(v.init, scope);
        if (typeof value === 'number') sym.constValue = value;
        out.push(`let ${w}: ${this.wgsl(type)} = ${init};`);
        sym.readonly = true;
      } else out.push(`var ${w}: ${this.wgsl(type)}${init !== null ? ` = ${init}` : ''};`);
    }
    return out;
  }
  /** Expression statement: assignments and increments become WGSL statements. */
  exprStatement(e, ctx) {
    switch (e.k) {
      case 'paren': return this.exprStatement(e.e, ctx);
      case 'seq': return e.list.flatMap(x => { const r = this.exprStatement(x, ctx); const pre = ctx.pre.splice(0); return [...pre, ...r]; });
      case 'assign': return this.assign(e.target, e.op, e.value, ctx, e.line);
      case 'preinc': case 'postinc': return this.assign(e.a, e.op === '++' ? '+=' : '-=', {k: 'num', v: '1', line: e.line, inc: true}, ctx, e.line);
      default: {
        const r = this.expr(e, ctx);
        if (!r.c) return [];
        if (e.k === 'call' && r.t.k === 'void') return [`${r.c};`];
        if (e.k === 'call' && this.isUserCall(e)) return [`_ = ${r.c};`];
        return []; // pure expression statement: no effect
      }
    }
  }
  isUserCall(e) { return e.k === 'call' && this.functions.has(e.name); }
  /** Emit an assignment (simple or compound) including swizzle stores. */
  assign(target, op, valueExpr, ctx, line) {
    const lv = this.lvalue(target, ctx);
    let value = valueExpr.inc ? {c: lv.t.k === 's' && lv.t.s === 'float' || scalarOf(lv.t) === 'float' ? '1.0' : scalarOf(lv.t) === 'uint' ? '1u' : '1', t: S(scalarOf(lv.t))} : this.expr(valueExpr, ctx);
    const binop = op === '=' ? null : op.slice(0, -1);
    if (binop === null) value = this.coerce(value, lv.t, line);
    if (lv.swizzle && lv.swizzle.length > 1) {
      const t = this.tmp();
      const lines = [`let ${t} = ${value.c};`];
      lv.swizzle.forEach((comp, i) => {
        const src = value.t.k === 'v' ? `${t}.${'xyzw'[i]}` : t;
        lines.push(binop === null ? `${lv.base}.${comp} = ${src};`
          : `${lv.base}.${comp} = ${this.binaryCode(binop, {c: `${lv.base}.${comp}`, t: S(scalarOf(lv.t))}, {c: src, t: S(scalarOf(value.t))}, line).c};`);
      });
      return ['{', ...lines.map(x => '  ' + x), '}'];
    }
    if (binop === null) return [`${lv.c} = ${value.c};`];
    const r = this.binaryCode(binop, {c: lv.c, t: lv.t}, value, line);
    return [`${lv.c} = ${r.c};`];
  }
  lvalue(e, ctx) {
    switch (e.k) {
      case 'paren': return this.lvalue(e.e, ctx);
      case 'id': {
        const s = this.lookup(e.name, ctx.scope);
        if (this.version === 100 && this.stage === 'fragment' && (e.name === 'gl_FragColor' || e.name === 'gl_FragData') && !s)
          return this.essl1FragColor(e.name, e.line);
        const b = BUILTIN_VARS[this.stage][e.name];
        if (b) { this.used.add(e.name); this.builtinPrivate(e.name, b); return {c: b.w, t: b.type}; }
        if (!s) essError(`Undeclared identifier ${e.name}`, e.line);
        if (s.kind === 'pparam') return {c: `(*${s.w})`, t: s.type};
        if (s.kind === 'local' || s.kind === 'global') { if (s.readonly || s.isConst) essError(`Cannot assign to const ${e.name}`, e.line); return {c: s.w, t: s.type}; }
        essError(`${e.name} is not assignable`, e.line);
      }
      case 'index': {
        const base = this.lvalue(e.a, ctx), index = this.indexCode(e.index, ctx);
        return {c: `${base.c}[${index}]`, t: elementType(base.t, e.line)};
      }
      case 'field': {
        const base = this.lvalue(e.a, ctx);
        if (base.t.k === 'st') {
          const f = this.shared.structs.get(base.t.name).fields.find(x => x.name === e.name);
          if (!f) essError(`No field ${e.name}`, e.line);
          return {c: `${base.c}.${safe(e.name)}`, t: f.type};
        }
        if (base.t.k === 'v') {
          const sw = swizzle(e.name, base.t.n, e.line);
          if (new Set(sw).size !== sw.length) essError('Repeated swizzle components cannot be assigned', e.line);
          if (sw.length === 1) return {c: `${base.c}.${sw[0]}`, t: S(base.t.s)};
          return {base: base.c, swizzle: sw, t: V(base.t.s, sw.length), c: null};
        }
        essError(`Cannot access .${e.name}`, e.line);
      }
      default: essError('Expression is not assignable', e.line);
    }
  }
  builtinPrivate(name, b) {
    const line = `var<private> ${b.w}: ${this.wgsl(b.type)};`;
    if (!this.privates.includes(line)) this.privates.push(line);
    if (name === 'gl_FragCoord') this.shared.needsTarget = true;
  }
  tmp() { return `f3d_t${this.temp++}`; }
  indexCode(e, ctx) {
    const r = this.expr(e, ctx);
    if (!(r.t.k === 's' && (r.t.s === 'int' || r.t.s === 'uint'))) essError('Index must be an integer', e.line);
    return r.c;
  }

  // ---- expressions --------------------------------------------------------
  /** Translate an expression to {c, t}; may append hoisted statements to ctx.pre. */
  expr(e, ctx) {
    const r = this.exprRaw(e, ctx);
    return r.u ? this.uniformValue(r) : r;
  }
  uniformValue(r) { return {c: this.fromUniform(r.c, r.t), t: r.t}; }
  fromUniform(code, t) {
    switch (t.k) {
      case 's': return t.s === 'bool' ? `(${code} != 0u)` : code;
      case 'v': return t.s === 'bool' ? `(${code} != vec${t.n}<u32>(0u))` : code;
      case 'm': return code;
      case 'st': return needsStructConversion(t.name, this.shared) ? `f3d_from_u_${this.shared.structs.get(t.name).wname}(${code})` : code;
      case 'a': {
        const r = uniformRepr(t, this.shared);
        if (!r.wrapped && !(t.of.k === 'st' && needsStructConversion(t.of.name, this.shared))) return code;
        const name = `f3d_from_u_${typeKey(t).replace(/\W/g, '_')}`;
        if (!this.helpers.has(name)) this.helpers.set(name, [`fn ${name}(v: ${this.wgslUniform(r)}) -> ${this.wgsl(t)} {`, `  var r: ${this.wgsl(t)};`,
          `  for (var i = 0; i < ${t.n}; i++) { r[i] = ${r.wrapped ? this.fromUniform(`v[i]${t.of.k === 's' ? '.x' : '.xy'}`, t.of) : this.fromUniform('v[i]', t.of)}; }`, '  return r;', '}'].join('\n'));
        return `${name}(${code})`;
      }
    }
    return code;
  }
  /** Capture an already-translated operand in a temp when later operands hoisted statements. */
  sequence(ctx, parts) {
    const out = [];
    for (const part of parts) {
      const mark = ctx.pre.length;
      const r = part();
      if (ctx.pre.length > mark) {
        // Earlier operands must be evaluated before these hoisted effects.
        for (let k = 0; k < out.length; k++) {
          if (isTrivialCode(out[k].c)) continue;
          const t = this.tmp();
          ctx.pre.splice(mark, 0, `let ${t} = ${out[k].c};`);
          out[k] = {...out[k], c: t};
        }
      }
      out.push(r);
    }
    return out;
  }
  exprRaw(e, ctx) {
    switch (e.k) {
      case 'num': { const n = parseNumber(e.v); return {c: n.code, t: n.type}; }
      case 'bool': return {c: String(e.v), t: BOOL};
      case 'paren': { const r = this.exprRaw(e.e, ctx); return r.u ? r : {c: `(${r.c})`, t: r.t, smp: r.smp}; }
      case 'id': return this.identifier(e, ctx);
      case 'seq': {
        for (const x of e.list.slice(0, -1)) ctx.pre.push(...this.exprStatement(x, ctx));
        return this.expr(e.list.at(-1), ctx);
      }
      case 'assign': {
        ctx.pre.push(...this.assign(e.target, e.op, e.value, ctx, e.line));
        const lv = this.lvalue(e.target, ctx);
        return lv.swizzle ? {c: `${lv.base}.${lv.swizzle.join('')}`, t: lv.t} : {c: lv.c, t: lv.t};
      }
      case 'preinc': {
        ctx.pre.push(...this.exprStatement(e, ctx));
        const lv = this.lvalue(e.a, ctx); return {c: lv.c, t: lv.t};
      }
      case 'postinc': {
        const lv = this.lvalue(e.a, ctx), t = this.tmp();
        ctx.pre.push(`let ${t} = ${lv.c};`, ...this.exprStatement({k: 'preinc', op: e.op, a: e.a, line: e.line}, ctx));
        return {c: t, t: lv.t};
      }
      case 'unary': {
        const a = this.expr(e.a, ctx);
        if (e.op === '+') return a;
        if (e.op === '-') { if (!isNumeric(a.t) && a.t.k !== 'm') essError('Bad operand for -', e.line); return {c: `(-${a.c})`, t: a.t}; }
        if (e.op === '!') { if (!same(a.t, BOOL)) essError('! needs a bool', e.line); return {c: `(!${a.c})`, t: BOOL}; }
        if (e.op === '~') return {c: `(~${a.c})`, t: a.t};
        break;
      }
      case 'bin': return this.binary(e, ctx);
      case 'cond': return this.conditional(e, ctx);
      case 'index': return this.index(e, ctx);
      case 'field': return this.field(e, ctx);
      case 'method': {
        if (e.name !== 'length' || e.args.length) essError(`Unsupported method .${e.name}()`, e.line);
        const a = this.exprRaw(e.a, ctx);
        if (a.t.k === 'a') return {c: String(a.t.n), t: INT};
        if (a.t.k === 'v') return {c: String(a.t.n), t: INT};
        if (a.t.k === 'm') return {c: String(a.t.c), t: INT};
        essError('length() needs an array, vector or matrix', e.line);
      }
      case 'ctor': return this.constructor_(e, ctx);
      case 'call': return this.call(e, ctx);
    }
    essError(`Unsupported expression ${e.k}`, e.line);
  }
  essl1FragColor(name, line) {
    if (name === 'gl_FragData') essError('gl_FragData is not supported; use ESSL 3.00 outputs', line);
    if (!this.outputs.some(o => o.name === 'gl_FragColor')) {
      this.outputs.push({name: 'gl_FragColor', type: V('float', 4), w: 'f3d_FragColor', flat: false, location: 0});
      this.privates.push('var<private> f3d_FragColor: vec4<f32>;');
    }
    return {c: 'f3d_FragColor', t: V('float', 4)};
  }
  identifier(e, ctx) {
    if (this.version === 100 && this.stage === 'fragment' && (e.name === 'gl_FragColor' || e.name === 'gl_FragData') && !this.lookup(e.name, ctx.scope))
      return this.essl1FragColor(e.name, e.line);
    const b = BUILTIN_VARS[this.stage][e.name];
    if (b && !this.lookup(e.name, ctx.scope)) { this.used.add(e.name); this.builtinPrivate(e.name, b); return {c: b.w, t: b.type}; }
    if (e.name === 'gl_PointCoord') essError('gl_PointCoord requires point-sprite expansion, which is not implemented', e.line);
    const s = this.lookup(e.name, ctx.scope);
    if (!s) essError(`Undeclared identifier ${e.name}`, e.line);
    switch (s.kind) {
      case 'uniform': return {c: `f3d_u.${this.shared.uniforms.get(s.name).field}`, t: s.type, u: true};
      case 'sampler': return {c: null, t: s.type, smp: {kind: 'global', name: s.name, sampler: s.sampler}};
      case 'samplerParam': return {c: null, t: s.type, smp: {kind: 'param', w: s.w}};
      case 'pparam': return {c: `(*${s.w})`, t: s.type};
      case 'ublock': return {c: null, t: {k: 'ublock'}, block: s};
      default: return {c: s.w, t: s.type};
    }
  }
  samplerNames(r, line) {
    if (!r.smp) essError('Expected a sampler', line);
    if (r.smp.kind === 'param') return {tex: `${r.smp.w}_texture`, smp: `${r.smp.w}_sampler`, type: r.t};
    const s = r.smp.sampler, base = s.count === null ? safe(r.smp.name) : `${safe(r.smp.name)}_${r.smp.element}`;
    if (s.count !== null && r.smp.element === undefined) essError('Sampler arrays must be indexed with a constant', line);
    return {tex: `${base}_texture`, smp: `${base}_sampler`, type: s.type};
  }
  index(e, ctx) {
    const a = this.exprRaw(e.a, ctx);
    if (a.smp) {
      const k = this.constValue(e.index, ctx.scope);
      if (typeof k !== 'number') essError('Sampler array index must be a constant expression', e.line);
      if (k < 0 || k >= a.smp.sampler.count) essError('Sampler array index out of range', e.line);
      return {c: null, t: a.t.of, smp: {...a.smp, element: k}};
    }
    const [base, idx] = a.u ? [a, null] : [a, null];
    void idx;
    const parts = this.sequence(ctx, [() => base, () => this.expr(e.index, ctx)]);
    const index = parts[1];
    if (!(index.t.k === 's' && (index.t.s === 'int' || index.t.s === 'uint'))) essError('Index must be an integer', e.line);
    const t = elementType(a.t, e.line);
    if (a.u) {
      const repr = uniformRepr(a.t, this.shared);
      if (repr.wrapped) {
        const code = `${parts[0].c}[${index.c}]${t.k === 's' ? '.x' : '.xy'}`;
        return {c: this.fromUniform(code, t), t};
      }
      return {c: `${parts[0].c}[${index.c}]`, t, u: true};
    }
    return {c: `${parts[0].c}[${index.c}]`, t};
  }
  field(e, ctx) {
    const a = this.exprRaw(e.a, ctx);
    if (a.block) {
      const name = a.block.fields.get(e.name);
      if (!name) essError(`No block member ${e.name}`, e.line);
      const u = this.shared.uniforms.get(name);
      return {c: `f3d_u.${u.field}`, t: u.type, u: true};
    }
    if (a.t.k === 'st') {
      const f = this.shared.structs.get(a.t.name).fields.find(x => x.name === e.name);
      if (!f) essError(`No field ${e.name} in ${a.t.name}`, e.line);
      if (a.u) {
        const r = uniformRepr(f.type, this.shared);
        return {c: `${a.c}.${safe(e.name)}`, t: f.type, u: true, repr: r};
      }
      return {c: `${a.c}.${safe(e.name)}`, t: f.type};
    }
    const v = a.u ? this.uniformValue(a) : a;
    if (v.t.k === 'v') {
      const sw = swizzle(e.name, v.t.n, e.line);
      return {c: `${v.c}.${sw.join('')}`, t: vecOf(v.t.s, sw.length)};
    }
    if (v.t.k === 's') {
      const sw = swizzle(e.name, 1, e.line);
      return {c: sw.length === 1 ? v.c : `vec${sw.length}<${WGSL_SCALAR[v.t.s]}>(${v.c})`, t: vecOf(v.t.s, sw.length)};
    }
    essError(`Cannot access .${e.name}`, e.line);
  }
  impure(e) {
    if (!e || typeof e !== 'object') return false;
    if (e.k === 'assign' || e.k === 'preinc' || e.k === 'postinc') return true;
    if (e.k === 'call' && (this.functions.has(e.name) || e.name === 'modf' || e.name === 'frexp')) return true;
    for (const v of Object.values(e)) {
      if (Array.isArray(v)) { if (v.some(x => this.impure(x))) return true; }
      else if (v && typeof v === 'object' && v.k && this.impure(v)) return true;
    }
    return false;
  }
  conditional(e, ctx) {
    const c = this.coerceBool(this.expr(e.c, ctx), e.line);
    const a = this.hoisted(ctx, () => this.expr(e.a, ctx)), b = this.hoisted(ctx, () => this.expr(e.b, ctx));
    if (!same(a.result.t, b.result.t)) essError('?: branches have different types', e.line);
    const t = a.result.t;
    if (!a.pre.length && !b.pre.length && !this.impure(e.a) && !this.impure(e.b) && (t.k === 's' || t.k === 'v'))
      return {c: `select(${b.result.c}, ${a.result.c}, ${c.c})`, t};
    const tmp = this.tmp();
    ctx.pre.push(`var ${tmp}: ${this.wgsl(t)};`, `if (${c.c}) {`, ...[...a.pre, `${tmp} = ${a.result.c};`].map(x => '  ' + x), '} else {',
      ...[...b.pre, `${tmp} = ${b.result.c};`].map(x => '  ' + x), '}');
    return {c: tmp, t};
  }
  binary(e, ctx) {
    if (e.op === '&&' || e.op === '||') {
      const a = this.coerceBool(this.expr(e.a, ctx), e.line);
      const b = this.hoisted(ctx, () => this.coerceBool(this.expr(e.b, ctx), e.line));
      if (!b.pre.length && !this.impure(e.b)) return {c: `(${a.c} ${e.op} ${b.result.c})`, t: BOOL};
      const tmp = this.tmp();
      ctx.pre.push(`var ${tmp}: bool = ${a.c};`, `if (${e.op === '&&' ? '' : '!'}${tmp}) {`, ...[...b.pre, `${tmp} = ${b.result.c};`].map(x => '  ' + x), '}');
      return {c: tmp, t: BOOL};
    }
    const [a, b] = this.sequence(ctx, [() => this.expr(e.a, ctx), () => this.expr(e.b, ctx)]);
    return this.binaryCode(e.op, a, b, e.line);
  }
  binaryCode(op, a, b, line) {
    const ta = a.t, tb = b.t;
    switch (op) {
      case '^^': return {c: `(${a.c} != ${b.c})`, t: BOOL};
      case '==': case '!=': {
        if (!same(ta, tb)) essError(`Cannot compare ${glslName(ta)} with ${glslName(tb)}`, line);
        if (ta.k === 's') return {c: `(${a.c} ${op} ${b.c})`, t: BOOL};
        if (ta.k === 'v') return {c: `${op === '==' ? 'all' : 'any'}(${a.c} ${op} ${b.c})`, t: BOOL};
        if (ta.k === 'm') {
          const cols = Array.from({length: ta.c}, (_, i) => `all(${a.c}[${i}] == ${b.c}[${i}])`).join(' && ');
          return {c: op === '==' ? `(${cols})` : `(!(${cols}))`, t: BOOL};
        }
        essError(`Equality on ${glslName(ta)} is not supported`, line);
      }
      case '<': case '>': case '<=': case '>=':
        if (!same(ta, tb) || ta.k !== 's') essError('Relational operators need matching scalars', line);
        return {c: `(${a.c} ${op} ${b.c})`, t: BOOL};
      case '<<': case '>>': {
        const rhs = scalarOf(tb) === 'uint' ? b.c : tb.k === 'v' ? `vec${tb.n}<u32>(${b.c})` : `u32(${b.c})`;
        return {c: `(${a.c} ${op} ${ta.k === 'v' && tb.k === 's' ? `vec${ta.n}<u32>(${rhs})` : rhs})`, t: ta};
      }
      case '&': case '|': case '^': case '%': {
        const t = ta.k === 'v' ? ta : tb;
        return {c: `(${a.c} ${op} ${b.c})`, t};
      }
      case '+': case '-': case '*': case '/': {
        let t;
        if (ta.k === 'm' && tb.k === 'm') {
          if (op === '*') { if (ta.c !== tb.r) essError('Matrix size mismatch', line); t = M(tb.c, ta.r); }
          else if (op === '/') return {c: this.helperMatDiv(a, b), t: ta};
          else t = ta;
        } else if (ta.k === 'm' && tb.k === 'v') { if (op !== '*') essError('Bad matrix-vector operator', line); t = V('float', ta.r); }
        else if (ta.k === 'v' && tb.k === 'm') { if (op !== '*') essError('Bad vector-matrix operator', line); t = V('float', tb.c); }
        else if (ta.k === 'm' && tb.k === 's') {
          if (op === '/') return {c: `(${a.c} * (1.0 / ${b.c}))`, t: ta};
          if (op === '+' || op === '-') return {c: this.matScalar(op, a, b, false), t: ta};
          t = ta;
        } else if (ta.k === 's' && tb.k === 'm') {
          if (op === '+' || op === '-' || op === '/') return {c: this.matScalar(op, b, a, true), t: tb};
          t = tb;
        } else {
          if (scalarOf(ta) !== scalarOf(tb) || scalarOf(ta) === 'bool') essError(`Operator ${op} on ${glslName(ta)} and ${glslName(tb)}`, line);
          if (ta.k === 'v' && tb.k === 'v' && ta.n !== tb.n) essError('Vector size mismatch', line);
          t = ta.k === 'v' ? ta : tb;
        }
        return {c: `(${a.c} ${op} ${b.c})`, t};
      }
    }
    essError(`Unsupported operator ${op}`, line);
  }
  matScalar(op, m, s, scalarFirst) {
    const cols = Array.from({length: m.t.c}, (_, i) => scalarFirst ? `(${s.c} ${op} ${m.c}[${i}])` : `(${m.c}[${i}] ${op} ${s.c})`);
    return `mat${m.t.c}x${m.t.r}<f32>(${cols.join(', ')})`;
  }
  helperMatDiv(a, b) {
    const cols = Array.from({length: a.t.c}, (_, i) => `(${a.c}[${i}] / ${b.c}[${i}])`);
    return `mat${a.t.c}x${a.t.r}<f32>(${cols.join(', ')})`;
  }
  coerceBool(r, line) { if (!same(r.t, BOOL)) essError(`Expected bool, found ${glslName(r.t)}`, line); return r; }
  /** ESSL 3.00 has no implicit conversions; only array sizes inferred from constructors. */
  coerce(r, t, line) {
    if (same(r.t, t)) return r;
    essError(`Type mismatch: expected ${glslName(t)}, found ${glslName(r.t)}`, line);
  }
  constructor_(e, ctx) {
    let type = this.resolveType({...e.type, array: null});
    if (e.type.array !== null) {
      const n = e.type.array === 'unsized' ? e.args.length : this.constInt(e.type.array, ctx.scope);
      const args = this.sequence(ctx, e.args.map(a => () => this.expr(a, ctx)));
      args.forEach(a => this.coerce(a, type, e.line));
      const at = {k: 'a', of: type, n};
      return {c: `${this.wgsl(at)}(${args.map(a => a.c).join(', ')})`, t: at};
    }
    const args = this.sequence(ctx, e.args.map(a => () => this.expr(a, ctx)));
    if (type.k === 'st') {
      const fields = this.shared.structs.get(type.name).fields;
      if (fields.length !== args.length) essError(`Constructor ${type.name} needs ${fields.length} arguments`, e.line);
      args.forEach((a, i) => this.coerce(a, fields[i].type, e.line));
      return {c: `${this.wgsl(type)}(${args.map(a => a.c).join(', ')})`, t: type};
    }
    if (type.k === 's') {
      if (args.length !== 1) essError('Scalar constructors take one argument', e.line);
      let a = args[0];
      if (a.t.k === 'v') a = {c: `${a.c}.x`, t: S(a.t.s)};
      else if (a.t.k === 'm') a = {c: `${a.c}[0].x`, t: FLOAT};
      if (a.t.k !== 's') essError('Bad scalar constructor argument', e.line);
      return {c: convertScalar(a, type.s), t: type};
    }
    if (type.k === 'v') return this.vectorCtor(type, args, ctx, e.line);
    if (type.k === 'm') return this.matrixCtor(type, args, ctx, e.line);
    essError(`Cannot construct ${glslName(type)}`, e.line);
  }
  vectorCtor(type, args, ctx, line) {
    const ws = `vec${type.n}<${WGSL_SCALAR[type.s]}>`;
    if (args.length === 1) {
      const a = args[0];
      if (a.t.k === 's') return {c: `${ws}(${convertScalar(a, type.s)})`, t: type};
      if (a.t.k === 'v') {
        if (a.t.n < type.n) essError('Too few components in vector constructor', line);
        const src = a.t.n > type.n ? `${a.c}.${'xyzw'.slice(0, type.n)}` : a.c;
        return {c: a.t.s === type.s ? (a.t.n > type.n ? src : a.c) : `${ws}(${src})`, t: type};
      }
      if (a.t.k === 'm') return this.flattenCtor(type, args, ctx, line);
    }
    let total = 0;
    for (const a of args) { if (a.t.k !== 's' && a.t.k !== 'v') return this.flattenCtor(type, args, ctx, line); total += comps(a.t); }
    if (total !== type.n) return this.flattenCtor(type, args, ctx, line);
    const parts = args.map(a => a.t.s === type.s ? a.c : a.t.k === 's' ? convertScalar(a, type.s) : `vec${a.t.n}<${WGSL_SCALAR[type.s]}>(${a.c})`);
    return {c: `${ws}(${parts.join(', ')})`, t: type};
  }
  flattenCtor(type, args, ctx, line) {
    const scalars = this.components(args, ctx);
    const need = comps(type);
    if (scalars.length < need) essError('Too few constructor components', line);
    const s = type.k === 'm' ? 'float' : type.s;
    const list = scalars.slice(0, need).map(x => convertScalar(x, s));
    return {c: `${this.wgsl(type)}(${list.join(', ')})`, t: type};
  }
  components(args, ctx) {
    const out = [];
    for (const a of args) {
      let base = a.c;
      if (!isTrivialCode(base) && comps(a.t) > 1) { const t = this.tmp(); ctx.pre.push(`let ${t} = ${base};`); base = t; }
      if (a.t.k === 's') out.push(a);
      else if (a.t.k === 'v') for (let i = 0; i < a.t.n; i++) out.push({c: `${base}.${'xyzw'[i]}`, t: S(a.t.s)});
      else if (a.t.k === 'm') for (let c = 0; c < a.t.c; c++) for (let r = 0; r < a.t.r; r++) out.push({c: `${base}[${c}][${r}]`, t: FLOAT});
      else essError('Bad constructor argument');
    }
    return out;
  }
  matrixCtor(type, args, ctx, line) {
    const wt = this.wgsl(type);
    if (args.length === 1 && args[0].t.k === 's') {
      const a = this.capture(args[0], ctx), v = convertScalar(a, 'float');
      const list = [];
      for (let c = 0; c < type.c; c++) for (let r = 0; r < type.r; r++) list.push(c === r ? v : '0.0');
      return {c: `${wt}(${list.join(', ')})`, t: type};
    }
    if (args.length === 1 && args[0].t.k === 'm') {
      const src = args[0], m = this.capture(src, ctx);
      const cols = [];
      for (let c = 0; c < type.c; c++) {
        const col = [];
        for (let r = 0; r < type.r; r++) col.push(c < src.t.c && r < src.t.r ? `${m.c}[${c}][${r}]` : c === r ? '1.0' : '0.0');
        cols.push(`vec${type.r}<f32>(${col.join(', ')})`);
      }
      return {c: `${wt}(${cols.join(', ')})`, t: type};
    }
    if (args.length === type.c && args.every(a => a.t.k === 'v' && a.t.n === type.r && a.t.s === 'float'))
      return {c: `${wt}(${args.map(a => a.c).join(', ')})`, t: type};
    if (args.length === type.c * type.r && args.every(a => a.t.k === 's'))
      return {c: `${wt}(${args.map(a => convertScalar(a, 'float')).join(', ')})`, t: type};
    return this.flattenCtor(type, args, ctx, line);
  }
  capture(r, ctx) {
    if (isTrivialCode(r.c)) return r;
    const t = this.tmp(); ctx.pre.push(`let ${t} = ${r.c};`); return {c: t, t: r.t};
  }

  // ---- calls --------------------------------------------------------------
  call(e, ctx) {
    if (this.functions.has(e.name)) return this.userCall(e, ctx);
    return this.builtin(e, ctx);
  }
  userCall(e, ctx) {
    const list = this.functions.get(e.name).filter(f => f.decl);
    // Translate arguments in order; out arguments are lvalues, not values.
    const raw = [];
    const parts = this.sequence(ctx, e.args.map((a, i) => () => {
      const r = this.exprRaw(a, ctx);
      raw[i] = r;
      return r.smp ? {c: '', t: r.t, smp: r.smp} : r.u ? this.uniformValue(r) : r;
    }));
    const f = list.find(x => x.params.length === parts.length && x.params.every((p, i) => p.type.k === 'smp'
      ? !!parts[i].smp && this.samplerNames(raw[i], e.line).type.name === p.type.name : !parts[i].smp && same(p.type, parts[i].t)));
    if (!f) essError(`No matching overload for ${e.name}(${parts.map(p => glslName(p.t)).join(', ')})`, e.line);
    const callArgs = [], writeback = [];
    f.params.forEach((p, i) => {
      if (p.type.k === 'smp') { const s = this.samplerNames(raw[i], e.line); callArgs.push(s.tex, s.smp); return; }
      if (p.q === 'out' || p.q === 'inout') {
        const lv = this.lvalue(e.args[i], ctx), t = this.tmp();
        ctx.pre.push(`var ${t}: ${this.wgsl(p.type)}${p.q === 'inout' ? ` = ${lv.swizzle ? `${lv.base}.${lv.swizzle.join('')}` : lv.c}` : ''};`);
        callArgs.push(`&${t}`);
        writeback.push(() => this.assign(e.args[i], '=', {k: 'f3d_code', c: t, t: p.type}, ctx, e.line));
        return;
      }
      callArgs.push(parts[i].c);
    });
    const code = `${f.wname}(${callArgs.join(', ')})`;
    if (!writeback.length) return {c: code, t: f.ret};
    let result = {c: '', t: f.ret};
    if (f.ret.k === 'void') ctx.pre.push(`${code};`);
    else { const t = this.tmp(); ctx.pre.push(`let ${t} = ${code};`); result = {c: t, t: f.ret}; }
    for (const w of writeback) ctx.pre.push(...w());
    return result;
  }
  builtin(e, ctx) {
    const name = e.name;
    const raw = this.sequence(ctx, e.args.map(a => () => { const r = this.exprRaw(a, ctx); return r.smp || !r.u ? r : this.uniformValue(r); }));
    const args = raw;
    const line = e.line;
    const n = args.length;
    const need = (...counts) => { if (!counts.includes(n)) essError(`${name} expects ${counts.join(' or ')} arguments`, line); };
    const widest = () => args.reduce((w, a) => (comps(a.t) > comps(w.t) && a.t.k === 'v' ? a : w), args[0]).t;
    const splat = (a, t) => (a.t.k === 's' && t.k === 'v' ? `vec${t.n}<${WGSL_SCALAR[t.s]}>(${a.c})` : a.c);
    const simple = (w, rt) => ({c: `${w}(${args.map(a => a.c).join(', ')})`, t: rt});
    const genericF = w => { const t = widest(); return {c: `${w}(${args.map(a => splat(a, t)).join(', ')})`, t}; };
    switch (name) {
      case 'radians': case 'degrees': case 'sin': case 'cos': case 'tan': case 'asin': case 'acos': case 'sinh': case 'cosh': case 'tanh':
      case 'asinh': case 'acosh': case 'atanh': case 'exp': case 'log': case 'exp2': case 'log2': case 'sqrt': case 'floor': case 'ceil':
      case 'fract': case 'trunc': case 'normalize':
        need(1); return simple(name, args[0].t);
      case 'round': case 'roundEven': need(1); return simple('round', args[0].t);
      case 'inversesqrt': need(1); return simple('inverseSqrt', args[0].t);
      case 'abs': case 'sign': need(1); return simple(name, args[0].t);
      case 'pow': need(2); return genericF('pow');
      case 'atan': need(1, 2); return n === 1 ? simple('atan', args[0].t) : genericF('atan2');
      case 'min': case 'max': need(2); return genericF(name);
      case 'clamp': need(3); return genericF('clamp');
      case 'mix': {
        need(3);
        if (scalarOf(args[2].t) === 'bool') {
          const t = args[0].t;
          return {c: `select(${args[0].c}, ${args[1].c}, ${t.k === 'v' && args[2].t.k === 's' ? `vec${t.n}<bool>(${args[2].c})` : args[2].c})`, t};
        }
        const t = widest();
        return {c: `mix(${splat(args[0], t)}, ${splat(args[1], t)}, ${splat(args[2], t)})`, t};
      }
      case 'step': need(2); return genericF('step');
      case 'smoothstep': need(3); return genericF('smoothstep');
      case 'mod': {
        need(2);
        const t = widest();
        const x = this.capture(args[0], ctx), y = this.capture(args[1], ctx);
        return {c: `(${splat(x, t)} - ${splat(y, t)} * floor(${splat(x, t)} / ${splat(y, t)}))`, t};
      }
      case 'modf': {
        need(2);
        const r = this.tmp();
        ctx.pre.push(`let ${r} = modf(${args[0].c});`);
        ctx.pre.push(...this.assign(e.args[1], '=', {k: 'f3d_code', c: `${r}.whole`, t: args[0].t}, ctx, line));
        return {c: `${r}.fract`, t: args[0].t};
      }
      case 'length': need(1); return {c: `length(${args[0].c})`, t: FLOAT};
      case 'distance': need(2); return {c: `distance(${args[0].c}, ${args[1].c})`, t: FLOAT};
      case 'dot': need(2); return {c: args[0].t.k === 's' ? `(${args[0].c} * ${args[1].c})` : `dot(${args[0].c}, ${args[1].c})`, t: S(scalarOf(args[0].t))};
      case 'cross': need(2); return simple('cross', V('float', 3));
      case 'faceforward': need(3); return simple('faceForward', args[0].t);
      case 'reflect': need(2); return simple('reflect', args[0].t);
      case 'refract': need(3); return simple('refract', args[0].t);
      case 'transpose': need(1); return {c: `transpose(${args[0].c})`, t: M(args[0].t.r, args[0].t.c)};
      case 'determinant': need(1); return {c: `determinant(${args[0].c})`, t: FLOAT};
      case 'inverse': need(1); return {c: `${this.inverseHelper(args[0].t, line)}(${args[0].c})`, t: args[0].t};
      case 'matrixCompMult': {
        need(2); const t = args[0].t, a = this.capture(args[0], ctx), b = this.capture(args[1], ctx);
        return {c: `mat${t.c}x${t.r}<f32>(${Array.from({length: t.c}, (_, i) => `${a.c}[${i}] * ${b.c}[${i}]`).join(', ')})`, t};
      }
      case 'outerProduct': {
        need(2); const c = this.capture(args[0], ctx), r = this.capture(args[1], ctx), t = M(r.t.n, c.t.n);
        return {c: `mat${t.c}x${t.r}<f32>(${Array.from({length: t.c}, (_, i) => `${c.c} * ${r.c}[${i}]`).join(', ')})`, t};
      }
      case 'lessThan': case 'lessThanEqual': case 'greaterThan': case 'greaterThanEqual': case 'equal': case 'notEqual': {
        need(2);
        const op = {lessThan: '<', lessThanEqual: '<=', greaterThan: '>', greaterThanEqual: '>=', equal: '==', notEqual: '!='}[name];
        return {c: `(${args[0].c} ${op} ${args[1].c})`, t: V('bool', args[0].t.n)};
      }
      case 'any': case 'all': need(1); return {c: `${name}(${args[0].c})`, t: BOOL};
      case 'not': need(1); return {c: `(!${args[0].c})`, t: args[0].t};
      case 'isnan': case 'isinf': {
        need(1); const t = args[0].t, u = t.k === 'v' ? `vec${t.n}<u32>` : 'u32';
        const bits = `(bitcast<${u}>(${args[0].c}) & ${u}(0x7fffffffu))`;
        return {c: name === 'isnan' ? `(${bits} > ${u}(0x7f800000u))` : `(${bits} == ${u}(0x7f800000u))`, t: withScalar(t, 'bool')};
      }
      case 'floatBitsToInt': need(1); return {c: `bitcast<${this.wgsl(withScalar(args[0].t, 'int'))}>(${args[0].c})`, t: withScalar(args[0].t, 'int')};
      case 'floatBitsToUint': need(1); return {c: `bitcast<${this.wgsl(withScalar(args[0].t, 'uint'))}>(${args[0].c})`, t: withScalar(args[0].t, 'uint')};
      case 'intBitsToFloat': case 'uintBitsToFloat': need(1); return {c: `bitcast<${this.wgsl(withScalar(args[0].t, 'float'))}>(${args[0].c})`, t: withScalar(args[0].t, 'float')};
      case 'packHalf2x16': need(1); return {c: `pack2x16float(${args[0].c})`, t: UINT};
      case 'unpackHalf2x16': need(1); return {c: `unpack2x16float(${args[0].c})`, t: V('float', 2)};
      case 'packSnorm2x16': need(1); return {c: `pack2x16snorm(${args[0].c})`, t: UINT};
      case 'unpackSnorm2x16': need(1); return {c: `unpack2x16snorm(${args[0].c})`, t: V('float', 2)};
      case 'packUnorm2x16': need(1); return {c: `pack2x16unorm(${args[0].c})`, t: UINT};
      case 'unpackUnorm2x16': need(1); return {c: `unpack2x16unorm(${args[0].c})`, t: V('float', 2)};
      case 'dFdx': need(1); return {c: `dpdx(${args[0].c})`, t: args[0].t};
      // WebGPU framebuffer Y grows downward; GL's dFdy is with respect to upward Y.
      case 'dFdy': need(1); return {c: `(-dpdy(${args[0].c}))`, t: args[0].t};
      case 'fwidth': need(1); return {c: `fwidth(${args[0].c})`, t: args[0].t};
    }
    if (this.version === 100 && ESSL1_TEXTURE[name]) return this.textureCall(ESSL1_TEXTURE[name], args, ctx, line);
    if (/^texture|^texelFetch/.test(name)) return this.textureCall(name, args, ctx, line);
    essError(`Unknown function ${name}`, line);
  }
  textureCall(name, args, ctx, line) {
    if (!args[0]?.smp) essError(`${name} needs a sampler`, line);
    const s = this.samplerNames(args[0], line), st = s.type, rest = args.slice(1);
    const vec4 = st.s === 'float' ? V('float', 4) : V(st.s, 4);
    const coordNeed = {'2d': 2, '3d': 3, 'cube': 3, '2d-array': 3}[st.dim] + (st.shadow ? 1 : 0);
    const isInt = st.s !== 'float';
    const coords = (p, n) => {
      // Split coordinate (+ array layer / depth reference) per WGSL signatures.
      const c = this.capture(p, ctx);
      if (st.dim === '2d-array') return {xy: `${c.c}.xy`, layer: `i32(floor(${c.c}.z + 0.5))`, ref: st.shadow ? `${c.c}.w` : null};
      if (st.shadow) return {xy: st.dim === 'cube' ? `${c.c}.xyz` : `${c.c}.xy`, ref: st.dim === 'cube' ? `${c.c}.w` : `${c.c}.z`};
      void n; return {xy: c.c};
    };
    const call = (fn, extra = []) => {
      const p = coords(rest[0], coordNeed);
      const list = [s.tex, s.smp, p.xy, ...(p.layer ? [p.layer] : []), ...(p.ref ? [p.ref] : []), ...extra];
      return `${fn}(${list.join(', ')})`;
    };
    switch (name) {
      case 'texture': {
        if (isInt) essError('Filtered sampling of integer textures is not supported; use texelFetch', line);
        // Outside fragment shaders ESSL implicit LOD is the base level.
        if (this.stage !== 'fragment') return st.shadow ? {c: call('textureSampleCompareLevel'), t: FLOAT} : {c: call('textureSampleLevel', ['0.0']), t: vec4};
        if (st.shadow) return {c: call('textureSampleCompare'), t: FLOAT};
        if (rest.length === 2) return {c: call('textureSampleBias', [rest[1].c]), t: vec4};
        return {c: call('textureSample'), t: vec4};
      }
      case 'textureLod': {
        if (isInt) essError('Filtered sampling of integer textures is not supported', line);
        if (st.shadow) return {c: call('textureSampleCompareLevel'), t: FLOAT};
        return {c: call('textureSampleLevel', [rest[1].c]), t: vec4};
      }
      case 'textureGrad': if (st.shadow) essError('textureGrad on shadow samplers is not supported', line); return {c: call('textureSampleGrad', [rest[1].c, rest[2].c]), t: vec4};
      case 'textureOffset': return {c: call(st.shadow ? 'textureSampleCompare' : 'textureSample', [rest[1].c]), t: st.shadow ? FLOAT : vec4};
      case 'textureLodOffset': return {c: call('textureSampleLevel', [rest[1].c, rest[2].c]), t: vec4};
      case 'textureProj': case 'textureProjLod': {
        if (st.dim !== '2d' || st.shadow) essError(`${name} is supported for sampler2D only`, line);
        const p = this.capture(rest[0], ctx), w = p.t.n === 4 ? `${p.c}.w` : `${p.c}.z`;
        const uv = `(${p.c}.xy / ${w})`;
        if (name === 'textureProj' && this.stage === 'fragment') return {c: `textureSample(${s.tex}, ${s.smp}, ${uv})`, t: vec4};
        return {c: `textureSampleLevel(${s.tex}, ${s.smp}, ${uv}, ${name === 'textureProj' ? '0.0' : rest[1].c})`, t: vec4};
      }
      case 'texelFetch': {
        const p = rest[0];
        const lod = rest[1]?.c ?? '0';
        if (st.dim === '2d-array') { const c = this.capture(p, ctx); return {c: `textureLoad(${s.tex}, ${c.c}.xy, ${c.c}.z, ${lod})`, t: vec4}; }
        return {c: `textureLoad(${s.tex}, ${p.c}, ${lod})`, t: vec4};
      }
      case 'textureSize': {
        const dims = st.dim === '2d' || st.dim === 'cube' ? 2 : 3;
        if (st.dim === '2d-array') return {c: `vec3<i32>(vec3<u32>(textureDimensions(${s.tex}, ${rest[0].c}), textureNumLayers(${s.tex})))`, t: V('int', 3)};
        return {c: `vec${dims}<i32>(textureDimensions(${s.tex}, ${rest[0].c}))`, t: V('int', dims)};
      }
    }
    essError(`Unsupported texture function ${name}`, line);
  }
  inverseHelper(t, line) {
    if (t.k !== 'm' || t.c !== t.r) essError('inverse needs a square matrix', line);
    const name = `f3d_inverse${t.c}`;
    if (!this.helpers.has(name)) this.helpers.set(name, INVERSE[t.c]);
    return name;
  }
}

// Handle synthesized code operands (writebacks/modf) in expr().
const baseExprRaw = Unit.prototype.exprRaw;
Unit.prototype.exprRaw = function (e, ctx) { return e.k === 'f3d_code' ? {c: e.c, t: e.t} : baseExprRaw.call(this, e, ctx); };

const INVERSE = {
  2: `fn f3d_inverse2(m: mat2x2<f32>) -> mat2x2<f32> {
  let d = 1.0 / determinant(m);
  return mat2x2<f32>(m[1][1] * d, -m[0][1] * d, -m[1][0] * d, m[0][0] * d);
}`,
  3: `fn f3d_inverse3(m: mat3x3<f32>) -> mat3x3<f32> {
  let r0 = cross(m[1], m[2]);
  let r1 = cross(m[2], m[0]);
  let r2 = cross(m[0], m[1]);
  return transpose(mat3x3<f32>(r0, r1, r2)) * (1.0 / dot(m[0], r0));
}`,
  4: `fn f3d_inverse4(m: mat4x4<f32>) -> mat4x4<f32> {
  let a00 = m[0][0]; let a01 = m[0][1]; let a02 = m[0][2]; let a03 = m[0][3];
  let a10 = m[1][0]; let a11 = m[1][1]; let a12 = m[1][2]; let a13 = m[1][3];
  let a20 = m[2][0]; let a21 = m[2][1]; let a22 = m[2][2]; let a23 = m[2][3];
  let a30 = m[3][0]; let a31 = m[3][1]; let a32 = m[3][2]; let a33 = m[3][3];
  let b00 = a00 * a11 - a01 * a10; let b01 = a00 * a12 - a02 * a10; let b02 = a00 * a13 - a03 * a10;
  let b03 = a01 * a12 - a02 * a11; let b04 = a01 * a13 - a03 * a11; let b05 = a02 * a13 - a03 * a12;
  let b06 = a20 * a31 - a21 * a30; let b07 = a20 * a32 - a22 * a30; let b08 = a20 * a33 - a23 * a30;
  let b09 = a21 * a32 - a22 * a31; let b10 = a21 * a33 - a23 * a31; let b11 = a22 * a33 - a23 * a32;
  let inv = 1.0 / (b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06);
  return mat4x4<f32>(
    vec4<f32>(a11 * b11 - a12 * b10 + a13 * b09, a02 * b10 - a01 * b11 - a03 * b09, a31 * b05 - a32 * b04 + a33 * b03, a22 * b04 - a21 * b05 - a23 * b03) * inv,
    vec4<f32>(a12 * b08 - a10 * b11 - a13 * b07, a00 * b11 - a02 * b08 + a03 * b07, a32 * b02 - a30 * b05 - a33 * b01, a20 * b05 - a22 * b02 + a23 * b01) * inv,
    vec4<f32>(a10 * b10 - a11 * b08 + a13 * b06, a01 * b08 - a00 * b10 - a03 * b06, a30 * b04 - a31 * b02 + a33 * b00, a21 * b02 - a20 * b04 - a23 * b00) * inv,
    vec4<f32>(a11 * b07 - a10 * b09 - a12 * b06, a00 * b09 - a01 * b07 + a02 * b06, a31 * b01 - a30 * b03 - a32 * b00, a20 * b03 - a21 * b01 + a22 * b00) * inv);
}`,
};

function textureType(t) {
  if (t.shadow) return {'2d': 'texture_depth_2d', cube: 'texture_depth_cube', '2d-array': 'texture_depth_2d_array'}[t.dim];
  const s = WGSL_SCALAR[t.s];
  return `${{'2d': 'texture_2d', '3d': 'texture_3d', cube: 'texture_cube', '2d-array': 'texture_2d_array'}[t.dim]}<${s}>`;
}
function elementType(t, line) {
  if (t.k === 'a') return t.of;
  if (t.k === 'v') return S(t.s);
  if (t.k === 'm') return V('float', t.r);
  essError(`Cannot index ${glslName(t)}`, line);
}
function swizzle(name, n, line) {
  const sets = ['xyzw', 'rgba', 'stpq'];
  const set = sets.find(s => [...name].every(ch => s.includes(ch)));
  if (!set || name.length > 4) essError(`Invalid swizzle .${name}`, line);
  return [...name].map(ch => { const i = set.indexOf(ch); if (i >= n) essError(`Swizzle .${name} out of range`, line); return 'xyzw'[i]; });
}
function convertScalar(a, s) {
  if (a.t.s === s) return a.c;
  if (s === 'bool') return a.t.s === 'float' ? `(${a.c} != 0.0)` : a.t.s === 'uint' ? `(${a.c} != 0u)` : `(${a.c} != 0)`;
  return `${WGSL_SCALAR[s]}(${a.c})`;
}
function parseNumber(text) {
  let v = text;
  const isFloat = /[.eE]/.test(v) && !/^0[xX]/.test(v) || /[fF]$/.test(v) && !/^0[xX]/.test(v);
  if (isFloat) {
    v = v.replace(/[fF]$/, '');
    let code = v;
    if (code.startsWith('.')) code = '0' + code;
    if (/\.$/.test(code)) code += '0';
    if (/\.[eE]/.test(code)) code = code.replace('.', '.0');
    if (!/[.eE]/.test(code)) code += '.0';
    return {code, type: FLOAT, value: Number(v)};
  }
  const unsigned = /[uU]$/.test(v);
  v = v.replace(/[uU]$/, '');
  const value = /^0[xX]/.test(v) ? parseInt(v, 16) : /^0[0-7]+$/.test(v) ? parseInt(v, 8) : parseInt(v, 10);
  if (unsigned) return {code: `${value >>> 0}u`, type: UINT, value: value >>> 0};
  if (value > 0x7fffffff) return {code: `bitcast<i32>(${value >>> 0}u)`, type: INT, value: value | 0};
  return {code: String(value), type: INT, value};
}
function zero(unit, t) {
  return `${unit.wgsl(t)}()`;
}
function endsWithReturn(lines) {
  const last = lines.filter(l => l.trim()).at(-1)?.trim() ?? '';
  return last.startsWith('return') || last === 'discard;';
}
const isTrivialCode = c => /^[A-Za-z_][\w]*$/.test(c) || /^-?[\d.]+[uf]?$/.test(c) || c === 'true' || c === 'false';
