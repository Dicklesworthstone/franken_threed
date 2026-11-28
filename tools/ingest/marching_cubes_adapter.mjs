/**
 * Compiler-owned bridge for the source-verified MarchingCubes addon. Construct
 * genuine retained objects and keep every original export, prototype and field.
 * Instrument numeric loops; original method identities, field setup and update
 * publication stay in place. Other addon methods remain entirely retained.
 *
 * Registration is emitted INSIDE verified base EventDispatcher/Color constructors
 * where `this` is freshly allocated, never an arbitrary caller-owned object.
 * Unknown owners/materials, including proxies around known objects, take the
 * original path without property inspection. Bootstrap trusted platform APIs.
 */
import {instantiateNumericKernel,numericArrayIterationIntact} from './numeric_kernel_runtime.mjs';
import {registerNumericObject,isNumericObject} from './numeric_objects.mjs';
const apply=Reflect.apply, descriptor=Object.getOwnPropertyDescriptor, hasOwn=Object.hasOwn;
const prototype=Object.getPrototypeOf,setHas=WeakSet.prototype.has,setAdd=WeakSet.prototype.add;
const materialPrototypes=new WeakSet();
// The primordial object is an ordinary identity, not an application's chain.
// Inspect its live flatShading descriptor too; inherited getters must retain JS.
apply(setAdd,materialPrototypes,[Object.prototype]);
const isInteger=Number.isInteger, U8=Uint8Array, decode=globalThis.atob, charCodeAt=String.prototype.charCodeAt;
const empty=new Float32Array(0);
// The polygonizer never reads these attributes. It writes every component
// of each emitted vertex and returns the vertex count. Disabled attributes
// receive empty views; cache and edge lists keep their full input histories.
const vertexWidths=Object.freeze({position:3,normal:3,uv:2,color:3});
// Do not invoke replacement platform getters/constructors during admission.
const data=(object,key)=>object && descriptor(object,key)?.value;
const wasm=data(globalThis,'WebAssembly');
const moduleType=data(wasm,'Module'),instanceType=data(wasm,'Instance'),memoryType=data(wasm,'Memory');
const memoryPrototype=memoryType?.prototype;
const moduleMethods=['customSections','imports','exports'].map(key=>[key,data(moduleType,key)]);
const grow=data(memoryPrototype,'grow'),bufferGetter=memoryPrototype && descriptor(memoryPrototype,'buffer')?.get;
function platformIntact() {
  if (!wasm || data(globalThis,'WebAssembly')!==wasm || data(wasm,'Module')!==moduleType ||
      data(wasm,'Instance')!==instanceType || data(wasm,'Memory')!==memoryType ||
      data(memoryType,'prototype')!==memoryPrototype || data(memoryPrototype,'grow')!==grow ||
      descriptor(memoryPrototype,'buffer')?.get!==bufferGetter) return false;
  for (let i=0;i<moduleMethods.length;i++) {
    const [key,value]=moduleMethods[i];if (data(moduleType,key)!==value) return false;
  }
  return true;
}
const objects={has:isNumericObject},dispatches=new WeakMap(),updates=new WeakMap(),fieldUpdates=new WeakMap();
const ballOwners=new WeakMap(),ballColors=new WeakMap(),ballManagements=new WeakMap(),ballRecipes=new WeakMap(),ballArrays=new WeakSet(),ballUpdates=new WeakMap();
const F64=Float64Array,functionPrototype=Function.prototype,objectPrototype=Object.prototype;
const functionSource=Function.prototype.toString,instanceCheck=data(functionPrototype,Symbol.hasInstance);
const mathObject=data(globalThis,'Math'),ballMath=new Map();
for(const name of ['abs','acos','asin','atan','atan2','ceil','cos','exp','floor','log','max','min','pow','round','sign','sin','sqrt','tan','trunc']) {
  const fn=data(mathObject,name);
  ballMath.set(name,typeof fn==='function'&&apply(functionSource,fn,[])===`function ${name}() { [native code] }`?fn:null);
}
const own=(object,key)=>{
  const d=descriptor(object,key);
  return d && hasOwn(d,'value') ? d : null;
};
const names=['size','size2','size3','halfsize','delta','yd','zd','isolation','field','normal_cache',
  'palette','positionArray','normalArray','material','geometry','enableUvs','enableColors','count'];

/** Called only by verified base constructors (EventDispatcher/Color) with fresh `this`. */
export function registerMarchingCubesObject(object) { registerNumericObject(object); }

/** Emitted in static blocks of exact source class definitions. Their freshly
 * created prototype is ordinary; callers never nominate a prototype by name. */
export function registerMarchingCubesMaterialPrototype(object) { apply(setAdd,materialPrototypes,[object]); }

/** Strict-true source semantics, including an absent inherited property.
 * The instance witness permits its own descriptor/prototype access. Every next
 * prototype must have its own compiler witness BEFORE it is inspected, so a
 * reparented unknown object or Proxy retains the source without extra traps. */
function flatShading(material) {
  if(!objects.has(material))return null;
  let current=material;
  while(current!==null) {
    const d=descriptor(current,'flatShading');
    if(d)return hasOwn(d,'value')?{flat:d.value===true?1:0}:null;
    current=prototype(current);
    if(current!==null && !apply(setHas,materialPrototypes,[current]))return null;
  }
  return {flat:0};
}

/** No binary decoding or Wasm allocation until the first admitted update. */
export function createMarchingCubesDispatch(base64,edgeTable,triTable) {
  if (typeof base64!=='string' || base64.length>1024*1024) throw new TypeError('Expected bounded compiled Wasm');
  const token=Object.freeze({});
  dispatches.set(token,{base64,edgeTable,triTable,resultPrefixWidths:vertexWidths,attempted:false,kernel:null,initializationFailure:null});
  return token;
}
function kernelFor(record) {
  if (!record.attempted) {
    record.attempted=true;
    try {
      const binary=apply(decode,globalThis,[record.base64]),bytes=new U8(binary.length);
      for (let k=0;k<binary.length;k++) bytes[k]=apply(charCodeAt,binary,[k]);
      // Independent storage is essential: do NOT enable preserveAliasing.
      record.kernel=instantiateNumericKernel(bytes,{resolveMath:record.resolveMath??null,
        resultPrefixWidths:record.resultPrefixWidths??null,fitResultPrefixes:record.resultPrefixWidths===vertexWidths});
    } catch {
      // Host policy may throw arbitrary values with effectful accessors.
      record.initializationFailure='MARCHING_CUBES_INITIALIZATION';
    }
    record.base64=null;
  }
  return record.kernel;
}
function snapshot(object) {
  const values={};
  for (const key of names) {
    const d=own(object,key);
    if (!d || (key==='count' && !d.writable)) return null;
    values[key]=d.value;
  }
  const s=values.size;
  if (!isInteger(s) || s<1 || s>256 || values.size2!==s*s || values.size3!==s*s*s ||
      values.halfsize!==s/2 || values.delta!==2/s || values.yd!==s || values.zd!==s*s ||
      typeof values.isolation!=='number' || typeof values.enableUvs!=='boolean' || typeof values.enableColors!=='boolean') return null;
  // WeakSet.has never invokes a Proxy's traps. Only a compiler registration of
  // the real constructed identity authorizes own-descriptor inspection.
  const shading=flatShading(values.material);
  if (!shading) return null;
  let uvs=null,colors=null;
  if (values.enableUvs) {const d=own(object,'uvArray');if (!d) return null;uvs=d.value;}
  if (values.enableColors) {const d=own(object,'colorArray');if (!d) return null;colors=d.value;}
  return {values,uvs:uvs??empty,colors:colors??empty,flat:shading.flat};
}

/**
 * Compiler-emitted branch at the start of the original update function. False
 * means execute its retained numeric body once; true skips only that body.
 * The original geometry/attribute/warning tail ALWAYS runs in the source method,
 * outside this helper's exception handling. Its identity/name/length are not
 * replaced, and it retains the construction geometry and maxPolyCount closure.
 */
export function tryMarchingCubesUpdate(receiver,object,token,edgePositions,edgeNormals,edgeColors) {
  let stats=updates.get(object);
  if (!stats) {stats={wasmCalls:0,fallbackCalls:0,lastFailure:null,token:null};updates.set(object,stats);}
  stats.token=token;
  const fallback=reason=>{stats.fallbackCalls++;stats.lastFailure=reason;return false;};
  // No inspection of a borrowed/proxied receiver: preserve original this/scope.
  if (receiver!==object || !objects.has(object)) return fallback('MARCHING_CUBES_RECEIVER');
  const record=dispatches.get(token);
  if (!record) return fallback('MARCHING_CUBES_UNREGISTERED');
  if (!snapshot(object)) return fallback('MARCHING_CUBES_OBJECT_GUARD');
  if (!platformIntact()) return fallback('MARCHING_CUBES_PLATFORM');
  const kernel=kernelFor(record);
  if (!kernel) return fallback(record.initializationFailure);
  // Initialization can encounter host policy hooks. Re-read admitted data.
  const input=snapshot(object);
  if (!input) return fallback('MARCHING_CUBES_OBJECT_GUARD');
  const v=input.values;
  let count;
  try {
    count=kernel.run(v.field,v.normal_cache,v.palette,v.positionArray,v.normalArray,input.uvs,input.colors,
      edgePositions,edgeNormals,edgeColors,record.edgeTable,record.triTable,
      v.size,v.size2,v.halfsize,v.delta,v.yd,v.zd,v.isolation,
      input.flat,v.enableUvs ? 1 : 0,v.enableColors ? 1 : 0);
  } catch {return fallback(kernel.diagnostics.lastGuardFailure ?? 'MARCHING_CUBES_EXECUTION');}
  // Own writable data count was guarded. No caller code has run since the last
  // snapshot. This is after native output publication: no fallback below here.
  object.count=count;
  stats.wasmCalls++;stats.lastFailure=null;
  return true;
}

/** Opt-in diagnostics; the generated Three module gains no new public exports. */
export function marchingCubesDiagnostics(object) {
  const stats=updates.get(object),fieldKernels=fieldDiagnostics(object),ballLoops=ballDiagnostics(object);
  if (!stats) return Object.freeze({wasmCalls:0,fallbackCalls:0,lastFailure:null,initialized:false,kernel:null,fieldKernels,ballLoops});
  const dispatch=dispatches.get(stats.token);
  return Object.freeze({wasmCalls:stats.wasmCalls,fallbackCalls:stats.fallbackCalls,lastFailure:stats.lastFailure,
    initialized:dispatch?.attempted??false,kernel:dispatch?.kernel?.diagnostics??null,fieldKernels,ballLoops});
}

/** Emitted around the pinned ColorManagement literal before its initialization
 * can invoke imported code. Only this freshly allocated identity authorizes
 * inspecting the resolved dependency, including through source re-exports. */
export function registerMarchingCubesBallManagement(management) {
  ballManagements.set(management,{convert:data(management,'convert'),toWorking:data(management,'colorSpaceToWorking')});
  return management;
}
/** Resolve the actual live import only when the application calls addBall.
 * Reading it at class evaluation would introduce a new TDZ failure in a cycle.
 * The emitted resolver contains only that lexical read, never an application call. */
export function registerMarchingCubesBallColorClass(Class,resolveManagement) {
  const proto=data(Class,'prototype');
  ballColors.set(Class,{Class,proto,resolveManagement,set:data(proto,'set'),setRGB:data(proto,'setRGB')});
}
export function attachMarchingCubesBallBatch(token,base64,resolveMath) {
  const record=dispatches.get(token);
  if(record)record.balls={base64,resolveMath,attempted:false,kernel:null,initializationFailure:null};
}
/** Invoked at allocation of the verified function, BEFORE assignment to the
 * public property. A prototype setter cannot nominate its own replacement. */
export function registerMarchingCubesBallOwner(object,token,Color,method) {
  if(objects.has(object))ballOwners.set(object,{record:dispatches.get(token),Color,method});
  return method;
}
export function registerMarchingCubesBallArray(array) {apply(setAdd,ballArrays,[array]);return array;}
export function createMarchingCubesBallRecipe(prepare,scalarCount,countIndex,math,condition) {
  const token=Object.freeze({});ballRecipes.set(token,{prepare,scalarCount,countIndex,math,condition});return token;
}
function ballPlatform(recipe) {
  if(!numericArrayIterationIntact()||!platformIntact()||data(globalThis,'Math')!==mathObject)return false;
  for(const name of ['sign','abs','sqrt','floor',...recipe.math])
    if(!ballMath.get(name)||data(mathObject,name)!==ballMath.get(name))return false;
  return true;
}
function ballColorIntact(record) {
  if(!record)return false;
  const {Class,proto}=record;
  let management;
  try{management=record.resolveManagement();}catch{return false;}
  const source=ballManagements.get(management);
  // No property/prototype inspection of a redirected object or Proxy, even
  // when its underlying source module was otherwise recognized.
  if(!source)return false;
  if(data(Class,'prototype')!==proto||prototype(Class)!==functionPrototype||prototype(proto)!==objectPrototype||
    descriptor(Class,Symbol.hasInstance)||data(functionPrototype,Symbol.hasInstance)!==instanceCheck||
    data(proto,'set')!==record.set||data(proto,'setRGB')!==record.setRGB||
    data(management,'convert')!==source.convert||data(management,'colorSpaceToWorking')!==source.toWorking||
    !own(management,'workingColorSpace')||!own(management,'enabled'))return false;
  // Fresh Color assignments must neither call inherited setters nor throw on
  // a nonwritable inherited slot. The two inspected prototypes are witnessed.
  for(const key of ['r','g','b','isColor'])for(const owner of [proto,objectPrototype]) {
    const d=descriptor(owner,key);if(d&&(!hasOwn(d,'value')||!d.writable))return false;
  }
  return true;
}
function ballInput(owner,object,palette) {
  if(data(object,'addBall')!==owner.method)return null;
  const color=ballColors.get(owner.Color);
  if(!ballColorIntact(color))return null;
  const size=data(object,'size'),size2=data(object,'size2');
  if(!isInteger(size)||size<1||size>256||size2!==size*size)return null;
  const field=own(object,'field'),values=own(object,'palette');
  if(!field||!values)return null;
  let colors=empty,colorCount=0;
  if(palette!==null) {
    if(!apply(setHas,ballArrays,[palette]))return null;
    const length=data(palette,'length');if(!isInteger(length)||length<1||length>4096)return null;
    colorCount=length;
    colors=new F64(length*3);
    for(let i=0;i<length;i++) {
      const value=data(palette,i);
      if(!objects.has(value)||prototype(value)!==color.proto)return null;
      for(let k=0;k<3;k++) {
        const component=own(value,['r','g','b'][k]);
        if(!component||typeof component.value!=='number')return null;
        colors[i*3+k]=component.value;
      }
    }
  }
  return {field:field.value,palette:values.value,size,size2,colors,colorCount};
}
/** Every preparation expression is compiler-checked scalar arithmetic over
 * numeric lexical captures and intact native Math. No field read, callback,
 * getter or application helper can observe the deferred per-ball writes. */
export function tryMarchingCubesBallLoop(token,object,values,palette) {
  const recipe=ballRecipes.get(token),owner=ballOwners.get(object);
  if(!recipe||!owner?.record?.balls)return false;
  let stats=ballUpdates.get(object);
  if(!stats){stats={wasmCalls:0,balls:0,fallbackCalls:0,lastFailure:null};ballUpdates.set(object,stats);}
  const fallback=reason=>{stats.fallbackCalls++;stats.lastFailure=reason;return false;};
  if(values.length!==recipe.scalarCount+(recipe.condition?1:0))return fallback('MARCHING_CUBES_BALL_CAPTURE');
  for(let i=0;i<recipe.scalarCount;i++)if(typeof values[i]!=='number')return fallback('MARCHING_CUBES_BALL_CAPTURE');
  if(recipe.condition&&typeof values[recipe.scalarCount]!=='boolean')return fallback('MARCHING_CUBES_BALL_CAPTURE');
  const count=values[recipe.countIndex];
  if(!isInteger(count)||count<0||count>4096)return fallback('MARCHING_CUBES_BALL_COUNT');
  if(count===0)return true;
  const record=owner.record.balls;let kernel,input;
  try {
    if(!ballPlatform(recipe))return fallback('MARCHING_CUBES_BALL_PLATFORM');
    input=ballInput(owner,object,palette);if(!input)return fallback('MARCHING_CUBES_BALL_GUARD');
    const initialized=record.attempted;
    kernel=kernelFor(record);if(!kernel)return fallback(record.initializationFailure);
    if(!ballPlatform(recipe))return fallback('MARCHING_CUBES_BALL_PLATFORM');
    if(!initialized&&!(input=ballInput(owner,object,palette)))return fallback('MARCHING_CUBES_BALL_GUARD');
    const balls=new F64(count*8);
    if(!recipe.prepare(balls,values,input.colors,input.colorCount))return fallback('MARCHING_CUBES_BALL_PREPARATION');
    kernel.run(input.field,input.palette,input.size,input.size2,balls,count);
  }catch{return fallback(kernel?.diagnostics.lastGuardFailure??'MARCHING_CUBES_BALL_EXECUTION');}
  // Publication is complete; no fallback or source callbacks occur below.
  stats.wasmCalls++;stats.balls+=count;stats.lastFailure=null;return true;
}
function ballDiagnostics(object) {
  return Object.freeze({...ballUpdates.get(object)??{wasmCalls:0,balls:0,fallbackCalls:0,lastFailure:null},
    kernel:ballOwners.get(object)?.record?.balls?.kernel?.diagnostics??null});
}


/** Attach compiler-owned ABI recipes, never descriptors supplied by the app. */
export function attachMarchingCubesFields(token,specifications,resolveMath) {
  const dispatch=dispatches.get(token);
  if (!dispatch || dispatch.fields || !Array.isArray(specifications) || specifications.length>6 ||
      typeof resolveMath!=='function') throw new TypeError('Invalid marching-cubes field dispatch');
  const fields=new Map();
  for (const specification of specifications) {
    const {name,base64,parameters,locals}=specification;
    if (!['addBall','addPlaneX','addPlaneY','addPlaneZ','blur','reset'].includes(name) || fields.has(name) ||
        typeof base64!=='string' || base64.length>1024*1024 || !Array.isArray(parameters) ||
        parameters.length>64 || !Array.isArray(locals)) throw new TypeError('Invalid field kernel recipe');
    const bindings=parameters.map(parameter=>{
      if (!parameter || !['local','owner','color'].includes(parameter.kind) ||
          typeof parameter.name!=='string' || typeof parameter.property!=='string' ||
          !['f32[]','f64'].includes(parameter.type)) throw new TypeError('Invalid field parameter binding');
      const index=parameter.kind==='local'?locals.indexOf(parameter.property):-1;
      if (parameter.kind==='local' && index<0) throw new TypeError('Missing field local binding');
      return Object.freeze({...parameter,index});
    });
    fields.set(name,{base64,parameters:Object.freeze(bindings),localCount:locals.length,resultPrefixWidths:specification.resultPrefixWidths,
      color:bindings.some(p=>p.kind==='color'),resolveMath,
      attempted:false,kernel:null,initializationFailure:null});
  }
  dispatch.fields=fields;
}

function fieldInput(record,object,locals,color) {
  if (locals.length!==record.localCount || (record.color && !objects.has(color))) return null;
  const values=[];
  for (const parameter of record.parameters) {
    if (parameter.kind==='local') values.push(locals[parameter.index]);
    else {
      // Only genuine registered identities are inspected. Proxy wrappers never
      // pass the WeakSet check, even when their underlying Color is registered.
      const d=own(parameter.kind==='owner'?object:color,parameter.property);
      if (!d) return null;
      values.push(d.value);
    }
  }
  return values;
}

/**
 * Runs just the original method's lifted loop. Its scalar/color/slice prefix has
 * ALREADY executed once in source; refusal resumes that same invocation at its
 * original loop, not at the method entry. No callback, count or geometry write
 * is performed here. The shared host publishes all numerical writes or none.
 */
export function tryMarchingCubesField(receiver,object,token,name,locals,color) {
  let methods=fieldUpdates.get(object);
  if (!methods) {methods=new Map();fieldUpdates.set(object,methods);}
  let stats=methods.get(name);
  if (!stats) {stats={wasmCalls:0,fallbackCalls:0,lastFailure:null,token};methods.set(name,stats);}
  stats.token=token;
  const fallback=reason=>{stats.fallbackCalls++;stats.lastFailure=reason;return false;};
  if (receiver!==object || !objects.has(object)) return fallback('MARCHING_CUBES_RECEIVER');
  const record=dispatches.get(token)?.fields?.get(name);
  if (!record) return fallback('MARCHING_CUBES_UNREGISTERED');
  if (!fieldInput(record,object,locals,color)) return fallback('MARCHING_CUBES_FIELD_GUARD');
  if (!platformIntact()) return fallback('MARCHING_CUBES_PLATFORM');
  const kernel=kernelFor(record);
  if (!kernel) return fallback(record.initializationFailure);
  if (!platformIntact()) return fallback('MARCHING_CUBES_PLATFORM');
  const values=fieldInput(record,object,locals,color);
  if (!values) return fallback('MARCHING_CUBES_FIELD_GUARD');
  try {kernel.run(...values);}
  catch {return fallback(kernel.diagnostics.lastGuardFailure??'MARCHING_CUBES_EXECUTION');}
  // Native output publication has completed. There is no source replay below.
  stats.wasmCalls++;stats.lastFailure=null;
  return true;
}

function fieldDiagnostics(object) {
  const report={};
  for (const [name,stats] of fieldUpdates.get(object)??[]) {
    const record=dispatches.get(stats.token)?.fields?.get(name);
    report[name]=Object.freeze({wasmCalls:stats.wasmCalls,fallbackCalls:stats.fallbackCalls,
      lastFailure:stats.lastFailure,initialized:record?.attempted??false,kernel:record?.kernel?.diagnostics??null});
  }
  return Object.freeze(report);
}
