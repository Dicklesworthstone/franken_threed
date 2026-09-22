/**
 * Recognize a source-pinned library kernel, not an application filename/trace.
 * Only verified numeric loops are bypassed. The original public methods and
 * their prefixes/publication tails retain all callbacks, conversions and errors.
 * Build-time parsing never evaluates the addon or its table initializers.
 */
import {createHash} from 'node:crypto';
import * as acorn from 'acorn';
import {compileMarchingCubesKernel} from './marching_cubes_compile.mjs';
import {compileMarchingCubesFields} from './marching_cubes_fields.mjs';
const MARCHING_CUBES_PIN='148ef33ecb6d2502ff796d4554abd1549c95d519';
export const MARCHING_CUBES_SOURCE_BLOB='29a405be3eae30a7e2b1ff04827068921d31d5dc';
export const EVENT_DISPATCHER_SOURCE_BLOB='ac793ea9081486b1b1e834c9df2480d18fb90e0c';
export const COLOR_SOURCE_BLOB='f42d664478a6952dd1889eebc9a16736a2c614ab';
// Checked-in pinned build, also recorded in evidence/01.1/build_sha256.txt.
const CORE_BUILD_SHA256='9edde002b066a9a05676a6127f67735b62baf399bdea529f2f7e31657da769e6';
export function sourceBlob(source) {
  const bytes=Buffer.from(source);
  return createHash('sha1').update(`blob ${bytes.length}\0`).update(bytes).digest('hex');
}
function parse(source) {return acorn.parse(source,{ecmaVersion:'latest',sourceType:'module'});}
function fresh(source,role) {
  let name='__f3d_marching_'+role;
  while (source.includes(name)) name+='_';
  return name;
}
function constructor(ast,name) {
  const declaration=ast.body.find(node=>node.type==='ClassDeclaration' && node.id?.name===name);
  const method=declaration?.body.body.find(node=>node.kind==='constructor');
  if (!method) throw new TypeError(`Missing verified ${name} constructor`);
  return method.value;
}
const thisMember=(node,name)=>node?.type==='MemberExpression' && !node.computed &&
  node.object.type==='ThisExpression' && node.property.name===name;

/**
 * expectedBlob is an explicit, independently audited source anchor. Its default
 * is the pinned upstream source. A mismatch leaves source byte-for-byte intact.
 * Supplying another hash does NOT establish that another algorithm is equivalent.
 */
export function specializeMarchingCubesModule(source,{
  runtimeModule='./marching_cubes_adapter.mjs',expectedBlob=MARCHING_CUBES_SOURCE_BLOB,maxMemoryPages=2048,maxIterations=100000000,
}={}) {
  if (typeof source!=='string' || typeof runtimeModule!=='string' || !runtimeModule ||
      typeof expectedBlob!=='string' || !/^[a-f0-9]{40}$/.test(expectedBlob)) throw new TypeError('Invalid marching-cubes source options');
  const hash=sourceBlob(source),report={upstreamCommit:MARCHING_CUBES_PIN,sourceBlob:hash,
    route:'retained-js',accelerationClaim:false,reason:null};
  if (hash!==expectedBlob) return {changed:false,code:source,report:{...report,reason:'SOURCE_PIN_MISMATCH'}};
  const ast=parse(source),ctor=constructor(ast,'MarchingCubes');
  const assignments=ctor.body.body.filter(node=>node.type==='ExpressionStatement').map(node=>node.expression);
  const update=assignments.find(node=>node.type==='AssignmentExpression' && node.operator==='=' && thisMember(node.left,'update'))?.right;
  if (update?.type!=='FunctionExpression' || update.params.length || update.async || update.generator) throw new TypeError('Invalid verified update function');
  const body=update.body.body;
  const tail=body.findIndex(node=>node.type==='ExpressionStatement' && node.expression.type==='CallExpression' &&
    node.expression.callee.type==='MemberExpression' && !node.expression.callee.computed &&
    node.expression.callee.property.name==='setDrawRange' && thisMember(node.expression.callee.object,'geometry'));
  if (tail<1 || !assignments.some(node=>node.type==='CallExpression' && thisMember(node.callee,'init'))) {
    throw new TypeError('Missing verified numeric/publication boundary');
  }
  const artifact=compileMarchingCubesKernel({maxMemoryPages,maxIterations});
  // Fixture-specific expectedBlob overrides do not authorize field lifting.
  // Its own pin is independent and covers every original loop and prefix.
  const fields=hash===MARCHING_CUBES_SOURCE_BLOB?compileMarchingCubesFields(source,{maxMemoryPages,maxIterations}):[];
  const tryName=fresh(source,'try'),createName=fresh(source,'create'),token=fresh(source,'dispatch');
  const fieldTry=fresh(source,'field_try'),attach=fresh(source,'attach_fields');
  const prefix=source.slice(update.body.start+1,body[tail].start);
  const edits=[{start:update.body.start+1,end:body[tail].start,
    text:`\nif (!${tryName}(this, scope, ${token}, vlist, nlist, clist)) {${prefix}\n}\n`}];
  for (const field of fields) {
    const {start,end}=field.sourceSpan;
    edits.push({start,end,text:`if (!${fieldTry}(this, scope, ${token}, ${JSON.stringify(field.name)}, `+
      `[${field.locals.join(',')}], ${field.color?'ballColor':'null'})) {\n${source.slice(start,end)}\n}`});
  }
  let transformed=source;
  for (const edit of edits.sort((a,b)=>b.start-a.start)) {
    transformed=transformed.slice(0,edit.start)+edit.text+transformed.slice(edit.end);
  }
  const fieldImports=fields.length?`, tryMarchingCubesField as ${fieldTry}, attachMarchingCubesFields as ${attach}`:'';
  const header=`import { tryMarchingCubesUpdate as ${tryName}, createMarchingCubesDispatch as ${createName}${fieldImports} } from ${JSON.stringify(runtimeModule)};\nvar ${token};\n`;
  // var is initialized before an ESM cycle can call the source method. Before
  // table/dispatch registration the original function supplies its own TDZ/error.
  let registration=`\n${token}=${createName}(${JSON.stringify(Buffer.from(artifact.wasm).toString('base64'))},edgeTable,triTable);\n`;
  if (fields.length) {
    const specifications=fields.map(field=>({name:field.name,parameters:field.parameters,locals:field.locals,
      base64:Buffer.from(field.wasm).toString('base64')}));
    registration+=`${attach}(${token},${JSON.stringify(specifications)},()=>Math);\n`;
  }
  return {changed:true,code:header+transformed+registration,wasm:artifact.wasm,report:{...report,
    route:'retained-addon-with-guarded-wasm-polygonizer',wasmBytes:artifact.wasm.length,
    compiledFieldKernels:fields.length,fieldWasmBytes:fields.reduce((sum,field)=>sum+field.wasm.length,0),
    fieldKernels:fields.map(field=>({name:field.name,wasmBytes:field.wasm.length,sourceSpan:field.sourceSpan,
      numericSemantics:field.manifest.numericSemantics})),
    numericSourceSpan:{start:update.body.start+1,end:body[tail].start},
    preservedPublicationSpan:{start:body[tail].start,end:update.body.end-1}}};
}

/**
 * Register genuine allocations in the verified base EventDispatcher and Color. Checking
 * a constructor chain later is unsound: a derived constructor may return a Proxy
 * or mutate its superclass while executing. A WeakSet records the fresh identity
 * at allocation instead, and misses proxies without triggering their traps.
 * The pinned core build is recognized too, so normal `three` package imports work.
 */
export function specializeMarchingCubesBase(source,{
  runtimeModule='./marching_cubes_adapter.mjs',expectedBlob=EVENT_DISPATCHER_SOURCE_BLOB,
}={}) {
  if (typeof source!=='string' || typeof runtimeModule!=='string' || !runtimeModule ||
      typeof expectedBlob!=='string' || !/^[a-f0-9]{40}$/.test(expectedBlob)) throw new TypeError('Invalid base source options');
  const hash=sourceBlob(source),defaultPin=expectedBlob===EVENT_DISPATCHER_SOURCE_BLOB;
  const core=defaultPin && createHash('sha256').update(source).digest('hex')===CORE_BUILD_SHA256;
  const colorSource=defaultPin && hash===COLOR_SOURCE_BLOB;
  if (hash!==expectedBlob && !core && !colorSource) return {changed:false,code:source};
  const ast=parse(source),register=fresh(source,'register'),edits=[],registeredClasses=[];
  if (!colorSource) {
    const base=ast.body.find(node=>node.type==='ClassDeclaration' && node.id?.name==='EventDispatcher');
    if (!base || base.superClass || base.body.body.some(node=>node.kind==='constructor')) throw new TypeError('Expected verified default base constructor');
    edits.push({at:base.body.start+1,text:`\nconstructor() { ${register}(this); }\n`});
    registeredClasses.push('EventDispatcher');
  }
  if (core || colorSource) {
    const color=ast.body.find(node=>node.type==='ClassDeclaration' && node.id?.name==='Color');
    if (!color || color.superClass) throw new TypeError('Expected verified base Color');
    const ctor=constructor(ast,'Color');
    // Color.set can be overridden and return a foreign object or Proxy. Record
    // the true allocation BEFORE that call, never the returned identity.
    edits.push({at:ctor.body.start+1,text:`\n${register}(this);\n`});
    registeredClasses.push('Color');
  }
  let code=source;
  for (const edit of edits.sort((a,b)=>b.at-a.at)) code=code.slice(0,edit.at)+edit.text+code.slice(edit.at);
  return {changed:true,registeredClasses,code:`import { registerMarchingCubesObject as ${register} } from ${JSON.stringify(runtimeModule)};\n`+code};
}
