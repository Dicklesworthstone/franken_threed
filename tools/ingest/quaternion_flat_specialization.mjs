/** Pinned flat-quaternion arithmetic used by ordinary animation tracks/mixers.
 * Public static methods and their fallback bodies retain their identities.
 * acos/sin are evaluated by the source JavaScript Math implementation; the
 * remaining source arithmetic and four destination stores execute in Wasm.
 */
import {createHash} from 'node:crypto';
import * as walk from 'acorn-walk';
import {sourceBlob} from './marching_cubes_specialization.mjs';
import {normalClasses,normalMethod} from './geometry_normals_compile.mjs';
import {compileNumericKernel} from './numeric_kernel.mjs';

const quaternionBlob='c9faca31665218cee87a7110e2badc7ba868beca';
const coreHash='9edde002b066a9a05676a6127f67735b62baf399bdea529f2f7e31657da769e6';
const fresh=(source,name)=>{let value='__f3d_quaternion_'+name;while(source.includes(value))value+='_';return value;};

export function quaternionFlatSourceRole(source) {
  if(sourceBlob(source)===quaternionBlob)return 'quaternion';
  return createHash('sha256').update(source).digest('hex')===coreHash?'core':null;
}

/** Extract the actual source statements, including its sign/equality and
 * near-angle decisions. Only the four unsupported transcendental calls become
 * precomputed Number inputs. No trigonometric approximation or reordered dot
 * product is introduced. The private preparation function sees four-component
 * snapshots after callback-free native-storage guards, never application views.
 */
export function deriveQuaternionFlatKernels(source) {
  const quaternion=normalClasses(source).get('Quaternion');
  const slerp=normalMethod(quaternion,'slerpFlat'),multiply=normalMethod(quaternion,'multiplyQuaternionsFlat');
  const calls=[];
  walk.simple(slerp.body,{CallExpression(node){
    if(node.callee.type==='MemberExpression' && !node.callee.computed &&
        node.callee.object.name==='Math' && ['acos','sin'].includes(node.callee.property.name))calls.push(node);
  }});
  calls.sort((a,b)=>a.start-b.start);
  if(calls.length!==4 || calls[0].callee.property.name!=='acos')
    throw new TypeError('Missing verified quaternion transcendental boundaries');
  const values=fresh(source,'math_values');
  function replace(start,end,prepare=false) {
    let result=source.slice(start,end);
    for(let i=calls.length-1;i>=0;i--) {
      const call=calls[i];if(call.start<start || call.end>end)continue;
      const input=values+'['+i+']';
      const text=prepare?'('+input+'='+source.slice(call.start,call.end)+')':input;
      result=result.slice(0,call.start-start)+text+result.slice(call.end-start);
    }
    return result;
  }
  const condition=slerp.body.body.find(node=>node.type==='IfStatement');
  // The first sign-correction if is followed by s and then the interpolation
  // branch. Locate the latter by its contained acos call, not a numeric literal.
  const interpolation=condition.consequent.body.find(node=>node.type==='IfStatement' &&
    calls[0].start>node.start && calls[0].end<node.end);
  if(!interpolation)throw new TypeError('Missing verified quaternion interpolation branch');
  const trigStatements=interpolation.consequent.body.filter(node=>calls.some(call=>call.start>=node.start && call.end<=node.end));
  if(trigStatements.length!==4)throw new TypeError('Unexpected quaternion trigonometric statement shape');
  const prepareBody=source.slice(slerp.body.start+1,condition.start)+
    source.slice(condition.start,interpolation.start)+
    source.slice(interpolation.start,interpolation.consequent.start+1)+
    replace(trigStatements[0].start,trigStatements.at(-1).end,true)+'\n}\n}';
  const parameters=slerp.params.map(node=>node.name);
  const slerpSource='function slerpFlat('+parameters.join(',')+','+values+')'+replace(slerp.body.start,slerp.body.end);
  const last=multiply.body.body.at(-1);
  if(last.type!=='ReturnStatement' || last.argument.name!=='dst')throw new TypeError('Unexpected quaternion multiply return');
  const multiplySource='function multiplyQuaternionsFlat('+multiply.params.map(node=>node.name).join(',')+'){'+
    source.slice(multiply.body.start+1,last.start)+'}';
  return {slerp,multiply,values,prepareBody,slerpSource,multiplySource};
}

export function specializeQuaternionFlat(source,{runtimeModule,maxMemoryPages=2048,maxIterations=100000000}={}) {
  const role=quaternionFlatSourceRole(source);
  if(!role)return {changed:false,edits:[],header:'',tail:''};
  const derived=deriveQuaternionFlatKernels(source),variants={};
  for(const name of ['slerpFlat','multiplyQuaternionsFlat']) {
    const slerp=name==='slerpFlat';
    const artifact=compileNumericKernel(slerp?derived.slerpSource:derived.multiplySource,{
      parameterTypes:['f64[]','f64','f64[]','f64','f64[]','f64',...(slerp?['f64','f64[]']:[])],
      generalControl:true,allowMath:true,maxMemoryPages,maxIterations,sourceName:'pinned Quaternion.'+name});
    variants[name]=Buffer.from(artifact.wasm).toString('base64');
  }
  const token=fresh(source,'dispatch'),create=fresh(source,'create'),attempt=fresh(source,'try'),prepare=fresh(source,'prepare');
  const edits=[['slerpFlat',derived.slerp,'return;'],['multiplyQuaternionsFlat',derived.multiply,'return dst;']].map(([name,method,result])=>({
    start:method.body.start+1,end:method.body.start+1,
    text:`\nif (${attempt}(${token},${JSON.stringify(name)},${method.params.map(node=>node.name).join(',')})) ${result}\n`,
  }));
  return {changed:true,edits,
    header:`import {createQuaternionFlatDispatch as ${create},tryQuaternionFlat as ${attempt}} from ${JSON.stringify(runtimeModule)};var ${token};\n`,
    tail:`\nfunction ${prepare}(src0,src1,t,${derived.values}) {const srcOffset0=0,srcOffset1=0;${derived.prepareBody}\n}\n`+
      `${token}=${create}(Quaternion,${JSON.stringify(variants)},${prepare},()=>Math);\n`,
    report:{sourceBlob:sourceBlob(source),upstreamCommit:'148ef33ecb6d2502ff796d4554abd1549c95d519',
      role,compiledMethods:Object.keys(variants),sourceDerived:true,retainedMath:['acos','sin'],
      wasmBytes:Object.values(variants).reduce((sum,value)=>sum+Buffer.from(value,'base64').length,0),accelerationClaim:false}};
}
