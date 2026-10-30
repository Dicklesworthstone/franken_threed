/** Recognized track interpolation arithmetic; interval searches, settings,
 * callbacks and public result identities remain in the pinned source methods.
 */
import {createHash} from 'node:crypto';
import * as acorn from 'acorn';
import {sourceBlob} from './marching_cubes_specialization.mjs';
import {normalMethod} from './geometry_normals_compile.mjs';
import {compileNumericKernel} from './numeric_kernel.mjs';

const sources=new Map([
  ['4729c13494ec171cd468ec75b10c7a598d09926a','LinearInterpolant'],
  ['24b3096de529e13fbe0723db76fd84a7028b14e4','CubicInterpolant'],
  ['b14b25b527f4c4c9a8ac81b64a65c6299f936181','BezierInterpolant'],
]);
const coreHash='9edde002b066a9a05676a6127f67735b62baf399bdea529f2f7e31657da769e6';
const fresh=(source,name)=>{let value='__f3d_interpolation_'+name;while(source.includes(value))value+='_';return value;};

export function specializeAnimationInterpolants(source,{runtimeModule,maxMemoryPages=2048,maxIterations=100000000}={}) {
  const role=sources.get(sourceBlob(source)) ??
    (createHash('sha256').update(source).digest('hex')===coreHash?'core':null);
  if(!role)return {changed:false,edits:[],header:'',tail:''};
  const ast=acorn.parse(source,{ecmaVersion:'latest',sourceType:'module'});
  const classes=new Map(ast.body.filter(node=>node.type==='ClassDeclaration').map(node=>[node.id.name,node]));
  const create=fresh(source,'create'),attempt=fresh(source,'try'),edits=[],tails=[],tokens=[],methods=[];
  let wasmBytes=0;
  for(const Class of role==='core'?[...sources.values()]:[role]) {
    const method=normalMethod(classes.get(Class),'interpolate_'),body=method.body.body;
    const last=body.at(-1),loop=body.findLast(node=>node.type==='ForStatement');
    if(last.type!=='ReturnStatement' || last.argument.name!=='result' || !loop)
      throw new TypeError('Missing pinned interpolation boundaries');
    const mode=Class.replace('Interpolant','').toLowerCase(),token=fresh(source,mode);
    let start,args,parameterTypes,prefix='';
    if(mode==='linear') {
      const first=body[0],stride=first.declarations.find(node=>node.id.name==='stride');
      const rest=first.declarations.find(node=>node.id.name==='offset1');
      if(!stride || !rest || source[stride.end]!==',')throw new TypeError('Missing pinned linear prefix');
      // Preserve the three original property reads, once and in order. Numeric
      // coercions remain in the original suffix when the native guard refuses.
      start=stride.end;
      prefix='const '+source.slice(rest.start,first.end)+'\n'+source.slice(first.end,loop.end);
      args=['result','values','stride','i1','t0','t','t1'];
      parameterTypes=['f64[]','f64[]',...Array(5).fill('f64')];
    }else if(mode==='cubic') {
      start=body[1].start;prefix=source.slice(start,loop.end);
      args=['result','values','stride','o1','o0','oP','oN','wP','wN','p','pp','ppp'];
      parameterTypes=['f64[]','f64[]',...Array(10).fill('f64')];
    }else {
      start=body.find(node=>node.type==='VariableDeclaration' && node.declarations[0].id.name==='tangentStride')?.start;
      if(start===undefined)throw new TypeError('Missing pinned Bezier prefix');
      prefix=source.slice(start,loop.end);
      args=['result','values','stride','offset1','offset0','inTangents','outTangents','i1','t0','t','t1'];
      parameterTypes=['f64[]','f64[]',...Array(3).fill('f64'),'f64[]','f64[]',...Array(4).fill('f64')];
    }
    const helperSources=new Map(mode==='bezier'?ast.body.filter(node=>node.type==='FunctionDeclaration' &&
      ['solveBezierParameter','cubicBezier','cubicBezierSlope'].includes(node.id.name)).map(node=>[node.id.name,source.slice(node.start,node.end)]):[]);
    const artifact=compileNumericKernel(`function interpolate(${args.join(',')}){${prefix}}`,{
      parameterTypes,helperSources,generalControl:true,allowMath:mode==='bezier',maxMemoryPages,maxIterations,
      sourceName:'pinned '+Class+'.interpolate_'});
    const branch=`if(${attempt}(${token},${args.join(',')})) return result;\n`;
    edits.push(mode==='linear'?{start,end:start+1,text:';\n'+branch+'const '}:{start,end:start,text:branch});
    tokens.push(token);methods.push(Class+'.interpolate_');wasmBytes+=artifact.wasm.length;
    tails.push(`${token}=${create}(${Class},${JSON.stringify(mode)},${JSON.stringify(Buffer.from(artifact.wasm).toString('base64'))},()=>Math,${maxMemoryPages*65536});`);
  }
  return {changed:true,edits,
    header:`import {createAnimationInterpolantDispatch as ${create},tryAnimationInterpolant as ${attempt}} from ${JSON.stringify(runtimeModule)};var ${tokens.join(',')};\n`,
    tail:'\n'+tails.join('\n')+'\n',
    report:{sourceBlob:sourceBlob(source),upstreamCommit:'148ef33ecb6d2502ff796d4554abd1549c95d519',role,
      compiledMethods:methods,sourceDerived:true,wasmBytes,accelerationClaim:false}};
}
