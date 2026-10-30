/** One source-derived transaction for the pinned skeletal palette loop. The
 * original texture upload request stays after synchronous materialization. */
import {createHash} from 'node:crypto';
import * as acorn from 'acorn';
import * as walk from 'acorn-walk';
import {sourceBlob} from './marching_cubes_specialization.mjs';
import {normalClasses,normalMethod} from './geometry_normals_compile.mjs';
import {compileNumericKernel} from './numeric_kernel.mjs';

const skeletonBlob='31384a9bcd40eb0b57063eeb1f1434dccb39ce82';
const objectBlob='a0a09156f7e781130de6a361cf2c443b8c27d925';
const matrixBlob='6522a6b57300d20bf6a36fbcbb3862e81291d060';
const coreHash='9edde002b066a9a05676a6127f67735b62baf399bdea529f2f7e31657da769e6';
const fresh=(source,name)=>{let value='__f3d_skeleton_'+name;while(source.includes(value))value+='_';return value;};
const unchanged=()=>({changed:false,edits:[],header:'',tail:''});

export function skeletonUpdateSourceRole(source) {
  const blob=sourceBlob(source);
  if(blob===skeletonBlob)return 'skeleton';
  if(blob===objectBlob)return 'object';
  return createHash('sha256').update(source).digest('hex')===coreHash?'core':null;
}

function witnessArrays(node,register,edits) {
  walk.simple(node,{ArrayExpression(value){
    edits.push({start:value.start,end:value.start,text:'('+register+'('},
      {start:value.end,end:value.end,text:'))'});
  }});
}

export function specializeSkeletonUpdate(source,{runtimeModule,matrixSource,maxMemoryPages=2048,maxIterations=100000000}={}) {
  const role=skeletonUpdateSourceRole(source);
  if(!role)return unchanged();
  const classes=normalClasses(source),register=fresh(source,'object'),array=fresh(source,'array'),edits=[],imports=[];
  if(role==='object'||role==='core') {
    const ctor=normalMethod(classes.get('Object3D'),'constructor'),superCall=ctor.body.body[0];
    edits.push({start:superCall.end,end:superCall.end,text:'\n'+register+'(this);\n'});
    imports.push('registerSkeletonObject as '+register);
  }
  if(role==='object')return {changed:true,edits,header:'import {'+imports+'} from '+JSON.stringify(runtimeModule)+';\n',tail:''};
  if(role!=='core'&&sourceBlob(matrixSource??'')!==matrixBlob)return unchanged();
  const matrixText=role==='core'?source:matrixSource,matrix=normalClasses(matrixText).get('Matrix4');
  const skeleton=classes.get('Skeleton'),ctor=normalMethod(skeleton,'constructor'),update=normalMethod(skeleton,'update');
  const loop=update.body.body.find(node=>node.type==='ForStatement');
  const multiply=normalMethod(matrix,'multiplyMatrices'),toArray=normalMethod(matrix,'toArray');
  const scratch=loop.body.body[1].expression.callee.object.name,identity=loop.body.body[0].declarations[0].init.alternate.name;
  const index=loop.init.declarations[0].id.name,base=fresh(source,'base');
  function rewrite(node,arrays) {
    const changes=[];
    walk.simple(node,{MemberExpression(value){
      if(value.computed&&Object.hasOwn(arrays,value.object.name))
        changes.push({start:value.start,end:value.end,text:arrays[value.object.name](value.property)});
    }});
    let text=matrixText.slice(node.start,node.end);
    for(const change of changes.sort((a,b)=>b.start-a.start))
      text=text.slice(0,change.start-node.start)+change.text+text.slice(change.end-node.start);
    return text;
  }
  const product=multiply.body.body.slice(3,-1).map(node=>rewrite(node,{
    ae:value=>'state['+base+'+'+value.value+']',
    be:value=>'state['+base+'+'+(16+value.value)+']',
    te:value=>'state['+base+'+'+(32+value.value)+']',
  })).join('\n');
  const stores=toArray.body.body.slice(1,-1).map(node=>rewrite(node,{
    te:value=>'state['+base+'+'+(32+value.value)+']',
    array:value=>'result['+matrixText.slice(value.start,value.end)+']',
  })).join('\n');
  const header=source.slice(loop.start,loop.body.start).replace('bones.length','count');
  const kernelSource='function updateSkeleton(state,result,count){'+header+'{const '+base+'='+index+'*48;'+
    product+'\nconst offset='+index+'*16;'+stores+'}}';
  const variants={};let wasmBytes=0;
  for(const type of ['f32','f64']) {
    const artifact=compileNumericKernel(kernelSource,{parameterTypes:['f64[]',type+'[]','f64'],
      generalControl:true,maxMemoryPages,maxIterations,sourceName:'pinned Skeleton.update / Matrix4 palette loop'});
    variants[type]=Buffer.from(artifact.wasm).toString('base64');wasmBytes+=artifact.wasm.length;
  }
  const owner=fresh(source,'owner'),slice=fresh(source,'slice'),token=fresh(source,'token');
  const create=fresh(source,'create'),attempt=fresh(source,'try');
  imports.push('registerSkeletonOwner as '+owner,'registerSkeletonArray as '+array,'trySkeletonBoneSlice as '+slice,
    'createSkeletonUpdateDispatch as '+create,'trySkeletonUpdate as '+attempt);
  edits.push({start:ctor.body.start+1,end:ctor.body.start+1,text:'\n'+owner+'(this);\n'});
  const bones=ctor.body.body.find(node=>node.expression?.left?.property?.name==='bones').expression.right;
  edits.push({start:bones.start,end:bones.end,text:'('+slice+'(bones)??'+source.slice(bones.start,bones.end)+')'});
  witnessArrays(skeleton,array,edits);
  edits.push({start:loop.start,end:loop.end,text:'if(!'+attempt+'(this,'+token+',bones,boneInverses,boneMatrices,'+
    scratch+','+identity+')){'+source.slice(loop.start,loop.end)+'}'});
  return {changed:true,edits,header:'import {'+imports.join(',')+'} from '+JSON.stringify(runtimeModule)+';var '+token+';\n',
    tail:'\n'+token+'='+create+'(Skeleton,Matrix4,'+JSON.stringify(variants)+','+maxMemoryPages*65536+');\n',
    report:{sourceBlob:sourceBlob(source),upstreamCommit:'148ef33ecb6d2502ff796d4554abd1549c95d519',role,
      compiledMethods:['Skeleton.update'],sourceDerived:true,scope:'whole-skeletal-palette-loop',wasmBytes,accelerationClaim:false}};
}

/** Allocation witnesses authorize inspection, never a replacement class name.
 * Only modules mentioning Skeleton are candidates; unknown constructors and
 * arbitrary values still fail the runtime owner/matrix guards. */
export function specializeSkeletonArrayWitnesses(source,{runtimeModule}={}) {
  if(!source.includes('Skeleton')||skeletonUpdateSourceRole(source))return unchanged();
  let ast;try{ast=acorn.parse(source,{ecmaVersion:'latest',sourceType:'module'});}catch{return unchanged();}
  let dynamic=false;walk.simple(ast,{CallExpression(node){if(node.callee.type==='Identifier'&&node.callee.name==='eval')dynamic=true;}});
  if(dynamic)return unchanged();
  const register=fresh(source,'array_witness'),edits=[];witnessArrays(ast,register,edits);
  return edits.length?{changed:true,edits,header:'import {registerSkeletonArray as '+register+'} from '+JSON.stringify(runtimeModule)+';\n',tail:''}:unchanged();
}
