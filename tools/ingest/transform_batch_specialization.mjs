/** One source-derived numeric transaction for a pinned Object3D descendant loop. */
import {createHash} from 'node:crypto';
import * as walk from 'acorn-walk';
import {sourceBlob} from './marching_cubes_specialization.mjs';
import {normalClasses,normalMethod} from './geometry_normals_compile.mjs';
import {compileNumericKernel} from './numeric_kernel.mjs';
const coreHash='9edde002b066a9a05676a6127f67735b62baf399bdea529f2f7e31657da769e6';
const sourceBlobs={Object3D:'a0a09156f7e781130de6a361cf2c443b8c27d925',
  Group:'5acef7bc677d6830b514a56e6a0b8bd24d3e27fd',Scene:'21bcb96007df7bc1bfa98c5c6d613205de96795d',
  Bone:'6078326b104657e9357e4cbeea51ab59e5d50f3d',Mesh:'b80d3de8f403a696145ab131581b17263e510c59',
  InstancedMesh:'d55dc5d1311200f9ea7815c7cb7669a593f73948',
  Vector3:'b2e4c4e65cc50afc89b6cd8f8e706a0d1001d71a',Quaternion:'c9faca31665218cee87a7110e2badc7ba868beca',
  Matrix4:'6522a6b57300d20bf6a36fbcbb3862e81291d060'};
const nodeClasses=['Object3D','Group','Scene','Bone','Mesh','InstancedMesh','Matrix4'];
const unchanged=()=>({changed:false,edits:[],header:'',tail:''});
export function transformBatchSourceRole(source) {
  const blob=sourceBlob(source);
  for(const [role,expected] of Object.entries(sourceBlobs))if(blob===expected)return role;
  return createHash('sha256').update(source).digest('hex')===coreHash?'core':null;
}
const fresh=(source,name)=>{let value='__f3d_transforms_'+name;while(source.includes(value))value+='_';return value;};
const slots={matrixAutoUpdate:0,matrixWorldAutoUpdate:1,matrixWorldNeedsUpdate:2};
// TRS/flags and local/world matrices occupy the first 45 values. The parent
// row (-1 for the already-updated owner) and outgoing force follow. The final
// four values carry an optional pivot flag and its source-observable XYZ values.
export const TRANSFORM_BATCH_STRIDE=51;

/** Source modules register their own prototypes and allocation witnesses. An
 * Object3D transaction is derived only from its verified Matrix4 dependency;
 * the runtime also binds the actual imported class, never a class-name lookup. */
export function specializeTransformBatch(source,{runtimeModule,matrixSource,maxMemoryPages=2048,maxIterations=100000000}={}) {
  const role=transformBatchSourceRole(source);
  if(!role)return unchanged();
  const classes=normalClasses(source),edits=[];
  const register=fresh(source,'register'),numericAllocation=fresh(source,'numeric');
  if(!['core','Object3D'].includes(role)) {
    const imports=[];
    if(nodeClasses.includes(role)) {
      const cls=classes.get(role);
      edits.push({start:cls.end,end:cls.end,text:'\n'+register+'(null,'+JSON.stringify(role)+','+role+');\n'});
      imports.push('registerTransformBatchClass as '+register);
    }
    if(['Vector3','Quaternion'].includes(role)) {
      const body=normalMethod(classes.get(role),'constructor').body;
      edits.push({start:body.start+1,end:body.start+1,text:'\n'+numericAllocation+'(this);\n'});
      imports.push('registerTransformNumeric as '+numericAllocation);
    }
    return {changed:true,edits,header:'import {'+imports.join(',')+'} from '+JSON.stringify(runtimeModule)+';\n',tail:''};
  }
  if(role==='core')matrixSource=source;
  else if(transformBatchSourceRole(matrixSource??'')!=='Matrix4')return unchanged();
  const object=classes.get('Object3D'),matrix=(role==='core'?classes:normalClasses(matrixSource)).get('Matrix4');
  const update=normalMethod(object,'updateMatrix'),world=normalMethod(object,'updateMatrixWorld');
  const selective=normalMethod(object,'updateWorldMatrix');
  const loop=world.body.body.find(node=>node.type==='ForStatement');
  const selectiveChildren=selective.body.body.find(node=>node.type==='IfStatement' &&
    node.test.left?.name==='updateChildren');
  const selectiveLoop=selectiveChildren.consequent.body.find(node=>node.type==='ForStatement');
  const compose=normalMethod(matrix,'compose').body.body,multiply=normalMethod(matrix,'multiplyMatrices').body.body;
  const base=fresh(source,'offset'),childForce=fresh(source,'force');
  const parentIndex=fresh(source,'parent'),parentBase=fresh(source,'parent_offset');
  function rewrite(node,member,{force=false,booleans=false,inputSource=source}={}) {
    const edits=[];
    walk.full(node,value=>{
      if(value.type==='MemberExpression'){const text=member(value);if(text!==null)edits.push({start:value.start,end:value.end,text});}
      else if(value.type==='Identifier' && force && value.name==='force')edits.push({start:value.start,end:value.end,text:childForce});
      else if(value.type==='Literal' && booleans && typeof value.value==='boolean')edits.push({start:value.start,end:value.end,text:value.value?'1':'0'});
    });
    let result=inputSource.slice(node.start,node.end);
    for(const edit of edits.sort((a,b)=>b.start-a.start))result=result.slice(0,edit.start-node.start)+edit.text+result.slice(edit.end-node.start);
    return result;
  }
  const slot=index=>'state['+base+'+'+index+']';
  const arithmetic=(nodes,arrays,fields,inputSource=matrixSource)=>nodes.map(node=>rewrite(node,value=>{
    const name=value.object.name;
    if(value.computed && Object.hasOwn(arrays,name)) {
      const index=value.property.value;
      if(!Number.isInteger(index))throw new TypeError('Pinned transform index changed');
      if(typeof arrays[name]==='function')return arrays[name](index);
      const [array,offset]=arrays[name];
      return array+'['+(offset===null?String(index):base+'+'+(offset+index))+']';
    }
    if(!value.computed && Object.hasOwn(fields,name)) {
      const index=fields[name][value.property.name];
      if(index===undefined)throw new TypeError('Pinned transform field changed');
      return slot(index);
    }
    return null;
  },{inputSource})).join('\n');
  const composition=arithmetic(compose.slice(1,-1),{te:['state',13]},
    {position:{x:3,y:4,z:5},quaternion:{_x:6,_y:7,_z:8,_w:9},scale:{x:10,y:11,z:12}});
  const pivotBody=update.body.body.find(node=>node.type==='IfStatement').consequent.body;
  // Keep the pinned source's operation ordering (including each += store).
  // The te binding alone is replaced by the already-borrowed local matrix.
  const pivot=arithmetic(pivotBody.filter(node=>node.type!=='VariableDeclaration' ||
    !node.declarations.some(value=>value.id.name==='te')),{te:['state',13]},
    {pivot:{x:48,y:49,z:50}},source);
  const multiplication=arithmetic(multiply.slice(3,-1),{
    ae:index=>'('+parentIndex+'<0?parent['+index+']:state['+parentBase+'+'+(29+index)+'])',
    be:['state',13],te:['state',29]},{});
  const expression=node=>rewrite(node,value=>value.object.type==='ThisExpression' && Object.hasOwn(slots,value.property.name)
    ?slot(slots[value.property.name]):null,{force:true,booleans:true});
  function statement(node) {
    if(node.type==='BlockStatement')return '{'+node.body.map(statement).join('\n')+'}';
    if(node.type==='IfStatement') {
      // Every admitted descendant has a guarded ordinary parent.
      if(node.test.left?.object?.type==='ThisExpression' && node.test.left.property.name==='parent')return statement(node.alternate);
      return 'if('+expression(node.test)+')'+statement(node.consequent)+(node.alternate?'else '+statement(node.alternate):'');
    }
    if(node.type==='ExpressionStatement') {
      const call=node.expression;
      if(call.type==='CallExpression') {
        if(call.callee.property.name==='updateMatrix')return '{'+composition+'\nif('+slot(47)+'){'+pivot+'}\n'+expression(update.body.body.at(-1))+'}';
        if(call.callee.property.name==='multiplyMatrices')return '{'+multiplication+'}';
        throw new TypeError('Unrecognized pinned transform call');
      }
      return expression(node);
    }
    throw new TypeError('Unrecognized pinned transform control');
  }
  const index=loop.init.declarations[0].id.name;
  const header=source.slice(loop.start,loop.body.start).replace('children.length','count');
  const body=world.body.body.slice(0,2).map(statement).join('\n');
  // Both pinned entry points have the same local/dirty/force propagation after
  // the selective method has updated its ancestors. Derive and compare that
  // source instead of implementing a second numerical algorithm.
  if(selective.body.body.slice(2,4).map(statement).join('\n')!==body)
    throw new TypeError('Pinned world-update arithmetic differs between entry points');
  const kernelSource='function updateTransformChildren(state,parent,count,force){'+header+'{const '+base+'='+index+'*'+TRANSFORM_BATCH_STRIDE+
    ';const '+parentIndex+'='+slot(45)+';const '+parentBase+'='+parentIndex+'*'+TRANSFORM_BATCH_STRIDE+
    ';let '+childForce+'='+parentIndex+'<0?force:state['+parentBase+'+46];'+
    body+'\n'+slot(46)+'='+childForce+';}}';
  const artifact=compileNumericKernel(kernelSource,{parameterTypes:['f64[]','f64[]','f64','f64'],generalControl:true,
    maxMemoryPages,maxIterations,sourceName:'pinned Object3D.updateMatrixWorld subtree batch'});
  const token=fresh(source,'token'),create=fresh(source,'create');
  const objectAllocation=fresh(source,'object'),arrayAllocation=fresh(source,'children');
  const attempt=fresh(source,'try_batch');
  const initialize=token+'='+create+'('+JSON.stringify(Buffer.from(artifact.wasm).toString('base64'))+','+
    TRANSFORM_BATCH_STRIDE+','+maxMemoryPages+(role==='Object3D'?',Matrix4':'')+');\n';
  for(const name of role==='core'?nodeClasses:['Object3D']) {
    const cls=classes.get(name);
    edits.push({start:cls.end,end:cls.end,text:'\n'+(role==='Object3D'?initialize:'')+
      register+'('+token+','+JSON.stringify(name)+','+name+');\n'});
  }
  const ctor=normalMethod(object,'constructor'),superCall=ctor.body.body[0];
  edits.push({start:superCall.end,end:superCall.end,text:'\n'+objectAllocation+'(this);\n'});
  const children=ctor.body.body.find(node=>node.expression?.left?.property?.name==='children').expression.right;
  edits.push({start:children.start,end:children.end,text:arrayAllocation+'('+source.slice(children.start,children.end)+')'});
  for(const name of role==='core'?['Vector3','Quaternion']:[]) {
    const body=normalMethod(classes.get(name),'constructor').body;
    edits.push({start:body.start+1,end:body.start+1,text:'\n'+numericAllocation+'(this);\n'});
  }
  edits.push({start:loop.start,end:loop.end,text:'if (!'+attempt+'(this,children,force,'+token+')) {\n'+source.slice(loop.start,loop.end)+'\n}'});
  edits.push({start:selectiveLoop.start,end:selectiveLoop.end,text:'if (!'+attempt+'(this,children,force,'+token+',"updateWorldMatrix")) {\n'+
    source.slice(selectiveLoop.start,selectiveLoop.end)+'\n}'});
  return {changed:true,edits,tail:'',header:
    'import {createTransformBatchDispatch as '+create+',registerTransformBatchClass as '+register+
    ',registerTransformObject as '+objectAllocation+',registerTransformNumeric as '+numericAllocation+
    ',registerTransformChildren as '+arrayAllocation+',tryTransformBatch as '+attempt+'} from '+JSON.stringify(runtimeModule)+';\n'+
    'var '+(role==='core'?initialize:token+';\n'),
    report:{upstreamCommit:'148ef33ecb6d2502ff796d4554abd1549c95d519',sourceDerived:true,scope:'guarded-ordinary-subtrees',
      role,compiledMethods:['Object3D.updateMatrixWorld','Object3D.updateWorldMatrix'],wasmBytes:artifact.wasm.length,accelerationClaim:false}};
}
