/** One source-derived numeric transaction for the pinned Object3D child loop. */
import {createHash} from 'node:crypto';
import * as walk from 'acorn-walk';
import {normalClasses,normalMethod} from './geometry_normals_compile.mjs';
import {compileNumericKernel} from './numeric_kernel.mjs';
const coreHash='9edde002b066a9a05676a6127f67735b62baf399bdea529f2f7e31657da769e6';
const fresh=(source,name)=>{let value='__f3d_transforms_'+name;while(source.includes(value))value+='_';return value;};
const slots={matrixAutoUpdate:0,matrixWorldAutoUpdate:1,matrixWorldNeedsUpdate:2};
export const TRANSFORM_BATCH_STRIDE=45;

/** Only the complete pinned core establishes every constructor/method below.
 * Other module arrangements continue through their original source methods. */
export function specializeTransformBatch(source,{runtimeModule,maxMemoryPages=2048,maxIterations=100000000}={}) {
  if(createHash('sha256').update(source).digest('hex')!==coreHash)return {changed:false,edits:[],header:'',tail:''};
  const classes=normalClasses(source),object=classes.get('Object3D'),matrix=classes.get('Matrix4');
  const update=normalMethod(object,'updateMatrix'),world=normalMethod(object,'updateMatrixWorld');
  const loop=world.body.body.find(node=>node.type==='ForStatement');
  const compose=normalMethod(matrix,'compose').body.body,multiply=normalMethod(matrix,'multiplyMatrices').body.body;
  const base=fresh(source,'offset'),childForce=fresh(source,'force');
  function rewrite(node,member,{force=false,booleans=false}={}) {
    const edits=[];
    walk.full(node,value=>{
      if(value.type==='MemberExpression'){const text=member(value);if(text!==null)edits.push({start:value.start,end:value.end,text});}
      else if(value.type==='Identifier' && force && value.name==='force')edits.push({start:value.start,end:value.end,text:childForce});
      else if(value.type==='Literal' && booleans && typeof value.value==='boolean')edits.push({start:value.start,end:value.end,text:value.value?'1':'0'});
    });
    let result=source.slice(node.start,node.end);
    for(const edit of edits.sort((a,b)=>b.start-a.start))result=result.slice(0,edit.start-node.start)+edit.text+result.slice(edit.end-node.start);
    return result;
  }
  const slot=index=>'state['+base+'+'+index+']';
  const arithmetic=(nodes,arrays,fields)=>nodes.map(node=>rewrite(node,value=>{
    const name=value.object.name;
    if(value.computed && Object.hasOwn(arrays,name)) {
      const [array,offset]=arrays[name],index=value.property.value;
      if(!Number.isInteger(index))throw new TypeError('Pinned transform index changed');
      return array+'['+(offset===null?String(index):base+'+'+(offset+index))+']';
    }
    if(!value.computed && Object.hasOwn(fields,name)) {
      const index=fields[name][value.property.name];
      if(index===undefined)throw new TypeError('Pinned transform field changed');
      return slot(index);
    }
    return null;
  })).join('\n');
  const composition=arithmetic(compose.slice(1,-1),{te:['state',13]},
    {position:{x:3,y:4,z:5},quaternion:{_x:6,_y:7,_z:8,_w:9},scale:{x:10,y:11,z:12}});
  const multiplication=arithmetic(multiply.slice(3,-1),{ae:['parent',null],be:['state',13],te:['state',29]},{});
  const expression=node=>rewrite(node,value=>value.object.type==='ThisExpression' && Object.hasOwn(slots,value.property.name)
    ?slot(slots[value.property.name]):null,{force:true,booleans:true});
  function statement(node) {
    if(node.type==='BlockStatement')return '{'+node.body.map(statement).join('\n')+'}';
    if(node.type==='IfStatement') {
      // Every admitted child has this ordinary parent and no pivot.
      if(node.test.left?.object?.type==='ThisExpression' && node.test.left.property.name==='parent')return statement(node.alternate);
      return 'if('+expression(node.test)+')'+statement(node.consequent)+(node.alternate?'else '+statement(node.alternate):'');
    }
    if(node.type==='ExpressionStatement') {
      const call=node.expression;
      if(call.type==='CallExpression') {
        if(call.callee.property.name==='updateMatrix')return '{'+composition+'\n'+expression(update.body.body.at(-1))+'}';
        if(call.callee.property.name==='multiplyMatrices')return '{'+multiplication+'}';
        throw new TypeError('Unrecognized pinned transform call');
      }
      return expression(node);
    }
    throw new TypeError('Unrecognized pinned transform control');
  }
  const index=loop.init.declarations[0].id.name;
  const header=source.slice(loop.start,loop.body.start).replace('children.length','count');
  const kernelSource='function updateTransformChildren(state,parent,count,force){'+header+'{const '+base+'='+index+'*'+TRANSFORM_BATCH_STRIDE+
    ';let '+childForce+'=force;'+world.body.body.slice(0,2).map(statement).join('\n')+'}}';
  const artifact=compileNumericKernel(kernelSource,{parameterTypes:['f64[]','f64[]','f64','f64'],generalControl:true,
    maxMemoryPages,maxIterations,sourceName:'pinned Object3D.updateMatrixWorld child batch'});
  const token=fresh(source,'token'),create=fresh(source,'create'),register=fresh(source,'register');
  const objectAllocation=fresh(source,'object'),numericAllocation=fresh(source,'numeric'),arrayAllocation=fresh(source,'children');
  const attempt=fresh(source,'try_batch'),edits=[];
  for(const name of ['Object3D','Mesh','InstancedMesh','Matrix4']) {
    const cls=classes.get(name);
    edits.push({start:cls.end,end:cls.end,text:'\n'+register+'('+token+','+JSON.stringify(name)+','+name+');\n'});
  }
  const ctor=normalMethod(object,'constructor'),superCall=ctor.body.body[0];
  edits.push({start:superCall.end,end:superCall.end,text:'\n'+objectAllocation+'(this);\n'});
  const children=ctor.body.body.find(node=>node.expression?.left?.property?.name==='children').expression.right;
  edits.push({start:children.start,end:children.end,text:arrayAllocation+'('+source.slice(children.start,children.end)+')'});
  for(const name of ['Vector3','Quaternion']) {
    const body=normalMethod(classes.get(name),'constructor').body;
    edits.push({start:body.start+1,end:body.start+1,text:'\n'+numericAllocation+'(this);\n'});
  }
  edits.push({start:loop.start,end:loop.end,text:'if (!'+attempt+'(this,children,force,'+token+')) {\n'+source.slice(loop.start,loop.end)+'\n}'});
  return {changed:true,edits,tail:'',header:
    'import {createTransformBatchDispatch as '+create+',registerTransformBatchClass as '+register+
    ',registerTransformObject as '+objectAllocation+',registerTransformNumeric as '+numericAllocation+
    ',registerTransformChildren as '+arrayAllocation+',tryTransformBatch as '+attempt+'} from '+JSON.stringify(runtimeModule)+';\n'+
    'var '+token+'='+create+'('+JSON.stringify(Buffer.from(artifact.wasm).toString('base64'))+','+TRANSFORM_BATCH_STRIDE+','+maxMemoryPages+');\n',
    report:{upstreamCommit:'148ef33ecb6d2502ff796d4554abd1549c95d519',sourceDerived:true,scope:'guarded-direct-leaf-children',
      compiledMethods:['Object3D.updateMatrixWorld'],wasmBytes:artifact.wasm.length,accelerationClaim:false}};
}
