/** Source-derived instanced rotation/update islands. The original application
 * supplies the loop; pinned Matrix4 arithmetic supplies the closed kernel. */
import {createHash} from 'node:crypto';
import * as acorn from 'acorn';
import * as walk from 'acorn-walk';
import {normalClasses,normalMethod} from './geometry_normals_compile.mjs';
import {compileNumericKernel} from './numeric_kernel.mjs';
import {numericLoopScopeBindings} from './numeric_loop_islands.mjs';

const CORE_HASH='9edde002b066a9a05676a6127f67735b62baf399bdea529f2f7e31657da769e6';
export const INSTANCE_MATRIX_STRIDE=39;
const fresh=(source,name)=>{let value='__f3d_instance_update_'+name;while(source.includes(value))value+='_';return value;};
const property=node=>node?.type!=='MemberExpression'||node.optional?null:
  !node.computed&&node.property.type==='Identifier'?node.property.name:
    node.computed&&node.property.type==='Literal'&&typeof node.property.value==='string'?node.property.value:null;
const identifier=(node,name)=>node?.type==='Identifier'&&(name===undefined||node.name===name);

export function specializeInstancedMatrixBase(source,{runtimeModule,maxMemoryPages=2048,maxIterations=100000000}={}) {
  if(createHash('sha256').update(source).digest('hex')!==CORE_HASH)return {changed:false,edits:[],header:''};
  const classes=normalClasses(source),matrix=classes.get('Matrix4'),mesh=classes.get('InstancedMesh');
  const rotation=normalMethod(matrix,'makeRotationFromEuler'),multiply=normalMethod(matrix,'multiplyMatrices');
  const branch=rotation.body.body.find(node=>node.type==='IfStatement'),orders=[];
  for(let current=branch;current?.type==='IfStatement';current=current.alternate)orders.push(current.test.right.value);
  function rewrite(node,arrays,rotationOrder=false) {
    const edits=[];
    walk.simple(node,{MemberExpression(value){
      if(value.computed&&Object.hasOwn(arrays,value.object.name))
        edits.push({start:value.start,end:value.end,text:arrays[value.object.name](value.property.value)});
      else if(rotationOrder&&property(value)==='order'&&identifier(value.object,'euler'))
        edits.push({start:value.start,end:value.end,text:'state[offset+22]'});
    },Literal(value){if(rotationOrder&&orders.includes(value.value))
      edits.push({start:value.start,end:value.end,text:String(orders.indexOf(value.value))});}});
    let text=source.slice(node.start,node.end);
    for(const edit of edits.sort((a,b)=>b.start-a.start))text=text.slice(0,edit.start-node.start)+edit.text+text.slice(edit.end-node.start);
    return text;
  }
  const rotate=rotation.body.body.slice(rotation.body.body.indexOf(branch),-1)
    .map(node=>rewrite(node,{te:index=>'state[offset+'+(23+index)+']'},true)).join('\n');
  const product=multiply.body.body.slice(3,-1).map(node=>rewrite(node,{
    ae:index=>'(reverse===0?state[offset+'+index+']:state[offset+'+(23+index)+'])',
    be:index=>'(reverse===0?state[offset+'+(23+index)+']:state[offset+'+index+'])',
    te:index=>'state[offset+'+index+']',
  })).join('\n');
  const kernelSource='function rotateInstanceMatrices(state,count,reverse){for(let i=0;i<count;i++){const offset=i*'+INSTANCE_MATRIX_STRIDE+';'+
    ['a','b','c','d','e','f'].map((name,index)=>'const '+name+'=state[offset+'+(16+index)+'];').join('')+rotate+'\n'+product+'}}';
  const artifact=compileNumericKernel(kernelSource,{parameterTypes:['f64[]','f64','f64'],generalControl:true,
    maxMemoryPages,maxIterations,sourceName:'pinned Matrix4.makeRotationFromEuler / multiplyMatrices instanced loop'});
  const token=fresh(source,'kernel'),create=fresh(source,'create'),register=fresh(source,'class'),owner=fresh(source,'owner');
  const constructor=normalMethod(mesh,'constructor'),superCall=constructor.body.body.find(node=>node.expression?.callee?.type==='Super');
  return {changed:true,header:'import {createInstancedMatrixKernel as '+create+',registerInstancedMatrixClass as '+register+
    ',registerInstancedMatrixOwner as '+owner+'} from '+JSON.stringify(runtimeModule)+';\nvar '+token+'='+create+'('+
    JSON.stringify(Buffer.from(artifact.wasm).toString('base64'))+','+JSON.stringify(orders)+','+maxMemoryPages+');\n',
    edits:[{start:superCall.end,end:superCall.end,text:'\n'+owner+'(this,'+token+');\n'},
      ...[matrix,mesh].map(cls=>({start:cls.end,end:cls.end,text:'\n'+register+'('+token+','+JSON.stringify(cls.id.name)+','+cls.id.name+');\n'}))],
    report:{kind:'pinned-instanced-matrix-kernel',sourceDerived:true,wasmBytes:artifact.wasm.length,
      upstreamCommit:'148ef33ecb6d2502ff796d4554abd1549c95d519',retainedMath:['cos','sin'],accelerationClaim:false}};
}

function shape(loop) {
  const init=loop.init, index=init?.declarations?.[0];
  if(init?.type!=='VariableDeclaration'||init.kind!=='let'||init.declarations.length!==1||!identifier(index.id)||
    index.init?.type!=='Literal'||index.init.value!==0||loop.test?.type!=='BinaryExpression'||loop.test.operator!=='<'||
    !identifier(loop.test.left,index.id.name)||property(loop.test.right)!=='count'||!identifier(loop.test.right.object)||
    loop.update?.type!=='UpdateExpression'||loop.update.operator!=='++'||!identifier(loop.update.argument,index.id.name)||
    loop.body.type!=='BlockStatement'||loop.body.body.length!==4)return null;
  const calls=loop.body.body.map(node=>node.type==='ExpressionStatement'?node.expression:null);
  if(calls.some(call=>call?.type!=='CallExpression'||call.optional||!identifier(call.callee.object)))return null;
  const [rotate,read,multiply,write]=calls, mesh=loop.test.right.object.name, rotation=rotate.callee.object.name;
  const speed=rotate.arguments[0],scratch=multiply.callee.object.name;
  if(property(rotate.callee)!=='makeRotationFromEuler'||rotate.arguments.length!==1||speed?.type!=='MemberExpression'||
    !speed.computed||speed.optional||!identifier(speed.object)||!identifier(speed.property,index.id.name)||
    property(read.callee)!=='getMatrixAt'||!identifier(read.callee.object,mesh)||read.arguments.length!==2||
    !identifier(read.arguments[0],index.id.name)||!identifier(read.arguments[1],scratch)||
    !['multiply','premultiply'].includes(property(multiply.callee))||multiply.arguments.length!==1||!identifier(multiply.arguments[0],rotation)||
    property(write.callee)!=='setMatrixAt'||!identifier(write.callee.object,mesh)||write.arguments.length!==2||
    !identifier(write.arguments[0],index.id.name)||!identifier(write.arguments[1],scratch))return null;
  const captures=[mesh,speed.object.name,rotation,scratch];
  if(new Set(captures).size!==4||captures.includes(index.id.name))return null;
  return {captures,reverse:property(multiply.callee)==='premultiply'};
}

export function specializeInstancedMatrixLoops(source,{runtimeModule,maxLoops=32}={}) {
  const unchanged={changed:false,edits:[],header:'',report:null};
  if(!source.includes('getMatrixAt')||!source.includes('makeRotationFromEuler'))return unchanged;
  let ast;try{ast=acorn.parse(source,{ecmaVersion:'latest',sourceType:'module',locations:true});}catch{return unchanged;}
  let evalCall=false;walk.simple(ast,{CallExpression(node){if(identifier(node.callee,'eval'))evalCall=true;}});
  if(evalCall)return unchanged;
  const bindings=numericLoopScopeBindings(),edits=[],loops=[],attempt=fresh(source,'try_loop'),register=fresh(source,'array');
  walk.fullAncestor(ast,(node,_state,ancestors)=>{
    if(node.type!=='ForStatement'||loops.length>=maxLoops||ancestors.at(-2)?.type==='LabeledStatement')return;
    const candidate=shape(node);if(!candidate)return;
    const scopes=ancestors.slice(0,-1);
    if(candidate.captures.some(name=>name==='arguments'||![...scopes].reverse().some(scope=>bindings(scope,node).has(name))))return;
    const inputs=fresh(source,'inputs_'+loops.length);
    loops.push({sourceSpan:{start:node.start,end:node.end,line:node.loc.start.line,column:node.loc.start.column},reverse:candidate.reverse});
    edits.push({start:node.start,end:node.end,text:'{let '+inputs+';try{'+inputs+'=['+candidate.captures.join(',')+'];}catch{}\n'+
      'if(!'+inputs+'||!'+attempt+'('+[0,1,2,3].map(index=>inputs+'['+index+']').join(',')+','+candidate.reverse+')){'+
      source.slice(node.start,node.end)+'}}'});
  });
  if(!loops.length)return unchanged;
  // Array literals are ordinary native allocations. Witnessing their identity
  // lets the runtime reject Proxy wrappers before inspecting indexed entries;
  // it does not infer ownership from shape or rewrite any array's properties.
  walk.simple(ast,{ArrayExpression(node){edits.push({start:node.start,end:node.start,text:'('+register+'('},
    {start:node.end,end:node.end,text:'))'});}});
  return {changed:true,edits,header:'import {tryInstancedMatrixLoop as '+attempt+',registerInstancedMatrixArray as '+register+
    '} from '+JSON.stringify(runtimeModule)+';\n',report:{kind:'source-instanced-matrix-loops',compiledLoops:loops.length,
      sourceDerived:true,accelerationClaim:false,loops:loops.map(loop=>({...loop,retainedMath:['cos','sin'],
        ownership:'synchronous-copying-transaction',fallback:'original-loop-before-publication'}))}};
}
