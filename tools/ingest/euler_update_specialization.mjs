/** Source-shaped object update islands. Pinned constructors establish allocation
 * and callback witnesses; application syntax supplies the actual component
 * operations. Trigonometry remains the guarded source Math implementation. */
import {createHash} from 'node:crypto';
import * as acorn from 'acorn';
import * as walk from 'acorn-walk';
import {normalClasses,normalMethod} from './geometry_normals_compile.mjs';
import {compileNumericKernel} from './numeric_kernel.mjs';
import {numericLoopScopeBindings} from './numeric_loop_islands.mjs';

const CORE_HASH='9edde002b066a9a05676a6127f67735b62baf399bdea529f2f7e31657da769e6';
const XYZ=['x','y','z'],OPERATORS=['+','-','*','/'];
export const EULER_UPDATE_STRIDE=17;
const fresh=(source,name)=>{let value='__f3d_euler_'+name;while(source.includes(value))value+='_';return value;};
const property=node=>node?.type!=='MemberExpression' || node.optional?null:
  !node.computed && node.property.type==='Identifier'?node.property.name:
    node.computed && node.property.type==='Literal' && typeof node.property.value==='string'?node.property.value:null;
const identifier=(node,name)=>node?.type==='Identifier' && (name===undefined || node.name===name);

export function specializeEulerUpdateBase(source,{runtimeModule,maxMemoryPages=2048,maxIterations=100000000}={}) {
  if(createHash('sha256').update(source).digest('hex')!==CORE_HASH)return {changed:false,edits:[],header:''};
  const classes=normalClasses(source),object=classes.get('Object3D'),quaternion=classes.get('Quaternion');
  const method=normalMethod(quaternion,'setFromEuler');
  const branches=method.body.body.find(node=>node.type==='SwitchStatement').cases.filter(node=>node.test);
  const orders=branches.map(node=>node.test.value),base='offset';
  const component=axis=>{
    const p=axis*3,left='state['+base+'+plan['+p+']]',right='state['+base+'+3+plan['+(p+1)+']]';
    return '(plan['+(p+2)+']===0?'+left+'+'+right+':plan['+(p+2)+']===1?'+left+'-'+right+
      ':plan['+(p+2)+']===2?'+left+'*'+right+':'+left+'/'+right+')';
  };
  function arithmetic(node) {
    const edits=[];
    walk.simple(node,{MemberExpression(value){
      if(value.object.type==='ThisExpression' && ['_x','_y','_z','_w'].includes(property(value)))
        edits.push({start:value.start,end:value.end,text:'state['+base+'+'+(13+['_x','_y','_z','_w'].indexOf(property(value)))+']'});
    }});
    let text=source.slice(node.start,node.end);
    for(const edit of edits.sort((a,b)=>b.start-a.start))text=text.slice(0,edit.start-node.start)+edit.text+text.slice(edit.end-node.start);
    return text;
  }
  // Each argument is evaluated before any Euler field is written. The six
  // trigonometric inputs were computed with those same Number arguments during
  // the guarded packing pass. Only the pinned quaternion arithmetic is inlined.
  const kernelSource='function updateEulerChildren(state,plan,count){for(let i=0;i<count;i++){const offset=i*'+EULER_UPDATE_STRIDE+';'+
    XYZ.map((axis,index)=>'const '+axis+'='+component(index)+';').join('')+
    XYZ.map((axis,index)=>'state[offset+'+index+']='+axis+';').join('')+
    ['c1','c2','c3','s1','s2','s3'].map((name,index)=>'const '+name+'=state[offset+'+(index+6)+'];').join('')+
    branches.map((branch,index)=>(index?'else ':'')+'if(state[offset+12]==='+index+'){'+
      branch.consequent.filter(node=>node.type!=='BreakStatement').map(arithmetic).join('\n')+'}').join('')+'}}';
  const artifact=compileNumericKernel(kernelSource,{parameterTypes:['f64[]','f64[]','f64'],generalControl:true,
    maxMemoryPages,maxIterations,sourceName:'pinned Euler.set / Object3D.onRotationChange / Quaternion.setFromEuler'});
  const token=fresh(source,'kernel'),create=fresh(source,'create_kernel'),register=fresh(source,'class');
  const numeric=fresh(source,'numeric'),owner=fresh(source,'owner'),data=fresh(source,'data'),edits=[];
  for(const name of ['Euler','Quaternion']) {
    const cls=classes.get(name),ctor=normalMethod(cls,'constructor');
    edits.push({start:ctor.body.start+1,end:ctor.body.start+1,text:'\n'+numeric+'(this,'+token+','+JSON.stringify(name)+');\n'});
    edits.push({start:cls.end,end:cls.end,text:'\n'+register+'('+token+','+JSON.stringify(name)+','+name+');\n'});
  }
  const ctor=normalMethod(object,'constructor');
  const userData=ctor.body.body.find(node=>node.expression?.left?.property?.name==='userData').expression.right;
  edits.push({start:userData.start,end:userData.end,text:data+'('+source.slice(userData.start,userData.end)+')'});
  edits.push({start:ctor.body.end-1,end:ctor.body.end-1,text:'\n'+owner+'(this,rotation,quaternion,onRotationChange,'+token+');\n'});
  return {changed:true,edits,header:'import {createEulerUpdateKernel as '+create+',registerEulerUpdateClass as '+register+
    ',registerEulerUpdateNumeric as '+numeric+',registerEulerUpdateOwner as '+owner+',registerEulerUpdateData as '+data+
    '} from '+JSON.stringify(runtimeModule)+';\nvar '+token+'='+create+'('+JSON.stringify(Buffer.from(artifact.wasm).toString('base64'))+
    ','+JSON.stringify(orders)+','+maxMemoryPages+');\n',
    report:{kind:'pinned-euler-kernel',sourceDerived:true,wasmBytes:artifact.wasm.length,
      upstreamCommit:'148ef33ecb6d2502ff796d4554abd1549c95d519',retainedMath:['cos','sin'],accelerationClaim:false}};
}

function shape(loop) {
  const declaration=loop.init;
  if(declaration?.type!=='VariableDeclaration' || declaration.kind!=='let' || declaration.declarations.length!==1)return null;
  const index=declaration.declarations[0];
  if(!identifier(index.id) || index.init?.type!=='Literal' || index.init.value!==0 ||
    loop.test?.type!=='BinaryExpression' || loop.test.operator!=='<' || !identifier(loop.test.left,index.id.name) ||
    loop.update?.type!=='UpdateExpression' || loop.update.operator!=='++' || !identifier(loop.update.argument,index.id.name) ||
    loop.body.type!=='BlockStatement' || loop.body.body.length!==3)return null;
  const [objectStatement,speedStatement,callStatement]=loop.body.body;
  const local=statement=>statement.type==='VariableDeclaration' && ['const','let'].includes(statement.kind) &&
    statement.declarations.length===1 && identifier(statement.declarations[0].id)?statement.declarations[0]:null;
  const child=local(objectStatement),speed=local(speedStatement),element=child?.init;
  if(!child || !speed || element?.type!=='MemberExpression' || !element.computed || element.optional ||
    !identifier(element.property,index.id.name) || property(element.object)!=='children' || !identifier(element.object.object))return null;
  const group=element.object.object.name,path=[];
  if(child.id.name===index.id.name || speed.id.name===index.id.name ||
    [index.id.name,child.id.name,speed.id.name].includes(group))return null;
  let current=speed.init;
  while(property(current)!==null){path.unshift(property(current));current=current.object;}
  if(!identifier(current,child.id.name) || path.length<1 || path.length>8)return null;
  const call=callStatement.type==='ExpressionStatement'?callStatement.expression:null;
  if(call?.type!=='CallExpression' || call.optional || call.arguments.length!==3 || property(call.callee)!=='set' ||
    property(call.callee.object)!=='rotation' || !identifier(call.callee.object.object,child.id.name))return null;
  const operations=[];
  for(const argument of call.arguments){
    if(argument.type!=='BinaryExpression' || !OPERATORS.includes(argument.operator))return null;
    const left=argument.left,right=argument.right;
    if(!XYZ.includes(property(left)) || property(left.object)!=='rotation' || !identifier(left.object.object,child.id.name) ||
      !XYZ.includes(property(right)) || !identifier(right.object,speed.id.name))return null;
    operations.push(XYZ.indexOf(property(left)),XYZ.indexOf(property(right)),OPERATORS.indexOf(argument.operator));
  }
  const bound=loop.test.right;
  if(identifier(bound))return bound.name===index.id.name?null:{group,bound:bound.name,boundField:null,path,operations,captures:[group,bound.name]};
  if(bound.type==='Literal' && typeof bound.value==='number')return {group,bound:bound.raw,boundField:null,path,operations,captures:[group]};
  if(property(bound)==='length' && property(bound.object)==='children' && identifier(bound.object.object,group))
    return {group,bound:'null',boundField:null,childrenBound:true,path,operations,captures:[group]};
  if(property(bound)!==null && identifier(bound.object) && bound.object.name!==index.id.name)
    return {group,bound:bound.object.name,boundField:property(bound),path,operations,captures:[group,bound.object.name]};
  return null;
}

/** Application spelling is irrelevant. Admission requires the exact ordinary
 * loop/call shape, declarative captures, then runtime allocation/alias/method
 * proofs. Unknown source, getters, callbacks and coercions keep the original. */
export function specializeEulerUpdateLoops(source,{runtimeModule,maxLoops=32}={}) {
  const unchanged={changed:false,edits:[],header:'',report:null};
  if(!source.includes('rotation') || !source.includes('children'))return unchanged;
  let ast;
  try{ast=acorn.parse(source,{ecmaVersion:'latest',sourceType:'module',locations:true});}catch{return unchanged;}
  let evalCall=false;
  walk.simple(ast,{CallExpression(node){if(identifier(node.callee,'eval'))evalCall=true;}});
  if(evalCall)return unchanged;
  const bindings=numericLoopScopeBindings(),edits=[],plans=[],literalRegistrations=new Set();
  const create=fresh(source,'create_plan'),attempt=fresh(source,'try_loop'),register=fresh(source,'literal');
  walk.fullAncestor(ast,(node,_state,ancestors)=>{
    if(node.type!=='ForStatement' || plans.length>=maxLoops || ancestors.at(-2)?.type==='LabeledStatement')return;
    const candidate=shape(node);if(!candidate)return;
    const scopes=ancestors.slice(0,-1),owner=name=>[...scopes].reverse().find(scope=>bindings(scope,node).has(name));
    if(candidate.captures.some(name=>name==='arguments' || !owner(name)))return;
    const serial=plans.length,token=fresh(source,'plan_'+serial),args=fresh(source,'inputs_'+serial);
    // The source API/control object remains ordinary and keeps its identity.
    // Only its literal allocation is witnessed; a proxy or rebound unregistered
    // object can never acquire the witness by shape or prototype imitation.
    if(candidate.boundField!==null){
      const scope=owner(candidate.bound);
      for(const statement of scope?.body instanceof Array?scope.body:[]){
        const value=statement.declaration??statement;
        if(value.type!=='VariableDeclaration')continue;
        for(const d of value.declarations)if(identifier(d.id,candidate.bound) && d.init?.type==='ObjectExpression' && !literalRegistrations.has(d.init)){
          literalRegistrations.add(d.init);edits.push({start:d.init.start,end:d.init.start,text:register+'('},{start:d.init.end,end:d.init.end,text:')'});
        }
      }
    }
    const metadata={boundField:candidate.boundField,childrenBound:candidate.childrenBound===true,path:candidate.path,operations:candidate.operations};
    plans.push({token,metadata,sourceSpan:{start:node.start,end:node.end,line:node.loc.start.line,column:node.loc.start.column}});
    // Capture failures (e.g. a zero-trip loop with a TDZ body binding) merely
    // retain the original. Dispatch/publication is outside this catch so a
    // post-publication exception can never replay source effects.
    edits.push({start:node.start,end:node.end,text:'{let '+args+';try{'+args+'=['+candidate.group+','+candidate.bound+'];}catch{}\n'+
      'if(!'+args+'||!'+attempt+'('+args+'[0],'+args+'[1],'+token+')){'+source.slice(node.start,node.end)+'}}'});
  });
  if(!plans.length)return unchanged;
  return {changed:true,edits,header:'import {createEulerUpdatePlan as '+create+',tryEulerUpdateLoop as '+attempt+
    ',registerEulerUpdateData as '+register+'} from '+JSON.stringify(runtimeModule)+';\n'+
    plans.map(plan=>'var '+plan.token+'='+create+'('+JSON.stringify(plan.metadata)+');').join('\n')+'\n',
    report:{kind:'source-euler-update-loops',compiledLoops:plans.length,sourceDerived:true,accelerationClaim:false,
      loops:plans.map(({sourceSpan,metadata})=>({sourceSpan,...metadata,retainedMath:['cos','sin'],
        ownership:'synchronous-copying-transaction',fallback:'original-loop-before-publication'}))}};
}
