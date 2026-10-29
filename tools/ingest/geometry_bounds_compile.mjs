/** Expand only the pinned bounds scan's actual attribute/vector statements.
 * The source still owns makeEmpty, morph selection and public observations.
 * Every component operation remains f64 and in its original evaluation order.
 */
import {normalClasses,normalMethod} from './geometry_normals_compile.mjs';
import {compileNumericKernel} from './numeric_kernel.mjs';
import * as walk from 'acorn-walk';

export function deriveGeometryBoxKernel(boxSource,vectorSource,attributeSource) {
  const box=normalClasses(boxSource).get('Box3'),vector=normalClasses(vectorSource).get('Vector3');
  const attribute=normalClasses(attributeSource).get('BufferAttribute');
  if(!box || !vector || !attribute)throw new TypeError('Missing verified bounds classes');
  const method=normalMethod(box,'setFromBufferAttribute'),loop=method.body.body.find(node=>node.type==='ForStatement');
  const expansion=loop.body.body[0].expression,load=expansion.arguments[0],scratchName=load.callee.object.name;
  const vectorValue=offset=>({kind:'vector',offset}),attr={kind:'attribute'},owner={kind:'box'};
  const requirements={box:new Set(),vector:new Set(),attribute:new Set()};
  let serial=0;
  function value(node,env) {
    if(node.type==='ThisExpression')return env.get('this');
    if(node.type==='Identifier')return env.get(node.name)??node.name;
    if(node.type==='MemberExpression' && !node.computed && value(node.object,env)===owner) {
      if(node.property.name==='min')return vectorValue(0);
      if(node.property.name==='max')return vectorValue(3);
    }
    return expression(node,env);
  }
  function expression(node,env) {
    if(node.type==='Identifier' || node.type==='ThisExpression') {
      const resolved=value(node,env);if(typeof resolved!=='string')throw new TypeError('Bounds object cannot escape');return resolved;
    }
    if(node.type==='Literal')return node.raw??JSON.stringify(node.value);
    if(node.type==='MemberExpression') {
      const object=value(node.object,env),key=node.property.name;
      if(object?.kind==='vector' && !node.computed && ['x','y','z'].includes(key))
        return 'state['+(object.offset+['x','y','z'].indexOf(key))+']';
      if(object===attr && !node.computed) {
        if(key==='array')return 'values';if(key==='itemSize')return 'itemSize';if(key==='count')return 'count';
      }
      if(typeof object!=='string')throw new TypeError('Unsupported bounds member');
      return node.computed?object+'['+expression(node.property,env)+']':object+'.'+key;
    }
    if(['BinaryExpression','AssignmentExpression'].includes(node.type))
      return '('+expression(node.left,env)+' '+node.operator+' '+expression(node.right,env)+')';
    if(node.type==='UpdateExpression')return node.prefix?node.operator+expression(node.argument,env):expression(node.argument,env)+node.operator;
    if(node.type==='CallExpression') {
      const receiver=value(node.callee.object,env),name=node.callee.property.name;
      if(receiver==='Math')return 'Math.'+name+'('+node.arguments.map(arg=>expression(arg,env)).join(',')+')';
      if(receiver===attr && ['getX','getY','getZ'].includes(name)) {
        requirements.attribute.add(name);const get=normalMethod(attribute,name);
        // The normalized branch remains on the source path under the guard.
        return expression(get.body.body[0].declarations[0].init,new Map(env).set('this',attr)
          .set(get.params[0].name,expression(node.arguments[0],env)));
      }
    }
    throw new TypeError('Unsupported verified bounds expression: '+node.type);
  }
  function call(node,env) {
    const receiver=value(node.callee.object,env),name=node.callee.property.name;
    const cls=receiver===owner?box:receiver?.kind==='vector'?vector:null;
    if(!cls)throw new TypeError('Unsupported bounds call');
    requirements[receiver.kind].add(name);
    const method=normalMethod(cls,name),scope=new Map(env).set('this',receiver),prefix=[];
    method.params.forEach((param,index)=>{
      const argument=value(node.arguments[index],env);
      if(typeof argument==='string') {
        const local='__bounds_arg'+serial++;prefix.push('const '+local+'='+argument+';');scope.set(param.name,local);
      }else scope.set(param.name,argument);
    });
    return '{'+prefix.join('')+method.body.body.filter(statement=>statement.type!=='ReturnStatement').map(statement=>{
      if(statement.type!=='ExpressionStatement')throw new TypeError('Unsupported bounds statement');
      return statement.expression.type==='CallExpression'?call(statement.expression,scope):expression(statement.expression,scope)+';';
    }).join('')+'}';
  }
  const env=new Map([['this',owner],[method.params[0].name,attr],[scratchName,vectorValue(6)]]);
  const init=loop.init.declarations.map(node=>node.id.name+'='+expression(node.init,env)).join(',');
  // fromBufferAttribute returns its scratch vector; expansion receives that
  // exact identity after all three loads. No per-vertex objects are allocated.
  const argument={type:'Identifier',name:scratchName};
  const body=call(load,env)+call({...expansion,arguments:[argument]},env);
  return {source:'function scanBounds(values,state,count,itemSize){for('+loop.init.kind+' '+init+';'+
    expression(loop.test,env)+';'+expression(loop.update,env)+'){'+body+'}}',
    span:{start:loop.start,end:loop.end},scratchName,
    requirements:Object.fromEntries(Object.entries(requirements).map(([name,methods])=>[name,[...methods]]))};
}

export function compileGeometryBoxKernels(derived,{maxMemoryPages=2048,maxIterations=100000000}={}) {
  return ['f32[]','f64[]'].map(type=>{
    const artifact=compileNumericKernel(derived.source,{parameterTypes:[type,'f64[]','f64','f64'],
      generalControl:true,allowMath:true,maxMemoryPages,maxIterations,sourceName:'pinned Box3.setFromBufferAttribute'});
    return {type,base64:Buffer.from(artifact.wasm).toString('base64')};
  });
}

/** Keep the radius loop and squared-distance formula as source text. Only
 * verified object/attribute accesses become packed scalar cells; no alternative
 * center/radius algorithm, reduction order or fused arithmetic is introduced.
 */
export function deriveGeometryRadiusKernels(geometrySource,vectorSource,attributeSource) {
  const geometry=normalClasses(geometrySource).get('BufferGeometry'),vector=normalClasses(vectorSource).get('Vector3');
  const attribute=normalClasses(attributeSource).get('BufferAttribute');
  if(!geometry || !vector || !attribute)throw new TypeError('Missing verified radius classes');
  const body=normalMethod(geometry,'computeBoundingSphere').body.body.at(-1).consequent.body;
  const ordinary=body.find(node=>node.type==='ForStatement');
  const morph=body[body.indexOf(ordinary)+1].consequent.body[0].body.body.find(node=>node.type==='ForStatement');
  const scratchName=ordinary.body.body[0].expression.callee.object.name;
  const distanceCall=ordinary.body.body[1].expression.right.arguments[1],centerName=distanceCall.callee.object.name;
  const reductionName=ordinary.body.body[1].expression.left.name;
  const positionName=ordinary.body.body[0].expression.arguments[0].name;
  const morphName=morph.body.body[0].expression.arguments[0].name;
  const conditional=morph.body.body[1],relativeName=conditional.test.name;
  const offsetName=conditional.consequent.body[0].expression.callee.object.name;
  const path=node=>node.type==='ThisExpression'?'this':node.type==='Identifier'?node.name:
    node.type==='MemberExpression' && !node.computed?path(node.object)+'.'+node.property.name:null;
  function rewrite(source,node,bindings,calls=()=>null) {
    const edits=[];
    walk.fullAncestor(node,(current,_state,parents)=>{
      const parent=parents.at(-2);
      if(current.type==='Identifier' && parent?.type==='MemberExpression' && parent.property===current && !parent.computed)return;
      const replacement=current.type==='CallExpression'?calls(current):bindings.get(path(current));
      if(replacement!==null && replacement!==undefined)edits.push({start:current.start,end:current.end,text:replacement});
    });
    // A replaced call/member owns its nested tokens. Keep only disjoint edits.
    const selected=[];let end=-1;
    for(const edit of edits.sort((a,b)=>a.start-b.start || b.end-a.end))if(edit.start>=end){selected.push(edit);end=edit.end;}
    let result=source.slice(node.start,node.end);
    for(const edit of selected.reverse())result=result.slice(0,edit.start-node.start)+edit.text+result.slice(edit.end-node.start);
    return result;
  }
  const cells=(name,offset)=>['x','y','z'].map((axis,index)=>[name+'.'+axis,'state['+(offset+index)+']']);
  const distance=normalMethod(vector,'distanceToSquared');
  const helper='function __boundsDistance(state)'+rewrite(vectorSource,distance.body,
    new Map([...cells('this',3),...cells(distance.params[0].name,0)]));
  const attributes=new Map([[positionName,{array:'baseValues',size:'baseItemSize'}],[morphName,{array:'values',size:'itemSize'}]]);
  function load(call,ordinaryScan) {
    const name=call.arguments[0].name,layout=ordinaryScan?{array:'values',size:'itemSize'}:attributes.get(name);
    const target=call.callee.object.name===scratchName?0:6,method=normalMethod(vector,'fromBufferAttribute');
    const index=geometrySource.slice(call.arguments[1].start,call.arguments[1].end);
    return method.body.body.filter(node=>node.type!=='ReturnStatement').map(node=>rewrite(vectorSource,node,new Map(cells('this',target)),getter=>{
      const source=normalMethod(attribute,getter.callee.property.name);
      return rewrite(attributeSource,source.body.body[0].declarations[0].init,
        new Map([['this.array',layout.array],['this.itemSize',layout.size],[source.params[0].name,'('+index+')']]));
    })).join('\n');
  }
  function emit(node,ordinaryScan) {
    if(node.type==='BlockStatement')return '{'+node.body.map(child=>emit(child,ordinaryScan)).join('\n')+'}';
    if(node.type==='IfStatement')return 'if(relative)'+emit(node.consequent,ordinaryScan);
    const call=node.expression;
    if(call.type==='CallExpression') {
      if(call.callee.property.name==='fromBufferAttribute')return load(call,ordinaryScan);
      if(call.callee.property.name==='add') {
        const method=normalMethod(vector,'add');
        return method.body.body.filter(node=>node.type!=='ReturnStatement').map(node=>rewrite(vectorSource,node,
          new Map([...cells('this',0),...cells(method.params[0].name,6)]))).join('\n');
      }
      throw new TypeError('Unsupported verified radius call');
    }
    return rewrite(geometrySource,node,new Map(),call=>path(call.callee)===centerName+'.distanceToSquared'?'__boundsDistance(state)':null);
  }
  const scans=[['radius',ordinary],['morph-radius',morph]].map(([method,loop])=>({method,
    span:{start:loop.start,end:loop.end},attribute:method==='radius'?positionName:morphName,
    source:'function scanRadius(values,baseValues,state,count,itemSize,baseItemSize,'+reductionName+',relative){'+
      'for('+rewrite(geometrySource,loop.init,new Map([[positionName+'.count','count'],[morphName+'.count','count']]))+';'+
      geometrySource.slice(loop.test.start,loop.test.end)+';'+geometrySource.slice(loop.update.start,loop.update.end)+')'+
      emit(loop.body,method==='radius')+'return '+reductionName+';}',
  }));
  return {scans,helper,scratchName,centerName,reductionName,positionName,offsetName,relativeName};
}

export function compileGeometryRadiusKernels(derived,{maxMemoryPages=2048,maxIterations=100000000}={}) {
  const variants=[];
  for(const scan of derived.scans)for(const type of ['f32[]','f64[]'])for(const baseType of scan.method==='radius'?['f64[]']:['f32[]','f64[]']) {
    const artifact=compileNumericKernel(scan.source,{parameterTypes:[type,baseType,'f64[]','f64','f64','f64','f64','f64'],
      helperSources:new Map([['__boundsDistance',derived.helper]]),generalControl:true,allowMath:true,maxMemoryPages,maxIterations,
      sourceName:'pinned BufferGeometry.computeBoundingSphere '+scan.method});
    variants.push({method:scan.method,type,baseType,base64:Buffer.from(artifact.wasm).toString('base64')});
  }
  return variants;
}
