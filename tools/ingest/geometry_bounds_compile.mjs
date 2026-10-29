/** Expand only the pinned bounds scan's actual attribute/vector statements.
 * The source still owns makeEmpty, morph selection and public observations.
 * Every component operation remains f64 and in its original evaluation order.
 */
import {normalClasses,normalMethod} from './geometry_normals_compile.mjs';
import {compileNumericKernel} from './numeric_kernel.mjs';

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
