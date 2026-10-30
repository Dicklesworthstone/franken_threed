/** Scalarize the pinned tangent algorithm and its actual vector/attribute
 * methods. Number accumulators stay f64; only source attribute stores round.
 * The two source group passes call private Wasm helpers in their original order.
 */
import {normalClasses,normalMethod} from './geometry_normals_compile.mjs';
import {compileNumericKernel} from './numeric_kernel.mjs';

export function deriveGeometryTangentKernel(geometrySource,vectorSource,vector2Source,attributeSource) {
  const geometry=normalClasses(geometrySource).get('BufferGeometry');
  const classes={Vector3:normalClasses(vectorSource).get('Vector3'),Vector2:normalClasses(vector2Source).get('Vector2'),
    attribute:normalClasses(attributeSource).get('BufferAttribute')};
  const method=normalMethod(geometry,'computeTangents'),body=method.body.body;
  const start=body.findIndex(node=>node.type==='VariableDeclaration' && node.declarations.some(value=>value.id.name==='tan1'));
  const end=body.findIndex(node=>node.type==='ExpressionStatement' && node.expression.left?.property?.name==='_transformed');
  const triangle=body.find(node=>node.type==='FunctionDeclaration' && node.id.name==='handleTriangle');
  const vertex=body.find(node=>node.type==='FunctionDeclaration' && node.id.name==='handleVertex');
  const passes=body.filter(node=>node.type==='ForStatement' && node.init.declarations[0].id.name==='i' && node.init.declarations.length===2);
  if(start<0 || end<=start || !triangle || !vertex || passes.length!==2 || Object.values(classes).some(value=>!value))
    throw new TypeError('Missing pinned tangent algorithm boundaries');
  let serial=0;
  const requirements={Vector3:new Set(),Vector2:new Set(),attribute:new Set()};
  const scalar=value=>typeof value==='string';
  const vector=(type,components)=>({kind:type,components});
  const attr=(array,size)=>({kind:'attribute',array,size});
  const base=new Map([['positionAttribute',attr('positions',3)],['normalAttribute',attr('normals',3)],
    ['uvAttribute',attr('uvs',2)],['tangentAttribute',attr('tangents',4)],['index',attr('indices',1)],
    ['groups',{kind:'groups'}],['tan1',{kind:'pool',array:'tan1'}],['tan2',{kind:'pool',array:'tan2'}]]);
  const number=(node,env,out)=>{const result=value(node,env,out);if(!scalar(result))throw new TypeError('A tangent object cannot escape');return result;};
  function value(node,env,out) {
    if(node.type==='Identifier')return env.get(node.name)??node.name;
    if(node.type==='ThisExpression')return env.get('this');
    if(node.type==='Literal')return node.raw??JSON.stringify(node.value);
    if(node.type==='MemberExpression') {
      const object=value(node.object,env,out),key=node.property.name;
      if(object?.components && !node.computed && key in object.components)return object.components[key];
      if(object?.kind==='attribute' && !node.computed) {
        if(key==='array')return object.array;
        if(key==='itemSize')return String(object.size);
      }
      if(object?.kind==='groups')return node.computed?{kind:'group',index:number(node.property,env,out)}:'groupCount';
      if(object?.kind==='group' && ['start','count'].includes(key))return `groups[${object.index}*2+${key==='start'?0:1}]`;
      if(object?.kind==='pool' && node.computed) {
        const index='__index'+serial++;out.push(`const ${index}=${number(node.property,env,out)};`);
        return vector('Vector3',Object.fromEntries(['x','y','z'].map((axis,i)=>[axis,`${object.array}[${index}*3+${i}]`])));
      }
      if(!scalar(object))throw new TypeError('Unsupported tangent member');
      return node.computed?`${object}[${number(node.property,env,out)}]`:`${object}.${key}`;
    }
    if(node.type==='LogicalExpression' && node.operator==='||') {
      const left='__truth'+serial++;out.push(`const ${left}=${number(node.left,env,out)};`);
      return `(${left}===0 || ${left}!==${left}?${number(node.right,env,out)}:${left})`;
    }
    if(['BinaryExpression','LogicalExpression','AssignmentExpression'].includes(node.type))
      return `(${number(node.left,env,out)} ${node.operator} ${number(node.right,env,out)})`;
    if(node.type==='UnaryExpression')return `(${node.operator}${number(node.argument,env,out)})`;
    if(node.type==='UpdateExpression')return node.prefix?node.operator+number(node.argument,env,out):number(node.argument,env,out)+node.operator;
    if(node.type==='ConditionalExpression')return `(${number(node.test,env,out)}?${number(node.consequent,env,out)}:${number(node.alternate,env,out)})`;
    if(node.type==='CallExpression') {
      if(node.callee.type==='Identifier') {
        const args=node.arguments.map(arg=>number(arg,env,out));
        if(node.callee.name==='isFinite') {
          const arg='__finite'+serial++;out.push(`const ${arg}=${args[0]};`);
          // All admitted arguments are Numbers; the native global is guarded.
          return `(${arg}===${arg} && ${arg}!==(1/0) && ${arg}!==(-1/0))`;
        }
        const captures=node.callee.name==='handleTriangle'?'positions,uvs,tan1,tan2':
          node.callee.name==='handleVertex'?'normals,tangents,tan1,tan2':null;
        if(!captures)throw new TypeError('Unknown tangent helper');
        out.push(`${node.callee.name}(${captures},${args.join(',')});`);return null;
      }
      const receiver=value(node.callee.object,env,out),name=node.callee.property.name;
      if(receiver==='Math')return `Math.${name}(${node.arguments.map(arg=>number(arg,env,out)).join(',')})`;
      if(!receiver || !classes[receiver.kind])throw new TypeError('Unknown tangent receiver');
      const called=normalMethod(classes[receiver.kind],name),scope=new Map(env).set('this',receiver);
      requirements[receiver.kind].add(name);
      called.params.forEach((parameter,i)=>{
        const argument=value(node.arguments[i],env,out);
        if(scalar(argument)) {
          const local='__arg'+serial++;out.push(`let ${local}=${argument};`);scope.set(parameter.name,local);
        }else scope.set(parameter.name,argument);
      });
      for(const statement of called.body.body) {
        // The only attribute branches normalize input/output. The runtime
        // requires normalized === false and the source's exact accessors.
        if(receiver.kind==='attribute' && statement.type==='IfStatement')continue;
        if(statement.type==='ReturnStatement')return value(statement.argument,scope,out);
        emit(statement,scope,out);
      }
      return null;
    }
    throw new TypeError(`Unsupported tangent expression ${node.type}`);
  }
  function emit(node,env,out) {
    if(node.type==='BlockStatement') {
      const inner=[],scope=new Map(env);for(const statement of node.body)emit(statement,scope,inner);
      out.push('{'+inner.join('')+'}');return;
    }
    if(node.type==='VariableDeclaration') {
      for(const declaration of node.declarations) {
        const result=value(declaration.init,env,out);
        if(scalar(result)) {
          const local='__local'+serial++;out.push(`${node.kind} ${local}=${result};`);env.set(declaration.id.name,local);
        }else env.set(declaration.id.name,result);
      }
      return;
    }
    if(node.type==='ExpressionStatement') {
      const result=value(node.expression,env,out);if(scalar(result))out.push(result+';');return;
    }
    if(node.type==='IfStatement') {
      const test=number(node.test,env,out),yes=[],no=[];
      emit(node.consequent,new Map(env),yes);if(node.alternate)emit(node.alternate,new Map(env),no);
      out.push(`if(${test}){${yes.join('')}}`+(node.alternate?`else{${no.join('')}}`:''));return;
    }
    if(node.type==='ReturnStatement' && !node.argument){out.push('return;');return;}
    if(node.type==='ForStatement') {
      const scope=new Map(env),init=[];emit(node.init,scope,init);
      const prefix=[],test=number(node.test,scope,prefix),update=number(node.update,scope,prefix),loop=[];
      if(prefix.length)throw new TypeError('Tangent loop controls must be pure');
      emit(node.body,scope,loop);
      out.push('{'+init.join('')+`for(;${test};${update})`+loop.join('')+'}');return;
    }
    throw new TypeError(`Unsupported tangent statement ${node.type}`);
  }
  function helper(node,captures) {
    const env=new Map(base),output=[];
    // Every source scratch vector is overwritten before use in each helper.
    // Constructors only initialize numeric own components and are guarded.
    for(const statement of body)if(statement.type==='VariableDeclaration')for(const declaration of statement.declarations) {
      if(declaration.init?.type!=='NewExpression' || !['Vector2','Vector3'].includes(declaration.init.callee.name))continue;
      const kind=declaration.init.callee.name,axes=kind==='Vector2'?['x','y']:['x','y','z'];
      const components=Object.fromEntries(axes.map(axis=>[axis,`__vector${serial++}_${axis}`]));
      env.set(declaration.id.name,vector(kind,components));
      output.push('let '+Object.values(components).map(name=>name+'=0').join(',')+';');
    }
    for(const statement of node.body.body)emit(statement,env,output);
    return `function ${node.id.name}(${captures},${node.params.map(param=>param.name).join(',')}){${output.join('')}}`;
  }
  const helperSources=new Map([
    ['handleTriangle',helper(triangle,'positions,uvs,tan1,tan2')],
    ['handleVertex',helper(vertex,'normals,tangents,tan1,tan2')],
  ]);
  const output=[];for(const pass of passes)emit(pass,new Map(base),output);
  const source='function computeTangents(positions,normals,uvs,indices,tangents,groups,tan1,tan2,groupCount){'+output.join('')+'}';
  return {source,helperSources,span:{start:body[start].start,end:body[end-1].end},
    requirements:Object.fromEntries(Object.entries(requirements).map(([kind,names])=>[kind,[...names]]))};
}

export function compileGeometryTangentKernels(derived,{maxMemoryPages=2048,maxIterations=100000000}={}) {
  const variants=[];
  // Widening source f32 reads to f64 is exact. Independent input/output layouts
  // remain typed so no source store boundary or NaN representation is guessed.
  for(const position of ['f32[]','f64[]'])for(const normal of ['f32[]','f64[]'])for(const uv of ['f32[]','f64[]'])
    for(const tangent of ['f32[]','f64[]'])for(const index of ['u16[]','u32[]']) {
      const types=[position,normal,uv,index,tangent];
      const artifact=compileNumericKernel(derived.source,{parameterTypes:[...types,'f64[]','f64[]','f64[]','f64'],
        helperSources:derived.helperSources,generalControl:true,allowMath:true,maxMemoryPages,maxIterations,
        sourceName:'pinned BufferGeometry.computeTangents'});
      variants.push({types,base64:Buffer.from(artifact.wasm).toString('base64')});
    }
  return variants;
}
