/** Inline the pinned normal loops' actual Vector3/BufferAttribute method bodies.
 * The result is ordinary numeric source fed to the existing checked Wasm
 * compiler. No alternative normal algorithm, reassociation or f32 fusion is
 * introduced. Guards establish native accessors and their integer record
 * layouts. Normalized position decoding comes from the pinned MathUtils
 * switch, before the original f64 vector arithmetic and typed-store boundaries.
 */
import * as acorn from 'acorn';
import {compileNumericKernel} from './numeric_kernel.mjs';

export function normalClasses(source) {
  const ast=acorn.parse(source,{ecmaVersion:'latest',sourceType:'module'});
  return new Map(ast.body.filter(node=>node.type==='ClassDeclaration').map(node=>[node.id.name,node]));
}
export const normalMethod=(cls,name)=>cls.body.body.find(node=>node.key?.name===name)?.value;

/** Preserve the source's constructor-specific division and signed clamping.
 * Consumers prove the constructor dispatch before invoking these scalar helpers.
 * Float64Array is deliberately absent: normalized Float64 input throws upstream.
 */
export function deriveAttributeDenormalizers(mathUtilsSource) {
  return deriveAttributeConversions(mathUtilsSource,'denormalize');
}

/** The matching source conversion for normalized integer attribute stores. */
export function deriveAttributeNormalizers(mathUtilsSource) {
  return deriveAttributeConversions(mathUtilsSource,'normalize');
}

function deriveAttributeConversions(mathUtilsSource,name) {
  const ast=acorn.parse(mathUtilsSource,{ecmaVersion:'latest',sourceType:'module'});
  const fn=ast.body.find(node=>node.type==='FunctionDeclaration' && node.id.name===name);
  const dispatch=fn?.body.body[0],types={Float32Array:'f32[]',Uint32Array:'u32[]',Uint16Array:'u16[]',
    Uint8Array:'u8[]',Uint8ClampedArray:'u8c[]',Int32Array:'i32[]',Int16Array:'i16[]',Int8Array:'i8[]'};
  if(dispatch?.type!=='SwitchStatement' || fn.params.length!==2)throw new TypeError('Missing verified '+name+' switch');
  const result=new Map(),pending=[];
  for(const entry of dispatch.cases) {
    if(!entry.test)continue;
    const constructor=entry.test.name,type=types[constructor];
    if(!type)throw new TypeError('Unknown verified '+name+' constructor');
    pending.push({constructor,type});
    if(!entry.consequent.length)continue;
    if(entry.consequent.length!==1 || entry.consequent[0].type!=='ReturnStatement')
      throw new TypeError('Unsupported verified '+name+' branch');
    const value=entry.consequent[0].argument,parameter=fn.params[0].name;
    const expression=mathUtilsSource.slice(value.start,value.end);
    for(const item of pending)result.set(item.type,{constructor:item.constructor,parameter,expression,
      source:`function __f3d_${name}(${parameter}){return ${expression};}`});
    pending.length=0;
  }
  if(pending.length || result.size!==Object.keys(types).length)throw new TypeError('Incomplete verified '+name+' switch');
  return result;
}

export function deriveGeometryNormalKernels(geometrySource,vectorSource,attributeSource,mathUtilsSource) {
  const geometry=normalClasses(geometrySource).get('BufferGeometry');
  const vector=normalClasses(vectorSource).get('Vector3');
  const attribute=normalClasses(attributeSource).get('BufferAttribute');
  if (!geometry || !vector || !attribute) throw new TypeError('Missing verified normal-kernel classes');
  const compute=normalMethod(geometry,'computeVertexNormals'),normalize=normalMethod(geometry,'normalizeNormals');
  const computeBody=compute.body.body.find(node=>node.type==='IfStatement').consequent.body;
  const first=computeBody.findIndex(node=>node.type==='VariableDeclaration' &&
    node.declarations.some(value=>value.init?.type==='NewExpression' && value.init.callee.name==='Vector3'));
  const end=computeBody.findIndex(node=>node.type==='ExpressionStatement' &&
    node.expression.callee?.property?.name==='normalizeNormals');
  const normalizationLoop=normalize.body.body.find(node=>node.type==='ForStatement');
  const scratchName=normalizationLoop.body.body[0].expression.callee.object.name;
  if (first<0 || end<=first || !scratchName) throw new TypeError('Missing verified normal-loop boundaries');
  let serial=0;
  const requirements={vector:new Set(),attribute:new Set()};
  const vectorValue=name=>({kind:'vector',name});
  const attr=(name,array,count,size,stride=String(size),offset='0')=>({kind:'attribute',name,array,count,stride,offset});
  const numeric=value=>typeof value==='string';
  function value(node,env) {
    if (node.type==='ThisExpression') return env.get('this');
    if (node.type==='Identifier') return env.get(node.name)??node.name;
    return expression(node,env);
  }
  function expression(node,env) {
    if (!node) return '';
    if (node.type==='Identifier' || node.type==='ThisExpression') {
      const result=value(node,env);if(!numeric(result))throw new TypeError('A normal-kernel object cannot escape');return result;
    }
    if (node.type==='Literal') return node.raw??JSON.stringify(node.value);
    if (node.type==='MemberExpression') {
      if(node.computed && node.object.type==='MemberExpression' && !node.object.computed && node.object.property.name==='array') {
        const owner=value(node.object.object,env);
        if(owner?.kind==='attribute') {
          const index=expression(node.property,env);
          return `${owner.array}[${owner.offset==='0'?index:`(${index} + ${owner.offset})`}]`;
        }
      }
      const object=value(node.object,env),property=node.property.name;
      if (object?.kind==='vector' && !node.computed && ['x','y','z'].includes(property)) return object.name+'_'+property;
      if (object?.kind==='attribute' && !node.computed) {
        if(property==='array')return object.array;
        if(property==='itemSize')return object.stride;
        if(property==='count')return object.count;
      }
      if (!numeric(object)) throw new TypeError('Unsupported verified member access');
      return node.computed?`${object}[${expression(node.property,env)}]`:`${object}.${property}`;
    }
    if (['BinaryExpression','LogicalExpression','AssignmentExpression'].includes(node.type))
      return `(${expression(node.left,env)} ${node.operator} ${expression(node.right,env)})`;
    if (node.type==='UpdateExpression')return node.prefix?node.operator+expression(node.argument,env):expression(node.argument,env)+node.operator;
    if (node.type==='UnaryExpression')return `(${node.operator}${expression(node.argument,env)})`;
    if (node.type==='ConditionalExpression')return `(${expression(node.test,env)}?${expression(node.consequent,env)}:${expression(node.alternate,env)})`;
    if (node.type==='CallExpression' && node.callee.type==='MemberExpression' && !node.callee.computed) {
      const receiver=value(node.callee.object,env),name=node.callee.property.name;
      if (receiver==='Math')return `Math.${name}(${node.arguments.map(arg=>expression(arg,env)).join(',')})`;
      if (receiver?.kind==='attribute' && ['getX','getY','getZ'].includes(name)) {
        requirements.attribute.add(name);
        const method=normalMethod(attribute,name),scope=new Map(env).set('this',receiver)
          .set(method.params[0].name,expression(node.arguments[0],env));
        const load=expression(method.body.body[0].declarations[0].init,scope);
        // Output normals and indices retain unnormalized access. Only position
        // loads gain the source's optional constructor-dependent conversion.
        return mathUtilsSource && receiver.name==='positionAttribute'
          ?`(positionNormalized?__f3d_denormalize(${load}):${load})`:load;
      }
      if (receiver?.kind==='vector' && name==='length') {
        requirements.vector.add(name);
        return expression(normalMethod(vector,name).body.body[0].argument,new Map(env).set('this',receiver));
      }
    }
    throw new TypeError(`Unsupported verified normal expression: ${node.type}`);
  }
  function call(node,env) {
    const receiver=value(node.callee.object,env),name=node.callee.property.name;
    if (!receiver || !['vector','attribute'].includes(receiver.kind)) throw new TypeError('Unknown normal method receiver');
    const method=normalMethod(receiver.kind==='vector'?vector:attribute,name);
    if (!method) throw new TypeError('Unknown verified normal method');
    requirements[receiver.kind].add(name);
    const scope=new Map(env).set('this',receiver),prefix=[];
    method.params.forEach((param,index)=>{
      const input=node.arguments[index];
      let argument;
      if (input.type==='LogicalExpression' && input.operator==='||') {
        // Vector3.normalize's Number-valued length() || 1: evaluate length
        // once, and retain its zero/NaN truthiness before taking the reciprocal.
        const test='__arg'+serial++;prefix.push(`const ${test}=${expression(input.left,env)};`);
        argument=`(${test}===0 || ${test}!==${test}?${expression(input.right,env)}:${test})`;
      } else argument=value(input,env);
      if (numeric(argument)) {
        const local='__arg'+serial++;prefix.push(`let ${local}=${argument};`);scope.set(param.name,local);
      } else scope.set(param.name,argument);
    });
    // A method call's arguments are evaluated before its body, including when
    // the receiver and another virtual vector are the same object (cross()).
    return '{'+prefix.join('')+method.body.body.map(statement=>{
      if (receiver.kind==='attribute' && statement.type==='IfStatement') return '';
      if (statement.type==='ReturnStatement') {
        if (statement.argument.type==='ThisExpression')return '';
        return call(statement.argument,scope);
      }
      return emit(statement,scope);
    }).join('')+'}';
  }
  function emit(node,env) {
    if (node.type==='BlockStatement')return '{'+node.body.map(statement=>emit(statement,env)).join('')+'}';
    if (node.type==='VariableDeclaration') {
      const output=[];
      for(const declaration of node.declarations) {
        if (declaration.init?.type==='NewExpression' && declaration.init.callee.name==='Vector3') {
          const name=declaration.id.name;env.set(name,vectorValue(name));
          output.push(`let ${name}_x=0,${name}_y=0,${name}_z=0;`);
        } else output.push(`${node.kind} ${declaration.id.name}=${expression(declaration.init,env)};`);
      }
      return output.join('');
    }
    if (node.type==='ExpressionStatement')return node.expression.type==='CallExpression'
      ?call(node.expression,env):expression(node.expression,env)+';';
    if (node.type==='IfStatement')return `if(${expression(node.test,env)})${emit(node.consequent,new Map(env))}`+
      (node.alternate?'else '+emit(node.alternate,new Map(env)):'');
    if (node.type==='ForStatement') {
      const scope=new Map(env),init=node.init.declarations.map(item=>item.id.name+'='+expression(item.init,scope)).join(',');
      return `for(${node.init.kind} ${init};${expression(node.test,scope)};${expression(node.update,scope)})${emit(node.body,scope)}`;
    }
    throw new TypeError(`Unsupported verified normal statement: ${node.type}`);
  }
  const computeEnv=new Map([['index',attr('index','indices','indexCount',1)],
    ['positionAttribute',attr('positionAttribute','positions','positionCount',3,'positionStride','positionOffset')],
    ['normalAttribute',attr('normalAttribute','normals','normalCount',3,'normalStride','normalOffset')]]);
  // The sole object-truthiness branch is the already captured index choice.
  const segment=computeBody.slice(first,end),branch=segment.find(node=>node.type==='IfStatement');
  const computeSource='function computeNormals(positions,indices,normals,positionCount,normalCount,indexCount,indexed,positionStride,positionOffset,normalStride,normalOffset,positionNormalized){'+
    segment.map(node=>node===branch
      ?`if(indexed)${emit(node.consequent,new Map(computeEnv))}else ${emit(node.alternate,new Map(computeEnv))}`
      :emit(node,computeEnv)).join('')+'}';
  const normalizeEnv=new Map([['normals',attr('normals','normals','normalCount',3,'normalStride','normalOffset')],
    [scratchName,vectorValue('__scratch')]]);
  const normalizeSource='function normalizeNormals(normals,scratch,normalCount,normalStride,normalOffset){'+
    'let __scratch_x=scratch[0],__scratch_y=scratch[1],__scratch_z=scratch[2];'+emit(normalizationLoop,normalizeEnv)+
    'scratch[0]=__scratch_x;scratch[1]=__scratch_y;scratch[2]=__scratch_z;}';
  return {computeSource,normalizeSource,computeSpan:{start:segment[0].start,end:segment.at(-1).end},
    normalizeSpan:{start:normalizationLoop.start,end:normalizationLoop.end},scratchName,
    requirements:{vector:[...requirements.vector],attribute:[...requirements.attribute]},
    denormalizers:mathUtilsSource?deriveAttributeDenormalizers(mathUtilsSource):null};
}

export function compileGeometryNormalKernels(sources,{maxMemoryPages=2048,maxIterations=100000000}={}) {
  const settings={maxMemoryPages,maxIterations,generalControl:true,allowMath:true};
  const variants=[];
  const positionTypes=sources.denormalizers?['f64[]',...sources.denormalizers.keys()]:['f32[]','f64[]'];
  for(const positionType of positionTypes)for(const normalType of ['f32[]','f64[]'])for(const indexType of ['u16[]','u32[]']) {
    const parameterTypes=[positionType,indexType,normalType,'f64','f64','f64','f64','f64','f64','f64','f64','f64'];
    // Float64 storage is admitted only when positionNormalized is false.
    const helper=sources.denormalizers?.get(positionType)?.source??'function __f3d_denormalize(value){return value;}';
    const artifact=compileNumericKernel(sources.computeSource,{...settings,parameterTypes,
      helperSources:new Map([['__f3d_denormalize',helper]]),sourceName:'pinned BufferGeometry.computeVertexNormals'});
    variants.push({method:'compute',types:[positionType,indexType,normalType],normalized:!!sources.denormalizers,
      base64:Buffer.from(artifact.wasm).toString('base64')});
  }
  for(const normalType of ['f32[]','f64[]']) {
    const artifact=compileNumericKernel(sources.normalizeSource,{...settings,parameterTypes:[normalType,'f64[]','f64','f64','f64'],sourceName:'pinned BufferGeometry.normalizeNormals'});
    variants.push({method:'normalize',types:[normalType],base64:Buffer.from(artifact.wasm).toString('base64')});
  }
  return variants;
}
