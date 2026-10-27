/** Bulk attribute transforms derived from the pinned BufferAttribute/Vector3
 * bodies. The source owns constructors, method identity and conservative paths;
 * only the guarded numeric loop executes in native Wasm. */
import {sourceBlob} from './marching_cubes_specialization.mjs';
import {normalClasses,normalMethod} from './geometry_normals_compile.mjs';
import {geometryNormalSourceRole} from './geometry_normals_specialization.mjs';
import {compileNumericKernel} from './numeric_kernel.mjs';

const methods=['applyMatrix4','applyMatrix3','applyNormalMatrix','transformDirection'];
const matrix3Blob='1ce4def6d838f12ef4feb7a5c1b4149efb39fcdf';
const fresh=(source,name)=>{let value='__f3d_attribute_'+name;while(source.includes(value))value+='_';return value;};
export function geometryTransformSourceRole(source) {
  const role=geometryNormalSourceRole(source);
  return ['core','attributes','vector'].includes(role)?role:sourceBlob(source)===matrix3Blob?'matrix3':null;
}

/** Inline only the verified vector/attribute calls. Arithmetic, component loads,
 * store order, perspective division and normalization come from those bodies. */
export function deriveGeometryTransformKernels(attributeSource,vectorSource) {
  const attribute=normalClasses(attributeSource).get('BufferAttribute'),vector=normalClasses(vectorSource).get('Vector3');
  if(!attribute || !vector)throw new TypeError('Missing verified attribute-transform classes');
  let serial=0;
  const requirements={attribute:new Set(),vector:new Set()},derived=[];
  const attr={kind:'attribute'},vec={kind:'vector'},matrix={kind:'matrix'};
  function value(node,env) {
    if(node.type==='Identifier')return env.get(node.name)??node.name;
    if(node.type==='ThisExpression')return env.get('this');
    return expression(node,env);
  }
  function expression(node,env) {
    if(node.type==='Identifier' || node.type==='ThisExpression') {
      const result=value(node,env);if(typeof result!=='string')throw new TypeError('Transform object cannot escape');return result;
    }
    if(node.type==='Literal')return node.raw??JSON.stringify(node.value);
    if(node.type==='MemberExpression') {
      const object=value(node.object,env),key=node.property.name;
      if(object===vec && !node.computed && ['x','y','z'].includes(key))return 'v_'+key;
      if(object===attr && !node.computed) {
        if(key==='array')return 'values';if(key==='count')return 'count';if(key==='itemSize')return '3';
      }
      if(object===matrix && !node.computed && key==='elements')return 'elements';
      if(typeof object!=='string')throw new TypeError('Unsupported transform member');
      return node.computed?`${object}[${expression(node.property,env)}]`:`${object}.${key}`;
    }
    if(['BinaryExpression','AssignmentExpression','LogicalExpression'].includes(node.type))
      return `(${expression(node.left,env)} ${node.operator} ${expression(node.right,env)})`;
    if(node.type==='UpdateExpression')return node.prefix?node.operator+expression(node.argument,env):expression(node.argument,env)+node.operator;
    if(node.type==='UnaryExpression')return `(${node.operator}${expression(node.argument,env)})`;
    if(node.type==='CallExpression' && node.callee.type==='MemberExpression' && !node.callee.computed) {
      const receiver=value(node.callee.object,env),name=node.callee.property.name;
      if(receiver==='Math')return `Math.${name}(${node.arguments.map(arg=>expression(arg,env)).join(',')})`;
      if(receiver===attr && ['getX','getY','getZ'].includes(name)) {
        requirements.attribute.add(name);const method=normalMethod(attribute,name);
        const scope=new Map(env).set('this',attr).set(method.params[0].name,expression(node.arguments[0],env));
        return expression(method.body.body[0].declarations[0].init,scope);
      }
      if(receiver===vec && name==='length') {
        requirements.vector.add(name);
        return expression(normalMethod(vector,name).body.body[0].argument,new Map(env).set('this',vec));
      }
    }
    throw new TypeError('Unsupported verified transform expression: '+node.type);
  }
  function call(node,env) {
    const previous=node.callee.object.type==='CallExpression'?call(node.callee.object,env):null;
    const receiver=previous?previous.receiver:value(node.callee.object,env),name=node.callee.property.name;
    if(receiver!==vec && receiver!==attr)throw new TypeError('Unsupported transform call');
    const kind=receiver.kind,method=normalMethod(kind==='vector'?vector:attribute,name),prefix=[];
    if(!method)throw new TypeError('Missing verified transform method');
    requirements[kind].add(name);const scope=new Map(env).set('this',receiver);
    method.params.forEach((param,index)=>{
      const input=node.arguments[index];let argument;
      if(input.type==='LogicalExpression' && input.operator==='||') {
        const local='__arg'+serial++;prefix.push(`const ${local}=${expression(input.left,env)};`);
        argument=`(${local}===0 || ${local}!==${local}?${expression(input.right,env)}:${local})`;
      }else argument=value(input,env);
      if(typeof argument==='string') {
        const local='__arg'+serial++;prefix.push(`let ${local}=${argument};`);scope.set(param.name,local);
      }else scope.set(param.name,argument);
    });
    const code=method.body.body.map(statement=>{
      if(kind==='attribute' && statement.type==='IfStatement')return ''; // normalized=false guard
      if(statement.type==='ReturnStatement')return statement.argument.type==='ThisExpression'?'':call(statement.argument,scope).code;
      return emit(statement,scope);
    }).join('');
    return {receiver,code:(previous?.code??'')+'{'+prefix.join('')+code+'}'};
  }
  function emit(node,env) {
    if(node.type==='BlockStatement')return '{'+node.body.map(statement=>emit(statement,env)).join('')+'}';
    if(node.type==='VariableDeclaration')return node.declarations.map(declaration=>{
      const init=expression(declaration.init,env),local='__local'+serial++;
      env.set(declaration.id.name,local);return `${node.kind} ${local}=${init};`;
    }).join('');
    if(node.type==='ExpressionStatement')return node.expression.type==='CallExpression'?call(node.expression,env).code:expression(node.expression,env)+';';
    if(node.type==='ForStatement') {
      const scope=new Map(env),init=node.init.declarations.map(declaration=>{
        const expressionValue=expression(declaration.init,scope),local='__local'+serial++;scope.set(declaration.id.name,local);
        return local+'='+expressionValue;
      }).join(',');
      return `for(${node.init.kind} ${init};${expression(node.test,scope)};${expression(node.update,scope)})${emit(node.body,scope)}`;
    }
    throw new TypeError('Unsupported verified transform statement: '+node.type);
  }
  for(const name of methods) {
    const method=normalMethod(attribute,name),body=method.body.body;
    const loop=name==='applyMatrix3'?body[0].alternate.consequent.body[0]:body[0];
    const scratch=loop.body.body[0].expression.callee.object.name;
    const env=new Map([['this',attr],[method.params[0].name,matrix],[scratch,vec]]);
    const source=`function transform(values,elements,scratch,count){let v_x=scratch[0],v_y=scratch[1],v_z=scratch[2];`+
      emit(loop,env)+'scratch[0]=v_x;scratch[1]=v_y;scratch[2]=v_z;}';
    derived.push({name,source,scratch,span:{start:body[0].start,end:body.at(-1).start}});
  }
  return {methods:derived,requirements:{attribute:[...requirements.attribute],vector:[...requirements.vector]}};
}

export function specializeGeometryTransforms(source,{runtimeModule,vectorSource,maxMemoryPages=2048,maxIterations=100000000}={}) {
  const role=geometryTransformSourceRole(source);
  if(!role)return {changed:false,edits:[],header:'',tail:''};
  const classes=normalClasses(source),edits=[],imports=[],tail=[];
  const report={sourceBlob:sourceBlob(source),upstreamCommit:'148ef33ecb6d2502ff796d4554abd1549c95d519',
    role,compiledMethods:[],sourceDerived:true,accelerationClaim:false};
  if(role==='matrix3' || role==='core') {
    const ctor=normalMethod(classes.get('Matrix3'),'constructor');
    const allocation=ctor.body.body.find(node=>node.expression?.left?.property?.name==='elements').expression.right;
    const register=fresh(source,'matrix'),elements=fresh(source,'elements');
    imports.push(`import {registerGeometryTransformMatrix as ${register},registerGeometryTransformElements as ${elements}} from ${JSON.stringify(runtimeModule)};`);
    edits.push({start:ctor.body.start+1,end:ctor.body.start+1,text:`\n${register}(this);\n`},
      {start:allocation.start,end:allocation.end,text:`${elements}(${source.slice(allocation.start,allocation.end)})`});
  }
  if(role==='vector' || role==='core') {
    const register=fresh(source,'vector');
    imports.push(`import {registerGeometryTransformVector as ${register}} from ${JSON.stringify(runtimeModule)};`);
    tail.push(`${register}(Vector3);`);
  }
  if(role==='attributes' || role==='core') {
    if(role==='core')vectorSource=source;
    if(role!=='core' && geometryTransformSourceRole(vectorSource??'')!=='vector')
      return {changed:false,edits:[],header:'',tail:'',report:{...report,reason:'DEPENDENCY_PIN_MISMATCH'}};
    const derived=deriveGeometryTransformKernels(source,vectorSource),variants=[];
    for(const method of derived.methods)for(const type of ['f32[]','f64[]']) {
      const artifact=compileNumericKernel(method.source,{parameterTypes:[type,'f64[]','f64[]','f64'],
        generalControl:true,allowMath:true,maxMemoryPages,maxIterations,sourceName:'pinned BufferAttribute.'+method.name});
      variants.push({method:method.name,type,base64:Buffer.from(artifact.wasm).toString('base64')});
    }
    const register=fresh(source,'register'),create=fresh(source,'create'),token=fresh(source,'dispatch'),attempt=fresh(source,'try');
    imports.push(`import {registerGeometryTransformAttributes as ${register},createGeometryTransformDispatch as ${create},tryGeometryTransform as ${attempt}} from ${JSON.stringify(runtimeModule)};var ${token};`);
    tail.push(`${register}(BufferAttribute,[Float32BufferAttribute]);`,`${token}=${create}(${JSON.stringify(variants)},Vector3,()=>Math);`);
    for(const method of derived.methods)edits.push({...method.span,text:`if (!${attempt}(this,${token},${JSON.stringify(method.name)},m,${method.scratch})) {\n`+
      source.slice(method.span.start,method.span.end)+'\n}\n'});
    report.compiledMethods=methods.slice();report.wasmVariants=variants.length;
    report.wasmBytes=variants.reduce((sum,variant)=>sum+Buffer.from(variant.base64,'base64').length,0);
  }
  return {changed:true,edits,header:imports.join('\n')+'\n',tail:'\n'+tail.join('\n')+'\n',report};
}
