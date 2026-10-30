/** Exact pinned sources select tangent lowering; method names alone never do. */
import * as walk from 'acorn-walk';
import {sourceBlob} from './marching_cubes_specialization.mjs';
import {geometryNormalSourceRole} from './geometry_normals_specialization.mjs';
import {normalClasses,normalMethod} from './geometry_normals_compile.mjs';
import {deriveGeometryTangentKernel,compileGeometryTangentKernels} from './geometry_tangents_compile.mjs';
const vector2Blob='c1c5c71d766647570533b7d392e1484b3a2e6eea';
const fresh=(source,name)=>{let value='__f3d_tangent_'+name;while(source.includes(value))value+='_';return value;};
export function geometryTangentSourceRole(source) {
  return sourceBlob(source)===vector2Blob?'vector2':geometryNormalSourceRole(source);
}
export function specializeGeometryTangents(source,{runtimeModule,vectorSource,vector2Source,attributeSource,
  maxMemoryPages=2048,maxIterations=100000000}={}) {
  const role=geometryTangentSourceRole(source);
  if(!role)return {changed:false,edits:[],header:'',tail:''};
  const classes=normalClasses(source),edits=[],headers=[],tails=[],register=fresh(source,'class');
  const report={sourceBlob:sourceBlob(source),upstreamCommit:'148ef33ecb6d2502ff796d4554abd1549c95d519',role,
    compiledMethods:[],accelerationClaim:false};
  headers.push(`import {registerGeometryTangentClass as ${register}} from ${JSON.stringify(runtimeModule)};`);
  if(role==='core' || role==='vector')tails.push(`${register}('Vector3',Vector3);`);
  if(role==='core' || role==='vector2')tails.push(`${register}('Vector2',Vector2);`);
  if(role==='core' || role==='attributes')
    tails.push(`${register}('attribute',BufferAttribute,[Float32BufferAttribute,Uint16BufferAttribute,Uint32BufferAttribute]);`);
  if(role==='core' || role==='geometry') {
    if(role==='core'){vectorSource=source;vector2Source=source;attributeSource=source;}
    else if(geometryTangentSourceRole(vectorSource??'')!=='vector' || geometryTangentSourceRole(vector2Source??'')!=='vector2' ||
        geometryTangentSourceRole(attributeSource??'')!=='attributes')
      return {changed:false,edits:[],header:'',tail:'',report:{...report,reason:'DEPENDENCY_PIN_MISMATCH'}};
    const derived=deriveGeometryTangentKernel(source,vectorSource,vector2Source,attributeSource);
    const variants=compileGeometryTangentKernels(derived,{maxMemoryPages,maxIterations});
    const attempt=fresh(source,'try'),create=fresh(source,'create'),token=fresh(source,'dispatch'),allocation=fresh(source,'allocation');
    headers.push(`import {tryGeometryTangents as ${attempt},createGeometryTangentDispatch as ${create},`+
      `registerGeometryTangentAllocation as ${allocation}} from ${JSON.stringify(runtimeModule)};var ${token};`);
    edits.push({...derived.span,text:`if(!${attempt}(this,${token},positionAttribute,normalAttribute,uvAttribute,index,tangentAttribute)) {\n`+
      source.slice(derived.span.start,derived.span.end)+'\n}'});
    const geometry=classes.get('BufferGeometry');
    walk.simple(geometry,{AssignmentExpression(node){
      if(node.left.type==='MemberExpression' && node.left.object.type==='ThisExpression' && node.left.property.name==='groups' &&
          node.right.type==='ArrayExpression' && node.right.elements.length===0)
        edits.push({start:node.right.start,end:node.right.end,text:`${allocation}(${source.slice(node.right.start,node.right.end)})`});
    }});
    const group=normalMethod(geometry,'addGroup').body.body[0].expression.arguments[0];
    if(group.type!=='ObjectExpression')throw new TypeError('Missing pinned group allocation');
    edits.push({start:group.start,end:group.end,text:`${allocation}(${source.slice(group.start,group.end)})`});
    tails.push(`${token}=${create}(${JSON.stringify(variants)},Vector3,Vector2,()=>Math,()=>isFinite,${maxMemoryPages*65536});`);
    report.compiledMethods=['BufferGeometry.computeTangents'];report.sourceDerived=true;report.wasmVariants=variants.length;
    report.wasmBytes=variants.reduce((sum,variant)=>sum+Buffer.from(variant.base64,'base64').length,0);
  }
  return {changed:true,edits,header:headers.join('\n')+'\n',tail:'\n'+tails.join('\n')+'\n',report};
}
