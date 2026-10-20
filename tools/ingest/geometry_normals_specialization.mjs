/** Exact pinned source selects the loop lowering; names alone never authorize it. */
import {createHash} from 'node:crypto';
import {sourceBlob} from './marching_cubes_specialization.mjs';
import {normalClasses,normalMethod,deriveGeometryNormalKernels,compileGeometryNormalKernels} from './geometry_normals_compile.mjs';
const blobs={geometry:'db7bfe67f8b159c9126da0c55e1cb5db66eca529',
  attributes:'958fb453e027aee4a048001b3232c92d6fd6716b',vector:'b2e4c4e65cc50afc89b6cd8f8e706a0d1001d71a'};
const coreHash='9edde002b066a9a05676a6127f67735b62baf399bdea529f2f7e31657da769e6';
export function geometryNormalSourceRole(source) {
  const blob=sourceBlob(source);
  for(const [role,expected] of Object.entries(blobs))if(blob===expected)return role;
  return createHash('sha256').update(source).digest('hex')===coreHash?'core':null;
}
const fresh=(source,name)=>{let candidate='__f3d_geometry_'+name;while(source.includes(candidate))candidate+='_';return candidate;};

/** Return original-coordinate edits so independently pinned library transforms
 * can share one core module without weakening either source fingerprint. */
export function specializeGeometryNormals(source,{runtimeModule,objectsModule,vectorSource,attributeSource,
  maxMemoryPages=2048,maxIterations=100000000}={}) {
  const role=geometryNormalSourceRole(source);
  if(!role)return {changed:false,edits:[],header:'',tail:''};
  const classes=normalClasses(source),edits=[],imports=[],tail=[],report={sourceBlob:sourceBlob(source),
    upstreamCommit:'148ef33ecb6d2502ff796d4554abd1549c95d519',role,compiledMethods:[],accelerationClaim:false};
  const registerVector=fresh(source,'register_vector'),registerAttributes=fresh(source,'register_attributes');
  if(role==='core' || role==='vector') {
    const register=fresh(source,'allocation'),ctor=normalMethod(classes.get('Vector3'),'constructor');
    edits.push({start:ctor.body.start+1,end:ctor.body.start+1,text:`\n${register}(this);\n`});
    imports.push(`import {registerNumericObject as ${register}} from ${JSON.stringify(objectsModule)};`);
    imports.push(`import {registerGeometryNormalVector as ${registerVector}} from ${JSON.stringify(runtimeModule)};`);
    tail.push(`${registerVector}(Vector3);`);
  }
  if(role==='core' || role==='attributes') {
    imports.push(`import {registerGeometryNormalAttributes as ${registerAttributes}} from ${JSON.stringify(runtimeModule)};`);
    tail.push(`${registerAttributes}(BufferAttribute,[Float32BufferAttribute,Uint16BufferAttribute,Uint32BufferAttribute]);`);
  }
  if(role==='core' || role==='geometry') {
    if(role==='core'){vectorSource=source;attributeSource=source;}
    if(role!=='core' && (geometryNormalSourceRole(vectorSource??'')!=='vector' ||
        geometryNormalSourceRole(attributeSource??'')!=='attributes')) {
      return {changed:false,edits:[],header:'',tail:'',report:{...report,reason:'DEPENDENCY_PIN_MISMATCH'}};
    }
    const sources=deriveGeometryNormalKernels(source,vectorSource,attributeSource);
    const variants=compileGeometryNormalKernels(sources,{maxMemoryPages,maxIterations});
    const tryName=fresh(source,'try_normals'),create=fresh(source,'create_normals'),token=fresh(source,'normals');
    imports.push(`import {tryGeometryNormals as ${tryName},createGeometryNormalDispatch as ${create}} from ${JSON.stringify(runtimeModule)};var ${token};`);
    const {computeSpan,normalizeSpan}=sources;
    edits.push({...computeSpan,text:`if (!${tryName}(this,${token},'compute',positionAttribute,index,normalAttribute)) {\n`+
      source.slice(computeSpan.start,computeSpan.end)+'\n}'});
    edits.push({...normalizeSpan,text:`if (!${tryName}(this,${token},'normalize',null,null,normals,${sources.scratchName})) {\n`+
      source.slice(normalizeSpan.start,normalizeSpan.end)+'\n}'});
    tail.push(`${token}=${create}(${JSON.stringify(variants)},Vector3,()=>Math);`);
    report.compiledMethods=['computeVertexNormals','normalizeNormals'];
    report.wasmVariants=variants.length;
    report.wasmBytes=variants.reduce((sum,variant)=>sum+Buffer.from(variant.base64,'base64').length,0);
    report.sourceDerived=true;
  }
  return {changed:edits.length>0 || imports.length>0,edits,header:imports.join('\n')+'\n',tail:'\n'+tail.join('\n')+'\n',report};
}
