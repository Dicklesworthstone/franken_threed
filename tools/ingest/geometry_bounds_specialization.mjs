/** Exact source identities admit bounds scans; modified sources remain JS. */
import {sourceBlob} from './marching_cubes_specialization.mjs';
import {geometryNormalSourceRole} from './geometry_normals_specialization.mjs';
import {normalClasses,normalMethod} from './geometry_normals_compile.mjs';
import {deriveGeometryBoxKernel,compileGeometryBoxKernels} from './geometry_bounds_compile.mjs';
const boxBlob='675f74156954c1df38cafb2862d8bbd21a0e6b98';
const fresh=(source,name)=>{let value='__f3d_bounds_'+name;while(source.includes(value))value+='_';return value;};
export function geometryBoundsSourceRole(source) {
  if(sourceBlob(source)===boxBlob)return 'box';
  const role=geometryNormalSourceRole(source);return ['core','vector','attributes'].includes(role)?role:null;
}
export function specializeGeometryBounds(source,{runtimeModule,objectsModule,vectorSource,attributeSource,
  maxMemoryPages=2048,maxIterations=100000000}={}) {
  const role=geometryBoundsSourceRole(source);if(!role)return {changed:false,edits:[],header:'',tail:''};
  const classes=normalClasses(source),edits=[],headers=[],tails=[];
  const register=fresh(source,'class'),allocation=fresh(source,'allocation');
  headers.push('import {registerGeometryBoundsClass as '+register+'} from '+JSON.stringify(runtimeModule)+';');
  if(role==='vector' || role==='core')tails.push(register+'("vector",Vector3);');
  if(role==='attributes' || role==='core')tails.push(register+'("attribute",BufferAttribute,Float32BufferAttribute);');
  const report={sourceBlob:sourceBlob(source),upstreamCommit:'148ef33ecb6d2502ff796d4554abd1549c95d519',role,
    compiledMethods:[],accelerationClaim:false};
  if(role==='box' || role==='core') {
    if(role==='core'){vectorSource=source;attributeSource=source;}
    if(role==='box' && (geometryNormalSourceRole(vectorSource??'')!=='vector' ||
        geometryNormalSourceRole(attributeSource??'')!=='attributes'))
      return {changed:false,edits:[],header:'',tail:'',report:{...report,reason:'DEPENDENCY_PIN_MISMATCH'}};
    const derived=deriveGeometryBoxKernel(source,vectorSource,attributeSource);
    const variants=compileGeometryBoxKernels(derived,{maxMemoryPages,maxIterations});
    const create=fresh(source,'create'),token=fresh(source,'token'),attempt=fresh(source,'try');
    headers.push('import {registerNumericObject as '+allocation+'} from '+JSON.stringify(objectsModule)+';',
      'import {createGeometryBoundsDispatch as '+create+',tryGeometryBoxBounds as '+attempt+'} from '+JSON.stringify(runtimeModule)+';var '+token+';');
    const ctor=normalMethod(classes.get('Box3'),'constructor');
    edits.push({start:ctor.body.start+1,end:ctor.body.start+1,text:'\n'+allocation+'(this);\n'},
      {...derived.span,text:'if(!'+attempt+'(this,'+token+',attribute,'+derived.scratchName+')){\n'+
        source.slice(derived.span.start,derived.span.end)+'\n}'});
    tails.push(register+'("box",Box3);',token+'='+create+'('+JSON.stringify(variants)+',()=>Math);');
    report.compiledMethods=['Box3.setFromBufferAttribute'];report.sourceDerived=true;
    report.wasmBytes=variants.reduce((sum,variant)=>sum+Buffer.from(variant.base64,'base64').length,0);
  }
  return {changed:true,edits,header:headers.join('\n')+'\n',tail:'\n'+tails.join('\n')+'\n',report};
}
