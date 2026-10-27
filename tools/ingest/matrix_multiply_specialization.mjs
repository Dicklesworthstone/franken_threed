/** Source-pinned Matrix4 arithmetic. Public methods, calls and arrays stay retained. */
import {createHash} from 'node:crypto';
import {sourceBlob} from './marching_cubes_specialization.mjs';
import {normalClasses,normalMethod} from './geometry_normals_compile.mjs';
import {compileNumericKernel} from './numeric_kernel.mjs';
const matrixBlob='6522a6b57300d20bf6a36fbcbb3862e81291d060';
const coreHash='9edde002b066a9a05676a6127f67735b62baf399bdea529f2f7e31657da769e6';
const fresh=(source,name)=>{let value='__f3d_matrix_'+name;while(source.includes(value))value+='_';return value;};

export function matrixMultiplySourceRole(source) {
  if(sourceBlob(source)===matrixBlob)return 'matrix';
  return createHash('sha256').update(source).digest('hex')===coreHash?'core':null;
}

/** The pinned method loads both complete operands before its first store. The
 * compiler consumes those actual statements, preserving expression order,
 * projective terms and authored nonfinite/signed-zero values without a TRS
 * decomposition or normalization. No synthetic source loop is introduced. */
export function specializeMatrixMultiply(source,{runtimeModule,maxMemoryPages=2048,maxIterations=100000000}={}) {
  const role=matrixMultiplySourceRole(source);
  if(!role)return {changed:false,edits:[],header:'',tail:''};
  const cls=normalClasses(source).get('Matrix4'),ctor=normalMethod(cls,'constructor');
  const method=normalMethod(cls,'multiplyMatrices'),body=method.body.body;
  const allocation=ctor.body.body.find(node=>node.type==='ExpressionStatement' &&
    node.expression.type==='AssignmentExpression' && node.expression.left.property?.name==='elements').expression.right;
  const prefix=body.slice(0,3),last=body.at(-1);
  const parameters=prefix.map(node=>node.declarations[0].id.name);
  const kernelSource='function multiplyMatrices('+parameters.join(',')+'){'+source.slice(body[3].start,last.start)+'}';
  const artifact=compileNumericKernel(kernelSource,{parameterTypes:['f64[]','f64[]','f64[]'],
    generalControl:true,maxMemoryPages,maxIterations,sourceName:'pinned Matrix4.multiplyMatrices'});
  const register=fresh(source,'allocation'),elements=fresh(source,'elements'),create=fresh(source,'create');
  const token=fresh(source,'token'),attempt=fresh(source,'try_multiply');
  const header=`import {registerMatrixObject as ${register},registerMatrixElements as ${elements},createMatrixMultiplyDispatch as ${create},tryMatrixMultiply as ${attempt}} from ${JSON.stringify(runtimeModule)};var ${token};\n`;
  return {changed:true,header,tail:`\n${token}=${create}(${JSON.stringify(Buffer.from(artifact.wasm).toString('base64'))});\n`,
    edits:[
      {start:ctor.body.start+1,end:ctor.body.start+1,text:`\n${register}(this);\n`},
      {start:allocation.start,end:allocation.end,text:`${elements}(${source.slice(allocation.start,allocation.end)})`},
      {start:body[0].start,end:last.start,text:`if (!${attempt}(this,${token},a,b)) {\n${source.slice(body[0].start,last.start)}\n}\n`},
    ],report:{sourceBlob:sourceBlob(source),upstreamCommit:'148ef33ecb6d2502ff796d4554abd1549c95d519',
      role,compiledMethods:['multiplyMatrices'],sourceDerived:true,wasmBytes:artifact.wasm.length,accelerationClaim:false}};
}
