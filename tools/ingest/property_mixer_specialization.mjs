/** Source-derived numeric PropertyMixer transactions. Binding calls and the
 * action/event loop remain in the original synchronous JavaScript program. */
import {createHash} from 'node:crypto';
import * as walk from 'acorn-walk';
import {sourceBlob} from './marching_cubes_specialization.mjs';
import {normalClasses,normalMethod} from './geometry_normals_compile.mjs';
import {compileNumericKernel} from './numeric_kernel.mjs';
import {deriveQuaternionFlatKernels} from './quaternion_flat_specialization.mjs';

const propertyBlob='e591f50e17e6f869b17ad1822b537b741e3bb2fd';
const quaternionBlob='c9faca31665218cee87a7110e2badc7ba868beca';
const coreHash='9edde002b066a9a05676a6127f67735b62baf399bdea529f2f7e31657da769e6';
const fresh=(source,name)=>{let value='__f3d_property_mixer_'+name;while(source.includes(value))value+='_';return value;};
const fields={buffer:'values',valueSize:'valueSize',_origIndex:'origIndex',_addIndex:'addIndex',
  _workIndex:'workIndex',
  cumulativeWeight:'state[0]',cumulativeWeightAdditive:'state[1]',
  _mixBufferRegion:'lerp',_mixBufferRegionAdditive:'lerpAdditive'};

export function propertyMixerSourceRole(source) {
  return sourceBlob(source)===propertyBlob?'property-mixer':
    createHash('sha256').update(source).digest('hex')===coreHash?'core':null;
}

export function specializePropertyMixer(source,{runtimeModule,quaternionSource,maxMemoryPages=2048,maxIterations=100000000}={}) {
  const role=propertyMixerSourceRole(source);
  if(!role)return {changed:false,edits:[],header:'',tail:''};
  if(role==='core')quaternionSource=source;
  const quaternion=role==='core'||sourceBlob(quaternionSource??'')===quaternionBlob;
  const cls=normalClasses(source).get('PropertyMixer'),ctor=normalMethod(cls,'constructor');
  function rewrite(node,extra=[]) {
    const changes=[...extra];
    walk.simple(node,{
      MemberExpression(value) {
        if(value.object.type==='ThisExpression' && Object.hasOwn(fields,value.property.name))
          changes.push({start:value.start,end:value.end,text:fields[value.property.name]});
        if(value.object.name==='Quaternion' && ['slerpFlat','multiplyQuaternionsFlat'].includes(value.property.name))
          changes.push({start:value.start,end:value.end,text:value.property.name});
      },
      CallExpression(value) {
        if(value.callee.object?.type==='ThisExpression' && value.callee.property.name==='_setIdentity')
          changes.push({start:value.start,end:value.end,text:'identity(values,valueSize,addIndex)'});
        if(value.callee.object?.type==='ThisExpression' && value.callee.property.name==='_setAdditiveIdentityNumeric')
          changes.push({start:value.start,end:value.end,text:'numericIdentity(values,valueSize,addIndex)'});
        if(value.callee.object?.type==='ThisExpression' &&
            ['_mixBufferRegion','_mixBufferRegionAdditive'].includes(value.callee.property.name))
          changes.push({start:value.end-1,end:value.end-1,text:',workIndex'});
      },
    });
    let text=source.slice(node.start,node.end);
    for(const change of changes.sort((a,b)=>b.start-a.start))
      text=text.slice(0,change.start-node.start)+change.text+text.slice(change.end-node.start);
    return text;
  }
  const helpers=new Map();
  for(const [method,name]of [['_lerp','lerp'],['_lerpAdditive','lerpAdditive'],['_setAdditiveIdentityNumeric','identity']]) {
    const value=normalMethod(cls,method),parameters=name==='identity'?['values','valueSize','addIndex']:[...value.params.map(x=>x.name),'workIndex'];
    helpers.set(name,`function ${name}(${parameters.join(',')})${rewrite(value.body)}`);
  }
  const quaternionHelpers=new Map();
  if(quaternion) {
    const derived=deriveQuaternionFlatKernels(quaternionSource);
    quaternionHelpers.set('slerpFlat',`function slerpFlat(${derived.slerp.params.map(x=>x.name).join(',')})`+
      quaternionSource.slice(derived.slerp.body.start,derived.slerp.body.end));
    quaternionHelpers.set('multiplyQuaternionsFlat',derived.multiplySource);
    quaternionHelpers.set('numericIdentity',helpers.get('identity').replace('function identity(','function numericIdentity('));
    for(const [method,name]of [['_slerp','lerp'],['_slerpAdditive','lerpAdditive'],['_setAdditiveIdentityQuaternion','identity']]) {
      const value=normalMethod(cls,method),parameters=name==='identity'?['values','valueSize','addIndex']:
        ['buffer','dstOffset','srcOffset','t','stride','workIndex'];
      quaternionHelpers.set(name,`function ${name}(${parameters.join(',')})${rewrite(value.body)}`);
    }
  }
  const create=fresh(source,'create'),register=fresh(source,'register'),attempt=fresh(source,'try'),token=fresh(source,'token');
  const edits=[{start:ctor.body.start+1,end:ctor.body.start+1,text:`\n${register}(this);\n`}],variants={};
  let wasmBytes=0;
  for(const name of ['accumulate','accumulateAdditive','apply']) {
    const method=normalMethod(cls,name),extra=[];
    let bindingCall;
    if(name==='apply') {
      const declaration=method.body.body[0],last=declaration.declarations.at(-1),previous=declaration.declarations.at(-2);
      if(last.id.name!=='binding')throw new TypeError('Missing pinned PropertyMixer binding capture');
      extra.push({start:previous.end,end:last.end,text:''});
      const loop=method.body.body.at(-1),branch=loop.body.body[0];
      if(branch.type!=='IfStatement' || branch.consequent.body[0].expression?.callee.object.name!=='binding')
        throw new TypeError('Missing pinned PropertyMixer publication boundary');
      bindingCall=source.slice(branch.consequent.body[0].start,branch.consequent.body[0].end);
      extra.push({start:branch.consequent.start,end:branch.consequent.end,text:'{return 1;}'},
        {start:method.body.end-1,end:method.body.end-1,text:'\nreturn 0;\n'});
    }
    const parameters=['values','state','valueSize','origIndex','addIndex','workIndex',...method.params.map(x=>x.name)];
    for(const profile of quaternion?['numeric','quaternion']:['numeric']) {
      const artifact=compileNumericKernel(`function ${name}(${parameters.join(',')})${rewrite(method.body,extra)}`,{
        parameterTypes:['f64[]','f64[]',...parameters.slice(2).map(()=>'f64')],
        helperSources:profile==='quaternion'?quaternionHelpers:helpers,
        generalControl:true,allowMath:profile==='quaternion',maxMemoryPages,maxIterations,sourceName:'pinned PropertyMixer.'+name});
      variants[profile==='numeric'?name:'quaternion:'+name]=Buffer.from(artifact.wasm).toString('base64');wasmBytes+=artifact.wasm.length;
    }
    const result=fresh(source,name+'_result');
    const publication=name==='apply'?`if(${result}) {const binding=${result}.binding,buffer=${result}.buffer,offset=${result}.offset;${bindingCall}}`:'';
    edits.push({start:method.body.start+1,end:method.body.start+1,text:
      `\nconst ${result}=${attempt}(this,${token},${JSON.stringify(name)},${method.params.map(x=>x.name).join(',')});\n`+
      `if(${result}!==null) {${publication}return;}\n`});
  }
  return {changed:true,edits,
    header:`import {createPropertyMixerDispatch as ${create},registerPropertyMixer as ${register},tryPropertyMixer as ${attempt}} from ${JSON.stringify(runtimeModule)};var ${token};\n`,
    tail:`\n${token}=${create}(PropertyMixer,${JSON.stringify(variants)},${maxMemoryPages*65536},${quaternion?'Quaternion,()=>Math':'null,null'});\n`,
    report:{sourceBlob:sourceBlob(source),upstreamCommit:'148ef33ecb6d2502ff796d4554abd1549c95d519',role,
      compiledMethods:['accumulate','accumulateAdditive','apply'].map(name=>'PropertyMixer.'+name),sourceDerived:true,
      quaternionMethods:quaternion?['accumulate','accumulateAdditive','apply']:[],
      scope:quaternion?'numeric-and-quaternion-property-accumulation-and-application':'numeric-property-accumulation-and-application',
      wasmBytes,accelerationClaim:false}};
}
