import assert from 'node:assert/strict';
import test from 'node:test';
import {ANIMATION_ENVIRONMENT_WGSL, animationEnvironmentShader, planAnimationEnvironment} from './animation_environment.mjs';

// Execute the emitted scalar sampling equations on the CPU. Only constructors,
// vector accumulation/division and u32 syntax are translated. This is not a WGSL
// compiler, native float32 execution or a rendered-pixel test.
function scalarKernel() {
  const source=ANIMATION_ENVIRONMENT_WGSL;
  const half=/fn visible_half_vector\([^]*?-> vec3<f32> \{([^]*?)\n\}/.exec(source)?.[1];
  const integrate=/fn integrate_dfg\([^]*?-> vec2<f32> \{([^]*?)\n\}/.exec(source)?.[1];
  assert.ok(half&&integrate);
  const halfBody=half.replace('return vec3<f32>(hx/norm,hy/norm,hz/norm);','return {x:hx/norm,y:hy/norm,z:hz/norm};');
  const body=integrate.replace('var result=vec2<f32>(0.0);','let result=[0,0];')
    .replaceAll('vec2<f32>(1.0-f,f)','[1-f,f]')
    .replace('result+=[1-f,f]*weight;','result[0]+=(1-f)*weight;result[1]+=f*weight;')
    .replace('return result/f32(info.count);','return [result[0]/info.count,result[1]/info.count];')
    .replaceAll('0u','0');
  const prelude=`const PI=Math.PI,{sqrt,max,cos,sin,pow}=Math,clamp=(v,a,b)=>Math.min(b,Math.max(a,v));
    function visible_half_vector(xi,nv,alpha){${halfBody}}`;
  return {
    half:new Function('xi','nv','alpha',prelude+'return visible_half_vector(xi,nv,alpha);'),
    integrate:new Function('nv','roughness','count','sequence',prelude+'const info={count};'+body),
  };
}
const kernel=scalarKernel();
function sequence(i,count) {
  let x=i>>>0,y=0;
  for(let k=0;k<32;k++){y=(y<<1)|(x&1);x>>>=1;}
  return {x:i/count,y:(y>>>0)*2**-32};
}
const dfg=(nv,r,count)=>kernel.integrate(nv,r,count,sequence);
const near=(a,b,t=2e-12)=>assert.ok(Number.isFinite(a)&&Math.abs(a-b)<=t,`${a} != ${b} (+/- ${t})`);

// The old full-NDF importance sampler is an independent comparison baseline,
// not used to compute the reference values or new visible-normal samples.
function fullNdf(nv,r,count) {
  const a2=r**4,vx=Math.sqrt(1-nv*nv),out=[0,0];
  for(let i=0;i<count;i++){
    const u=sequence(i,count),hz=Math.sqrt((1-u.y)/(1+(a2-1)*u.y));
    const hx=Math.cos(2*Math.PI*u.x)*Math.sqrt(1-hz*hz);
    const vh=Math.max(0,vx*hx+nv*hz),nl=2*vh*hz-nv;
    if(nl>0){
      const v=.5/(nl*Math.sqrt(nv*nv*(1-a2)+a2)+nv*Math.sqrt(nl*nl*(1-a2)+a2));
      const w=4*v*nl*vh/hz,f=(1-vh)**5;out[0]+=(1-f)*w;out[1]+=f*w;
    }
  }
  return out.map(x=>x/count);
}

// Independent oracle: Gauss-Legendre integration in cos(theta_L), midpoint
// quadrature in phi_L. Evaluate D*V*NoL directly over incoming solid angle.
// It neither samples half vectors nor uses the estimator's G2/G1 weight.
function gauss(n) {
  const nodes=[];
  for(let i=0;i<n;i++){
    let z=Math.cos(Math.PI*(i+.75)/(n+.5)),derivative;
    for(let iteration=0;iteration<50;iteration++){
      let p0=1,p1=z;
      for(let k=2;k<=n;k++){const p=((2*k-1)*z*p1-(k-1)*p0)/k;p0=p1;p1=p;}
      derivative=n*(z*p1-p0)/(z*z-1);const next=z-p1/derivative;
      if(Math.abs(next-z)<1e-15){z=next;break;}z=next;
    }
    nodes.push([(z+1)/2,1/((1-z*z)*derivative*derivative)]);
  }
  return nodes;
}
const polar=gauss(192),cosines=Float64Array.from({length:8192},(_,i)=>Math.cos(2*Math.PI*(i+.5)/8192));
function quadrature(nv,r) {
  const out=[0,0],a2=r**4,vx=Math.sqrt(1-nv*nv),sv=Math.sqrt(nv*nv*(1-a2)+a2);
  for(const [z,weight]of polar){
    const radius=Math.sqrt(1-z*z),sl=Math.sqrt(z*z*(1-a2)+a2),visibility=.5/(z*sv+nv*sl);
    for(const cosine of cosines){
      const vl=vx*radius*cosine+nv*z,len=Math.sqrt(2+2*vl),hz=(z+nv)/len,vh=(1+vl)/len;
      const distribution=a2/(Math.PI*(1+(a2-1)*hz*hz)**2),f=(1-vh)**5;
      const w=weight*distribution*visibility*z;out[0]+=(1-f)*w;out[1]+=f*w;
    }
  }
  return out.map(x=>x*2*Math.PI/cosines.length);
}
const cases=[...[[.01,.6],[.01,1],[.05,.3],[.05,.6],[.05,1],[.3,.3],[.3,.6],[.3,1],[1,.3],[1,.6],[1,1]]];
const references=new Map();
for(const [nv,r]of cases) test(`visible-normal DFG matches independent solid-angle quadrature: NoV=${nv}, roughness=${r}`,()=>{
  const reference=quadrature(nv,r);references.set(`${nv}:${r}`,reference);
  const actual=dfg(nv,r,4096);
  for(let k=0;k<2;k++)near(actual[k],reference[k],2.5e-4);
});

test('visible half vectors are unit length, upper hemisphere and visible at all practical limits',()=>{
  for(const nv of [0,1/65536,.001,.1,.5,1]) for(const r of [1/65536,.045,.3,1])
    for(let i=0;i<1024;i++){
      const h=kernel.half(sequence(i,1024),nv,r*r);
      near(Math.hypot(h.x,h.y,h.z),1,5e-15);
      assert.ok(h.z>=0&&h.x*Math.sqrt(1-nv*nv)+h.z*nv>=-1e-14);
    }
});

test('smooth-surface limit is analytic, including exact grazing and normal incidence',()=>{
  for(const nv of [0,1/65536,.2,.5,1]) for(const count of [1,17,4096]) {
    const actual=dfg(nv,0,count),f=(1-nv)**5;near(actual[0],1-f);near(actual[1],f);
  }
});

test('DFG energy is finite and bounded even with one sample and grazing directions',()=>{
  for(const nv of [0,1/65536,.001,.01,.1,.5,1]) for(const r of [1/65536,.045,.1,.3,.6,1])
    for(const count of [1,2,3,7,16,128,1024,4096]){
      const [a,b]=dfg(nv,r,count);
      assert.ok(Number.isFinite(a)&&Number.isFinite(b)&&a>=0&&b>=0&&a+b<=1+2e-12,`${nv},${r},${count}: ${a},${b}`);
    }
});

test('128 visible-normal samples reduce aggregate grazing-angle error without increasing the sample budget',t=>{
  let visibleError=0,fullError=0;
  for(const [nv,r]of cases.filter(([nv])=>nv<=.05)){
    const reference=references.get(`${nv}:${r}`)??quadrature(nv,r),v=dfg(nv,r,128),n=fullNdf(nv,r,128);
    visibleError+=Math.hypot(v[0]-reference[0],v[1]-reference[1]);
    fullError+=Math.hypot(n[0]-reference[0],n[1]-reference[1]);
  }
  t.diagnostic(`sum of coefficient-vector errors over five grazing cases: VNDF=${visibleError}, full NDF=${fullError}, ratio=${visibleError/fullError}`);
  assert.ok(visibleError<fullError*.3,`${visibleError} vs ${fullError}`);
});

test('normal-incidence sampling converges to the previous correlated-Smith profile',()=>{
  for(const r of [.045,.2,.6,1]){
    const a=dfg(1,r,4096),b=fullNdf(1,r,4096);
    near(a[0],b[0],1e-12);near(a[1],b[1],1e-12);
  }
});

test('the production filter uses visible normals only for DFG and retains diffuse/specular sampling and budgets',()=>{
  for(const cube of [false,true]){
    const code=animationEnvironmentShader(cube);
    assert.match(code,/let h=visible_half_vector\(sequence\(i,info.count\),nv,alpha\)/);
    assert.match(code,/let weight=nl\*\(nv\+smith_v\)\/\(nl\*smith_v\+nv\*smith_l\)/);
    const fragment=code.slice(code.indexOf('@fragment fn fragment_main'));
    assert.match(fragment,/integrate_dfg\(uv.y,uv.x\)/);
    assert.match(fragment,/frame\*half_vector\(xi,info.roughness\)/);
    assert.match(fragment,/let r=sqrt\(xi.y\)/);
    assert.doesNotMatch(code,/SOURCE_DECLARATION|SOURCE_SAMPLE/);
  }
  const plan=planAnimationEnvironment({size:8,diffuseSize:2,lutSize:4,samples:32});
  assert.equal(plan.passes.length,31);assert.equal(plan.textureBytes,(6*(64+16+4+1)+6*4+16)*8);
  assert.equal(plan.sampleWork,6*64+(6*(16+4+1)+6*4+16)*32);
  assert.equal(plan.uniformBytes,31*256);
});
