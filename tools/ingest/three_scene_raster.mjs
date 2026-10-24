/** The native r186 material raster mapping shared by fixed and TSL programs. */
import {threeProgramStencil} from './three_program_stencil.mjs';
const FACTORS=[['ZeroFactor','zero'],['OneFactor','one'],['SrcColorFactor','src'],['OneMinusSrcColorFactor','one-minus-src'],
  ['SrcAlphaFactor','src-alpha'],['OneMinusSrcAlphaFactor','one-minus-src-alpha'],['DstColorFactor','dst'],['OneMinusDstColorFactor','one-minus-dst'],
  ['DstAlphaFactor','dst-alpha'],['OneMinusDstAlphaFactor','one-minus-dst-alpha'],['SrcAlphaSaturateFactor','src-alpha-saturated'],
  ['ConstantColorFactor','constant'],['ConstantAlphaFactor','constant'],['OneMinusConstantColorFactor','one-minus-constant'],['OneMinusConstantAlphaFactor','one-minus-constant']];
const EQUATIONS=[['AddEquation','add'],['SubtractEquation','subtract'],['ReverseSubtractEquation','reverse-subtract'],['MinEquation','min'],['MaxEquation','max']];
const DEPTH=['never','always','less','less-equal','equal','greater-equal','greater','not-equal'];
export function createThreeSceneRaster(three,{stencilAttachment=false,fail}={}){
  const factors=new Map(FACTORS.map(([k,v])=>[three[k],v])),equations=new Map(EQUATIONS.map(([k,v])=>[three[k],v]));
  const finite=(n,key)=>{if(typeof n!=='number'||!Number.isFinite(n))fail('VALUE',`Invalid ${key}`);return n;};
  function describe(m,topology='triangles'){
    const options={},values={};
    const blended=m.blending!==three.NoBlending&&(m.blending!==three.NormalBlending||m.transparent!==false);
    const premultiplied=m.premultipliedAlpha===true;
    if(m.premultipliedAlpha!==undefined&&typeof m.premultipliedAlpha!=='boolean')fail('MATERIAL','Expected boolean premultipliedAlpha');
    const set=(src,dst,srcAlpha,dstAlpha)=>({color:{operation:'add',srcFactor:src,dstFactor:dst},alpha:{operation:'add',srcFactor:srcAlpha,dstFactor:dstAlpha}});
    if(!blended)options.blend=null;
    else if(m.blending===three.CustomBlending){
      const factor=f=>{const name=factors.get(f);if(!name)fail('MATERIAL','Unsupported source blend factor');return name;};
      const equation=e=>{const name=equations.get(e);if(!name)fail('MATERIAL','Unsupported source blend equation');return name;};
      options.blend={color:{operation:equation(m.blendEquation),srcFactor:factor(m.blendSrc),dstFactor:factor(m.blendDst)},
        alpha:{operation:equation(m.blendEquationAlpha??m.blendEquation),srcFactor:factor(m.blendSrcAlpha??m.blendSrc),dstFactor:factor(m.blendDstAlpha??m.blendDst)}};
      if([options.blend.color,options.blend.alpha].some(c=>[c.srcFactor,c.dstFactor].some(f=>f.includes('constant'))))
        values.blendConstant=[m.blendColor.r,m.blendColor.g,m.blendColor.b,m.blendAlpha];
    }else if(m.blending===three.NormalBlending)options.blend=premultiplied?set('one','one-minus-src-alpha','one','one-minus-src-alpha'):set('src-alpha','one-minus-src-alpha','one','one-minus-src-alpha');
    else if(m.blending===three.AdditiveBlending)options.blend=premultiplied?set('one','one','one','one'):set('src-alpha','one','one','one');
    else if(m.blending===three.SubtractiveBlending||m.blending===three.MultiplyBlending){
      if(premultiplied)options.blend=m.blending===three.SubtractiveBlending?set('zero','one-minus-src','zero','one'):set('dst','one-minus-src-alpha','zero','one');
      else{
        (three.error??console.error)(`WebGPURenderer: "${m.blending===three.SubtractiveBlending?'SubtractiveBlending':'MultiplyBlending'}" requires "material.premultipliedAlpha = true".`);
        options.blend=null;
      }
    }else fail('MATERIAL','Unsupported source blending mode');
    if(premultiplied)options.premultipliedAlpha=true;
    if(m.polygonOffset===true&&topology==='triangles'){
      for(const key of ['polygonOffsetUnits','polygonOffsetFactor'])finite(m[key],key);
      if(!Number.isSafeInteger(m.polygonOffsetUnits))fail('MATERIAL','polygonOffsetUnits must be an integer depth bias');
      Object.assign(options,{depthBias:m.polygonOffsetUnits,depthBiasSlopeScale:m.polygonOffsetFactor,depthBiasClamp:0});
    }
    // The core renderer supplies ordinary straight-alpha NormalBlending.
    if(m.blending===three.NormalBlending&&!premultiplied)delete options.blend;
    else if(m.blending===three.NormalBlending&&!m.transparent)options.blend=null;
    if(stencilAttachment){
      const {stencil,stencilReference}=threeProgramStencil(three,m,fail,{webgpu:true});
      if(stencil){options.stencil=stencil;values.stencilReference=stencilReference;}
    }
    return {options,values,key:JSON.stringify(options)};
  }
  // Native programs need the complete pipeline state rather than the core's
  // compact material defaults. Both routes use the same blend/bias/stencil map.
  describe.program=(m,{mrt=null,textures=null}={})=>{
    if(![three.FrontSide,three.BackSide,three.DoubleSide].includes(m.side)||!DEPTH[m.depthFunc])
      fail('MATERIAL','Unsupported native side/depth state');
    for(const key of ['transparent','depthWrite','depthTest','colorWrite'])
      // Renderer.renderObject's inherited OR expression ends in a nullable
      // node slot for opaque shadow casters. Preserve that exact source field;
      // raster selection interprets it as falsy without changing the material.
      if(typeof m[key]!=='boolean'&&!(key==='transparent'&&m.isShadowPassMaterial===true&&m[key]==null))
        fail('MATERIAL',`Expected boolean ${key}`);
    const {options,values}=describe(m),normal={color:{operation:'add',srcFactor:'src-alpha',dstFactor:'one-minus-src-alpha'},
      alpha:{operation:'add',srcFactor:'one',dstFactor:'one-minus-src-alpha'}};
    const state={...options,...values,blend:options.blend===undefined?(m.transparent?normal:null):options.blend,
      writeMask:m.colorWrite?15:0,depthWriteEnabled:m.depthWrite,depthCompare:m.depthTest?DEPTH[m.depthFunc]:'always',
      frontFace:m.side===three.BackSide?'cw':'ccw',cullMode:m.side===three.DoubleSide?'none':'back'};
    if(mrt&&textures){
      // The source render context controls attachment blending; material MRT
      // nodes override shader outputs without replacing that context.
      state.colorBlends=textures.map(texture=>{
        const mode=mrt.getBlendMode(texture.name);
        if(mode.blending===three.MaterialBlending)return state.blend;
        const mapped=describe({...m,...mode,transparent:true,premultipliedAlpha:mode.premultipliedAlpha===true});
        return mapped.options.blend===undefined?normal:mapped.options.blend;
      });
      // WebGPUBackend advances this pass state for CustomBlending materials
      // only. An auxiliary constant-factor output on any other material uses
      // the preceding draw's constant, or the pass's initial zero value.
      state.blendConstant=m.blending===three.CustomBlending
        ?[m.blendColor.r,m.blendColor.g,m.blendColor.b,m.blendAlpha]:null;
    }
    return state;
  };
  return describe;
}
