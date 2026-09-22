/** Source-shaped retained fixture. This is not the full pinned Three.js addon. */
import {testTables} from './reference.mjs';
export const baseSource='class EventDispatcher {}\nexport { EventDispatcher };\n';
export function addonSource(referenceUrl,baseUrl) {
  const {edgeTable,triTable}=testTables();
  return `import { EventDispatcher } from ${JSON.stringify(baseUrl)};
import { reference } from ${JSON.stringify(referenceUrl)};
class Geometry {
  constructor(){this.attributes={};this.drawRange={start:0,count:0};this.onRange=null;}
  getAttribute(name){return this.attributes[name];}
  setDrawRange(start,count){this.drawRange={start,count};this.onRange?.(start,count);}
}
class Material extends EventDispatcher {
  constructor(flat=false){super();this.flatShading=flat;}
}
class MarchingCubes extends EventDispatcher {
  constructor(resolution,material,enableUvs=false,enableColors=false,maxPolyCount=1000) {
    super();
    const scope=this, geometry=new Geometry();
    const vlist=new Float32Array(36),nlist=new Float32Array(36),clist=new Float32Array(36);
    // Test-only inspection; production adds no public edge-list properties.
    this.testEdgeLists=()=>[vlist,nlist,clist];
    this.geometry=geometry;this.material=material;this.enableUvs=enableUvs;this.enableColors=enableColors;
    this.init=function(resolution){
      this.size=resolution;this.size2=resolution*resolution;this.size3=resolution**3;
      this.halfsize=resolution/2;this.delta=2/resolution;this.yd=resolution;this.zd=this.size2;this.isolation=0.1;
      this.field=new Float32Array(this.size3);this.normal_cache=new Float32Array(this.size3*3);
      this.palette=new Float32Array(this.size3*3);this.count=0;
      for(const [key,name,width] of [['positionArray','position',3],['normalArray','normal',3],
        ...this.enableUvs ? [['uvArray','uv',2]] : [],...this.enableColors ? [['colorArray','color',3]] : []]) {
        this[key]=new Float32Array(maxPolyCount*3*width);
        geometry.attributes[name]={array:this[key],version:0,set needsUpdate(value){if(value===true)this.version++;}};
      }
    };
    this.update=function(){
      this.count=0;
      this.count=reference({size:this.size,isolation:this.isolation,field:scope.field,normalCache:scope.normal_cache,
        palette:scope.palette,positions:scope.positionArray,normals:scope.normalArray,
        edgePositions:vlist,edgeNormals:nlist,edgeColors:clist,
        uvs:scope.enableUvs ? scope.uvArray : null,colors:scope.enableColors ? scope.colorArray : null,
        flatShading:scope.material.flatShading===true},{edgeTable,triTable});
      this.geometry.setDrawRange(0,this.count);
      geometry.getAttribute('position').needsUpdate=true;
      geometry.getAttribute('normal').needsUpdate=true;
      if(this.enableUvs)geometry.getAttribute('uv').needsUpdate=true;
      if(this.enableColors)geometry.getAttribute('color').needsUpdate=true;
      if(this.count/3>maxPolyCount)console.warn('capacity');
    };
    this.init(resolution);
  }
}
const edgeTable=new Int32Array(${JSON.stringify([...edgeTable])});
const triTable=new Int32Array(${JSON.stringify([...triTable])});
export { MarchingCubes, Material, Geometry, edgeTable, triTable };
`;
}
