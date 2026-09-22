/** Test-only scalar oracle following the pinned addon's published algorithm.
 * Uses canonical cube coordinates (different indexing from the Wasm emitter).
 * The synthetic topology table exercises every edge/order, not geometric
 * correctness of the Bourke lookup table. Production takes the pinned tables.
 */
const vertices=[[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]];
const edgePairs=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
export function testTables() {
  const edgeTable=new Int32Array(256),triTable=new Int32Array(4096).fill(-1);
  for (let mask=0;mask<256;mask++) {
    const active=[];
    for (let e=0;e<12;e++) {
      const [a,b]=edgePairs[e];
      if (!!(mask & (1<<a))!==!!(mask & (1<<b))) {active.push(e);edgeTable[mask]|=1<<e;}
    }
    const count=Math.ceil(active.length/3)*3;
    for (let j=0;j<count;j++) triTable[mask*16+j]=active[j%active.length];
  }
  return {edgeTable,triTable};
}
export function reference(input,{edgeTable,triTable}) {
  const {size:s,isolation,field,normalCache:nc,palette,positions,normals,uvs,colors,flatShading}=input;
  const half=s/2,delta=2/s,position=input.edgePositions??new Float32Array(36),normal=input.edgeNormals??new Float32Array(36),color=input.edgeColors??new Float32Array(36);
  let count=0;
  function computeNormal(q) {
    if (nc[q*3]===0) {
      nc[q*3]=field[q-1]-field[q+1];
      nc[q*3+1]=field[q-s]-field[q+s];
      nc[q*3+2]=field[q-s*s]-field[q+s*s];
    }
  }
  for (let z=1;z<s-2;z++) for (let y=1;y<s-2;y++) for (let x=1;x<s-2;x++) {
    const q=z*s*s+y*s+x,origin=[(x-half)/half,(y-half)/half,(z-half)/half];
    const indices=vertices.map(([a,b,c])=>q+a+b*s+c*s*s);
    let mask=0;
    for (let k=0;k<8;k++) if (field[indices[k]]<isolation) mask|=1<<k;
    if (!edgeTable[mask]) continue;
    for (let e=0;e<12;e++) {
      if (!(edgeTable[mask] & (1<<e))) continue;
      let [a,b]=edgePairs[e];
      const axis=vertices[a].findIndex((value,j)=>value!==vertices[b][j]);
      if (vertices[a][axis]>vertices[b][axis]) [a,b]=[b,a];
      const lo=indices[a],hi=indices[b];computeNormal(lo);computeNormal(hi);
      const mu=(isolation-field[lo])/(field[hi]-field[lo]);
      for (let j=0;j<3;j++) {
        const start=vertices[a][j] ? origin[j]+delta : origin[j];
        position[e*3+j]=axis===j ? start+mu*delta : start;
        normal[e*3+j]=nc[lo*3+j]+(nc[hi*3+j]-nc[lo*3+j])*mu;
        color[e*3+j]=palette[lo*3+j]+(palette[hi*3+j]-palette[lo*3+j])*mu;
      }
    }
    for (let t=mask*16;triTable[t]!==-1;t+=3) {
      const edges=[triTable[t],triTable[t+1],triTable[t+2]];
      const average=[0,1,2].map(j=>(normal[edges[0]*3+j]+normal[edges[1]*3+j]+normal[edges[2]*3+j])/3);
      for (let v=0;v<3;v++) {
        const edge=edges[v];
        for (let j=0;j<3;j++) {
          positions[(count+v)*3+j]=position[edge*3+j];
          normals[(count+v)*3+j]=flatShading ? average[j] : normal[edge*3+j];
          if (colors) colors[(count+v)*3+j]=color[edge*3+j];
        }
        if (uvs) {uvs[(count+v)*2]=position[edge*3];uvs[(count+v)*2+1]=position[edge*3+2];}
      }
      count+=3;
    }
  }
  return count;
}
export function input(size,capacity=15*Math.max(0,size-3)**3,flags=3,flatShading=false) {
  return {size,isolation:0.1,flatShading,field:new Float32Array(size**3),normalCache:new Float32Array(size**3*3),
    palette:new Float32Array(size**3*3),positions:new Float32Array(capacity*3).fill(123),
    normals:new Float32Array(capacity*3).fill(456),
    uvs:flags&1 ? new Float32Array(capacity*2).fill(789) : null,
    colors:flags&2 ? new Float32Array(capacity*3).fill(1011) : null};
}
export const clone = state => Object.fromEntries(Object.entries(state).map(([key,value])=>
  [key,ArrayBuffer.isView(value) ? value.slice() : value]));
