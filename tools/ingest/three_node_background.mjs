/** Native source authored-node and PMREM sky backgrounds. The graph and sphere match the pinned
 * r186 common Background; the existing native capture/submit path owns draws.
 */
import {ThreeNodeProgramError} from './three_node_program.mjs';

export function createThreeNodeBackgrounds(three) {
  const N = three.TSL, scenes = new WeakMap(), records = new Set();
  let disposed = false;
  function get(scene) {
    if (disposed) throw new ThreeNodeProgramError('DISPOSED', 'Native node backgrounds are disposed');
    const authored = scene?.isScene && scene.backgroundNode?.isNode ? scene.backgroundNode : null;
    const texture = scene?.isScene ? scene.background : null;
    const panoramic = texture instanceof three.Texture &&
      [three.EquirectangularReflectionMapping, three.EquirectangularRefractionMapping,
        three.CubeUVReflectionMapping].includes(texture.mapping);
    if (!authored && (!panoramic || texture.mapping !== three.CubeUVReflectionMapping && scene.backgroundBlurriness === 0)) return null;
    if (!Number.isFinite(scene.backgroundBlurriness) || scene.backgroundBlurriness < 0 || scene.backgroundBlurriness > 1 ||
        !Number.isFinite(scene.backgroundIntensity) || scene.backgroundIntensity < 0)
      throw new ThreeNodeProgramError('BACKGROUND', 'Native PMREM backgrounds require finite blur in [0,1] and nonnegative intensity');
    // Keep each generator alive while earlier draws still reference its atlas.
    // Resizing one source PMREM generator retires its previous filtering targets.
    const input = authored ?? texture;
    let byInput = scenes.get(scene);
    if (!byInput) { byInput = new WeakMap(); scenes.set(scene, byInput); }
    let record = byInput.get(input);
    if (!record) {
      const pmrem = authored ? null : N.pmremTexture(texture);
      const color = N.vec4(authored ?? pmrem).mul(N.backgroundIntensity).context({
        getUV: () => N.backgroundRotation.mul(N.normalWorldGeometry),
        getTextureLevel: () => N.backgroundBlurriness,
      });
      const projection = N.cameraProjectionMatrix;
      const orthographic = projection.element(3).element(3).equal(1);
      const scale = N.div(1, projection.element(1).element(1)).mul(3);
      const position = orthographic.select(N.positionLocal.mul(scale), N.positionLocal);
      // A zero view-space w removes camera translation; restore w before
      // projection for orthographic cameras and place the sky at the far plane.
      const view = N.modelViewMatrix.mul(N.vec4(position, 0));
      const clip = projection.mul(N.vec4(view.xyz, 1));
      const material = new three.NodeMaterial();
      material.name = 'Background.material'; material.side = three.BackSide;
      material.depthTest = false; material.depthWrite = false;
      material.allowOverride = false; material.fog = false; material.lights = false;
      material.vertexNode = clip.setZ(clip.w); material.colorNode = color;
      const mesh = new three.Mesh(new three.SphereGeometry(1, 32, 32), material);
      mesh.name = 'Background.mesh'; mesh.frustumCulled = false;
      record = {mesh, pmrem, texture}; byInput.set(input, record); records.add(record);
    }
    return record.mesh;
  }
  return Object.freeze({get, dispose() {
    if (disposed) return; disposed = true;
    for (const {mesh, pmrem} of records) {
      mesh.material.dispose(); mesh.geometry.dispose(); pmrem?.dispose();
    }
    records.clear();
  }});
}
