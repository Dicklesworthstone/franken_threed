// Deterministic source scenes shared by the routed WebGPURenderer build and the
// pinned upstream WebGLRenderer reference. Each receives the page's own THREE
// namespace; nothing here depends on which renderer executes it.
export const scenarios = {
  standard_lights(THREE) {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x202830);
    const sphere = new THREE.Mesh(new THREE.SphereGeometry(0.8, 48, 24), new THREE.MeshStandardMaterial({ color: 0xcc6633, roughness: 0.4, metalness: 0.2 }));
    sphere.position.x = -0.9;
    const box = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshStandardMaterial({ color: 0x3399cc, roughness: 0.8 }));
    box.position.x = 1; box.rotation.set(0.5, 0.7, 0.1);
    const sun = new THREE.DirectionalLight(0xffffff, 2.5); sun.position.set(3, 4, 5);
    const point = new THREE.PointLight(0xffaa66, 8, 0, 2); point.position.set(-1, 1.5, 2);
    scene.add(sphere, box, sun, point, new THREE.AmbientLight(0x404050, 1));
    return { scene, camera: camera(THREE) };
  },
  phong_lambert_basic(THREE) {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x101010);
    const geo = new THREE.TorusKnotGeometry(0.35, 0.12, 96, 12);
    const mats = [new THREE.MeshPhongMaterial({ color: 0x44aa88, shininess: 60, specular: 0x444444 }),
      new THREE.MeshLambertMaterial({ color: 0xaa4488 }), new THREE.MeshBasicMaterial({ color: 0x8888ff })];
    mats.forEach((m, i) => { const mesh = new THREE.Mesh(geo, m); mesh.position.x = (i - 1) * 1.1; scene.add(mesh); });
    const sun = new THREE.DirectionalLight(0xffffff, 2); sun.position.set(1, 2, 3);
    scene.add(sun, new THREE.HemisphereLight(0x8899ff, 0x443322, 1));
    return { scene, camera: camera(THREE) };
  },
  spot_and_toon(THREE) {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(6, 6), new THREE.MeshStandardMaterial({ color: 0xffffff }));
    floor.rotation.x = -Math.PI / 2; floor.position.y = -0.8;
    const toon = new THREE.Mesh(new THREE.SphereGeometry(0.6, 32, 16), new THREE.MeshToonMaterial({ color: 0x33cc66 }));
    const spot = new THREE.SpotLight(0xffffff, 30, 0, Math.PI / 6, 0.3, 2); spot.position.set(0, 3, 1);
    scene.add(floor, toon, spot, spot.target, new THREE.AmbientLight(0xffffff, 0.2));
    return { scene, camera: camera(THREE, [0, 1.2, 3.5]) };
  },
  transparency_and_sides(THREE) {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x335577);
    const back = new THREE.Mesh(new THREE.PlaneGeometry(3, 2), new THREE.MeshBasicMaterial({ color: 0xffcc00 }));
    back.position.z = -0.5;
    const glass = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshLambertMaterial({ color: 0xff3333, transparent: true, opacity: 0.5, side: THREE.DoubleSide }));
    glass.rotation.set(0.4, 0.6, 0);
    const inside = new THREE.Mesh(new THREE.SphereGeometry(0.6, 32, 16), new THREE.MeshBasicMaterial({ color: 0x00ff88, side: THREE.BackSide }));
    inside.position.x = 1.3;
    scene.add(back, glass, inside, new THREE.AmbientLight(0xffffff, 3));
    return { scene, camera: camera(THREE) };
  },
  fog_and_instancing(THREE) {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x99aabb);
    scene.fog = new THREE.Fog(0x99aabb, 3, 9);
    const mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(0.4, 0.4, 0.4), new THREE.MeshLambertMaterial({ color: 0xffffff }), 25);
    const m = new THREE.Matrix4(), c = new THREE.Color();
    for (let i = 0; i < 25; i++) {
      m.makeTranslation((i % 5) - 2, -0.5, -Math.floor(i / 5) * 1.5);
      mesh.setMatrixAt(i, m); mesh.setColorAt(i, c.setHSL(i / 25, 0.7, 0.5));
    }
    const sun = new THREE.DirectionalLight(0xffffff, 2); sun.position.set(1, 3, 2);
    scene.add(mesh, sun, new THREE.AmbientLight(0xffffff, 0.5));
    return { scene, camera: camera(THREE, [0, 1, 3]) };
  },
  textured(THREE) {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x222222);
    const size = 8, data = new Uint8Array(size * size * 4);
    for (let i = 0; i < size * size; i++) { const on = ((i % size) + Math.floor(i / size)) % 2; data.set(on ? [255, 220, 40, 255] : [30, 60, 200, 255], i * 4); }
    const tex = new THREE.DataTexture(data, size, size);
    tex.colorSpace = THREE.SRGBColorSpace; tex.needsUpdate = true;
    tex.magFilter = THREE.NearestFilter;
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(2.4, 2.4), new THREE.MeshBasicMaterial({ map: tex }));
    plane.rotation.z = 0.3;
    scene.add(plane);
    return { scene, camera: camera(THREE) };
  },
  shadows(THREE, renderer) {
    renderer.shadowMap.enabled = true;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111122);
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(6, 6), new THREE.MeshStandardMaterial({ color: 0xdddddd }));
    floor.rotation.x = -Math.PI / 2; floor.position.y = -0.7; floor.receiveShadow = true;
    const box = new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.8, 0.8), new THREE.MeshStandardMaterial({ color: 0xff8844 }));
    box.castShadow = true; box.rotation.y = 0.6;
    const sun = new THREE.DirectionalLight(0xffffff, 3); sun.position.set(2, 4, 1); sun.castShadow = true;
    sun.shadow.mapSize.set(1024, 1024);
    scene.add(floor, box, sun, new THREE.AmbientLight(0xffffff, 0.3));
    return { scene, camera: camera(THREE, [0, 1.5, 3.5]) };
  },
  soft_shadows(THREE, renderer) {
    renderer.shadowMap.enabled = true;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x222233);
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(8, 8), new THREE.MeshPhongMaterial({ color: 0xbbbbbb }));
    floor.rotation.x = -Math.PI / 2; floor.position.y = -0.7; floor.receiveShadow = true;
    const knot = new THREE.Mesh(new THREE.TorusKnotGeometry(0.4, 0.13, 96, 12), new THREE.MeshStandardMaterial({ color: 0x44aaff, roughness: 0.5 }));
    knot.castShadow = true; knot.receiveShadow = true;
    const spot = new THREE.SpotLight(0xffffff, 40, 0, Math.PI / 5, 0.3, 2);
    spot.position.set(1.5, 3, 1); spot.castShadow = true; spot.shadow.radius = 4; spot.shadow.mapSize.set(512, 512);
    scene.add(floor, knot, spot, spot.target, new THREE.AmbientLight(0xffffff, 0.25));
    return { scene, camera: camera(THREE, [0, 1.6, 3.4]) };
  },
  helpers_lines_points(THREE) {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);
    scene.add(new THREE.GridHelper(4, 8, 0xff0000, 0x00ff00), new THREE.AxesHelper(1.5));
    const pts = new THREE.Points(new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute([0, 1, 0, 0.5, 1, 0, -0.5, 1, 0], 3)), new THREE.PointsMaterial({ color: 0xffffff, size: 6, sizeAttenuation: false }));
    scene.add(pts);
    return { scene, camera: camera(THREE, [2, 2, 3]) };
  },
  line_strips_and_loops(THREE) {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x101820);
    const pts = [];
    for (let i = 0; i <= 64; i++) pts.push(new THREE.Vector3(Math.cos(i / 8) * 1.2, (i / 64) * 2 - 1, Math.sin(i / 8) * 0.5));
    const spiral = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: 0xffcc00 }));
    const indexed = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-1.5, -0.8, 0), new THREE.Vector3(-0.5, 0.9, 0), new THREE.Vector3(0.5, -0.9, 0), new THREE.Vector3(1.5, 0.8, 0)]);
    indexed.setIndex([0, 1, 2, 3, 1]);
    const zigzag = new THREE.Line(indexed, new THREE.LineBasicMaterial({ color: 0x33ddff, transparent: true, opacity: 0.6 }));
    const loop = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-1, -1, 0.5), new THREE.Vector3(1, -1, 0.5), new THREE.Vector3(0, 1, 0.5)]), new THREE.LineBasicMaterial({ color: 0xff0000 }));
    scene.add(spiral, zigzag, loop);
    return { scene, camera: camera(THREE) };
  },
  blending_modes(THREE) {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x406080);
    const geo = new THREE.PlaneGeometry(0.9, 0.9);
    const quad = (x, y, color, options) => {
      const m = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color, ...options }));
      m.position.set(x, y, 0); scene.add(m); return m;
    };
    quad(-1.2, 0.5, 0xff8800, { blending: THREE.AdditiveBlending, transparent: true, opacity: 0.7, depthWrite: false });
    quad(-0.6, 0.5, 0x0088ff, { blending: THREE.AdditiveBlending, opacity: 0.5 }); // additive without transparent
    quad(0.0, 0.5, 0x88ff00, { blending: THREE.SubtractiveBlending, premultipliedAlpha: true, transparent: true, opacity: 0.8 });
    quad(0.6, 0.5, 0xff00ff, { blending: THREE.MultiplyBlending, premultipliedAlpha: true, transparent: true, opacity: 0.9 });
    quad(1.2, 0.5, 0xffffff, { blending: THREE.NormalBlending, premultipliedAlpha: true, transparent: true, opacity: 0.4 });
    // Upstream WebGPU passes non-one factors with min/max, which WebGPU rejects
    // (the object is not drawn); F3D keeps the GL meaning. Use valid factors here.
    quad(-1.2, -0.5, 0x00ffcc, { blending: THREE.CustomBlending, blendEquation: THREE.MaxEquation, blendSrc: THREE.OneFactor, blendDst: THREE.OneFactor, transparent: true });
    quad(-0.6, -0.5, 0xffff00, { blending: THREE.CustomBlending, blendSrc: THREE.ConstantColorFactor, blendDst: THREE.OneMinusConstantColorFactor,
      blendColor: new THREE.Color(0.25, 0.5, 0.75), transparent: true });
    quad(0.0, -0.5, 0xff0000, { blending: THREE.NoBlending, transparent: true, opacity: 0.3 });
    // Coplanar decal resolved by polygon offset.
    const base = quad(0.9, -0.5, 0x222222, {});
    base.scale.set(1.6, 1, 1);
    quad(0.9, -0.5, 0xffffff, { polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -4 }).scale.set(0.5, 0.5, 1);
    return { scene, camera: camera(THREE) };
  },
  normal_material(THREE) {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);
    const knot = new THREE.Mesh(new THREE.TorusKnotGeometry(0.45, 0.15, 96, 16), new THREE.MeshNormalMaterial());
    knot.position.x = -0.8; knot.rotation.set(0.3, 0.5, 0);
    const flat = new THREE.Mesh(new THREE.IcosahedronGeometry(0.6, 0), new THREE.MeshNormalMaterial({ flatShading: true }));
    flat.position.x = 0.9; flat.rotation.y = 0.4;
    scene.add(knot, flat);
    return { scene, camera: camera(THREE, [0.3, 0.4, 3.2]) };
  },
  physical_neutral(THREE) {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x202020);
    const a = new THREE.Mesh(new THREE.SphereGeometry(0.7, 48, 24), new THREE.MeshPhysicalMaterial({ color: 0xcc3355, roughness: 0.35, metalness: 0.1 }));
    a.position.x = -0.8;
    const b = new THREE.Mesh(new THREE.TorusGeometry(0.45, 0.18, 24, 48), new THREE.MeshPhysicalMaterial({ color: 0xaaaaaa, roughness: 0.2, metalness: 0.9 }));
    b.position.x = 0.9;
    const sun = new THREE.DirectionalLight(0xffffff, 3); sun.position.set(2, 3, 4);
    const fill = new THREE.PointLight(0x88aaff, 6, 0, 2); fill.position.set(-2, -1, 2);
    scene.add(a, b, sun, fill, new THREE.HemisphereLight(0xffffff, 0x334455, 0.6));
    return { scene, camera: camera(THREE) };
  },
  wireframes(THREE) {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x101010);
    const lit = new THREE.Mesh(new THREE.SphereGeometry(0.7, 16, 12), new THREE.MeshStandardMaterial({ color: 0x66ccff, wireframe: true }));
    lit.position.x = -0.9;
    const basic = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1).toNonIndexed(), new THREE.MeshBasicMaterial({ color: 0xffaa00, wireframe: true }));
    basic.position.x = 0.9; basic.rotation.set(0.4, 0.6, 0);
    const sun = new THREE.DirectionalLight(0xffffff, 3); sun.position.set(1, 2, 3);
    scene.add(lit, basic, sun, new THREE.AmbientLight(0xffffff, 0.3));
    return { scene, camera: camera(THREE) };
  },
  // WebGPU-build node-material classes with no assigned nodes (H1 uses MeshToonNodeMaterial).
  node_materials(THREE) {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xc1c1c1);
    const kinds = ['MeshToonNodeMaterial', 'MeshStandardNodeMaterial', 'MeshPhongNodeMaterial', 'MeshLambertNodeMaterial', 'MeshBasicNodeMaterial'];
    kinds.forEach((kind, i) => {
      const mesh = new THREE.Mesh(new THREE.IcosahedronGeometry(0.4, 1), new THREE[kind]({ color: new THREE.Color().setHSL(i / 5, 0.6, 0.5) }));
      mesh.position.x = (i - 2) * 0.85;
      scene.add(mesh);
    });
    const light = new THREE.DirectionalLight(0xffffff, 3.4); light.position.set(1, 1, 1);
    const group = new THREE.BundleGroup();
    const lines = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(3.6, 0.2, 0.2)), new THREE.LineBasicNodeMaterial({ color: 0x000000 }));
    lines.position.y = -0.8; group.add(lines);
    scene.add(light, group);
    return { scene, camera: camera(THREE) };
  },
};
function camera(THREE, position = [0, 0, 3.2]) {
  const c = new THREE.PerspectiveCamera(50, 320 / 240, 0.1, 50);
  c.position.set(...position); c.lookAt(0, 0, 0);
  return c;
}
