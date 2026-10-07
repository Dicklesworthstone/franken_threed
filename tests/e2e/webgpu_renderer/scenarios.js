// Deterministic source scenes shared by the routed WebGPURenderer build and the
// pinned upstream WebGLRenderer reference. Each receives the page's own THREE
// namespace; nothing here depends on which renderer executes it.
import { RoomEnvironment } from '../../../upstream/three.js/examples/jsm/environments/RoomEnvironment.js';
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
  room_environment(THREE, renderer) {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x303030);
    const pmrem = new THREE.PMREMGenerator(renderer);
    scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    const metal = new THREE.Mesh(new THREE.SphereGeometry(0.6, 48, 24), new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 1, roughness: 0.3 }));
    metal.position.x = -0.75;
    const plastic = new THREE.Mesh(new THREE.TorusKnotGeometry(0.35, 0.12, 96, 16), new THREE.MeshStandardMaterial({ color: 0xcc4422, roughness: 0.6 }));
    plastic.position.x = 0.8;
    scene.add(metal, plastic);
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
  // r186 SunLight (examples/jsm/lights): two cascades refit from the camera, with
  // the near/far fade band, over a long ground plane receding from the viewer.
  sun_cascades(THREE, renderer) {
    if (renderer.library && THREE.SunLightNode) renderer.library.addLight(THREE.SunLightNode, THREE.SunLight);
    renderer.shadowMap.enabled = true;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x8899aa);
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(60, 60), new THREE.MeshStandardMaterial({ color: 0xdddddd }));
    ground.rotation.x = -Math.PI / 2; ground.receiveShadow = true;
    scene.add(ground);
    for (let i = 0; i < 8; i++) {
      const box = new THREE.Mesh(new THREE.BoxGeometry(0.6, 1.5, 0.6), new THREE.MeshStandardMaterial({ color: new THREE.Color().setHSL(i / 8, 0.5, 0.5) }));
      box.position.set((i % 2 ? 1 : -1) * 1.2, 0.75, 1 - i * 3.5); box.castShadow = true; box.receiveShadow = true;
      scene.add(box);
    }
    const sun = new THREE.SunLight(0xffffff, 3);
    sun.position.set(-3, 6, 2); sun.castShadow = true; sun.shadow.mapSize.set(512, 512);
    scene.add(sun, new THREE.HemisphereLight(0xffffff, 0x444444, 0.5));
    const cam = new THREE.PerspectiveCamera(55, 320 / 240, 0.1, 60);
    cam.position.set(0, 3, 4); cam.lookAt(0, 0, -6);
    return { scene, camera: cam };
  },
  // ShaderMaterial / RawShaderMaterial (WebGLRenderer feature): uniforms of several
  // types, a texture, a custom attribute, varyings, gl_FragCoord, discard,
  // additive blending, and an ESSL 1.00 raw program.
  shader_material(THREE) {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x101820);
    const size = 8, data = new Uint8Array(size * size * 4);
    for (let i = 0; i < size * size; i++) data.set([(i * 37) % 256, (i * 91) % 256, ((i >> 3) * 60) % 256, 255], i * 4);
    const tex = new THREE.DataTexture(data, size, size); tex.needsUpdate = true;
    const geometry = new THREE.TorusKnotGeometry(0.45, 0.16, 96, 12);
    const wobble = new Float32Array(geometry.attributes.position.count);
    for (let i = 0; i < wobble.length; i++) wobble[i] = Math.sin(i * 0.7) * 0.5 + 0.5;
    geometry.setAttribute('wobble', new THREE.BufferAttribute(wobble, 1));
    const knot = new THREE.Mesh(geometry, new THREE.ShaderMaterial({
      uniforms: { tint: { value: new THREE.Color(0.9, 0.5, 0.2) }, amount: { value: 0.08 }, map: { value: tex }, scales: { value: [1, 0.5, 0.25] }, flip: { value: true } },
      vertexShader: `attribute float wobble; uniform float amount; varying vec2 vUv; varying float vW; varying vec3 vN;
        void main() { vUv = uv; vW = wobble; vN = normalize(normalMatrix * normal);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position + normal * wobble * amount, 1.0); }`,
      fragmentShader: `uniform vec3 tint; uniform sampler2D map; uniform float scales[3]; uniform bool flip; varying vec2 vUv; varying float vW; varying vec3 vN;
        void main() { vec4 t = texture2D(map, vUv * 4.0); float l = 0.4 + 0.6 * max(dot(vN, normalize(vec3(0.3, 0.6, 1.0))), 0.0);
          vec3 c = mix(tint, t.rgb, scales[1]) * l * (flip ? 1.0 : 0.2);
          if (mod(floor(gl_FragCoord.x / 4.0) + floor(gl_FragCoord.y / 4.0), 2.0) < 1.0 && vW > 0.9) discard;
          gl_FragColor = vec4(c, 1.0); }`,
    }));
    knot.position.x = -0.6; knot.rotation.set(0.4, 0.3, 0);
    const quad = new THREE.Mesh(new THREE.PlaneGeometry(1.1, 1.1), new THREE.RawShaderMaterial({
      uniforms: { glow: { value: new THREE.Vector4(0.2, 0.8, 0.4, 0.6) } },
      vertexShader: `precision highp float; uniform mat4 modelViewMatrix; uniform mat4 projectionMatrix; attribute vec3 position; attribute vec2 uv; varying vec2 vUv;
        void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
      fragmentShader: `precision mediump float; uniform vec4 glow; varying vec2 vUv;
        void main() { float d = length(vUv - 0.5); gl_FragColor = vec4(glow.rgb * (1.0 - 2.0 * d), glow.a * smoothstep(0.5, 0.1, d)); }`,
      transparent: true, blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    quad.position.set(0.55, 0.1, 0.3);
    scene.add(knot, quad);
    return { scene, camera: camera(THREE) };
  },
  // ShaderMaterial points: gl_PointSize squares, gl_PointCoord (GL upper-left
  // origin) with discard, per-point size attribute, and a clipped-center point.
  shader_points(THREE) {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a14);
    const n = 40, pos = new Float32Array(n * 3), size = new Float32Array(n), col = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const a = i / n * Math.PI * 2;
      pos.set([Math.cos(a) * (0.4 + 0.03 * i), Math.sin(a) * 0.8, (i % 5) * 0.1], i * 3);
      size[i] = 4 + (i % 7) * 5; col.set([(i % 3) / 2, ((i + 1) % 4) / 3, 1 - i / n], i * 3);
    }
    pos.set([1.9, 0, 0], 0); // center outside the frustum edge: GL drops the whole point
    const g = new THREE.BufferGeometry().setAttribute('position', new THREE.BufferAttribute(pos, 3))
      .setAttribute('size', new THREE.BufferAttribute(size, 1)).setAttribute('customColor', new THREE.BufferAttribute(col, 3));
    const points = new THREE.Points(g, new THREE.ShaderMaterial({
      vertexShader: `attribute float size; attribute vec3 customColor; varying vec3 vColor;
        void main() { vColor = customColor; gl_PointSize = size; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
      fragmentShader: `varying vec3 vColor;
        void main() { vec2 d = gl_PointCoord - vec2(0.5); if (dot(d, d) > 0.25) discard; gl_FragColor = vec4(vColor * (0.6 + gl_PointCoord.y * 0.4), 1.0); }`,
    }));
    scene.add(points);
    return { scene, camera: camera(THREE) };
  },
  // Built-in materials the WebGL surface draws with ShaderLib programs: cube
  // envMap reflection/refraction with each combine op, lightMap, bumpMap.
  shaderlib_maps(THREE) {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x202020);
    const face = (r, g, b) => {
      const n = 8, d = new Uint8Array(n * n * 4);
      for (let i = 0; i < n * n; i++) { const k = ((i % n) + (i >> 3)) % 2 ? 1 : 0.55; d.set([r * k, g * k, b * k, 255], i * 4); }
      const t = new THREE.DataTexture(d, n, n); t.needsUpdate = true; return t;
    };
    const cube = new THREE.CubeTexture([face(255, 60, 60), face(60, 255, 60), face(60, 60, 255), face(255, 255, 60), face(60, 255, 255), face(255, 60, 255)]);
    cube.needsUpdate = true;
    const refract = cube.clone(); refract.mapping = THREE.CubeRefractionMapping; refract.needsUpdate = true;
    const lightTex = face(255, 200, 120); lightTex.channel = 0;
    const bump = new THREE.DataTexture(Uint8Array.from({ length: 16 * 16 * 4 }, (_, i) => (i % 4 === 3 ? 255 : ((i >> 2) % 16 < 8 ? 40 : 220))), 16, 16); bump.needsUpdate = true;
    const mats = [
      new THREE.MeshBasicMaterial({ color: 0xffffff, envMap: cube }),
      new THREE.MeshLambertMaterial({ color: 0xdddddd, envMap: cube, combine: THREE.MixOperation, reflectivity: 0.6 }),
      new THREE.MeshPhongMaterial({ color: 0xffffff, envMap: refract, refractionRatio: 0.85, combine: THREE.AddOperation, reflectivity: 0.4, shininess: 40 }),
      new THREE.MeshLambertMaterial({ color: 0xffffff, lightMap: lightTex, lightMapIntensity: 1.5 }),
      new THREE.MeshPhongMaterial({ color: 0x88aaff, bumpMap: bump, bumpScale: 4 }),
    ];
    mats.forEach((m, i) => {
      const mesh = new THREE.Mesh(new THREE.SphereGeometry(0.42, 32, 16), m);
      mesh.position.set((i % 3 - 1) * 1.0, i < 3 ? 0.45 : -0.5, 0); mesh.rotation.y = 0.5 + i;
      scene.add(mesh);
    });
    const light = new THREE.DirectionalLight(0xffffff, 2); light.position.set(1, 2, 3);
    scene.add(light, new THREE.AmbientLight(0x404040, 1), new THREE.PointLight(0xff8844, 4, 0, 2));
    return { scene, camera: camera(THREE, [0, 0, 3.4]) };
  },
  // r186 PMREM (PMREMGenerator cube-UV, GGX mips) for MeshStandard/Physical
  // envMaps through ShaderLib programs: a cube source and an equirect source.
  shaderlib_pmrem(THREE) {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x101010);
    const face = (r, g, b) => {
      const n = 64, d = new Uint8Array(n * n * 4);
      for (let i = 0; i < n * n; i++) {
        const x = i % n, y = i >> 6, k = ((x >> 3) + (y >> 3)) % 2 ? 1 : 0.35, h = 0.5 + 0.5 * (y / n);
        d.set([r * k * h, g * k * h, b * k * h, 255], i * 4);
      }
      const t = new THREE.DataTexture(d, n, n); t.needsUpdate = true; return t;
    };
    const cube = new THREE.CubeTexture([face(255, 80, 80), face(80, 255, 80), face(80, 80, 255), face(255, 255, 80), face(80, 255, 255), face(255, 80, 255)]);
    cube.needsUpdate = true;
    const w = 128, h = 64, pano = new Uint8Array(w * h * 4);
    for (let i = 0; i < w * h; i++) { const x = i % w, y = (i / w) | 0; pano.set([(x * 2) & 255, y * 4, ((x >> 4) + (y >> 4)) % 2 ? 230 : 40, 255], i * 4); }
    const equirect = new THREE.DataTexture(pano, w, h); equirect.mapping = THREE.EquirectangularReflectionMapping; equirect.needsUpdate = true;
    const mats = [
      new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 1, roughness: 0.05, envMap: cube }),
      new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 1, roughness: 0.45, envMap: cube }),
      new THREE.MeshStandardMaterial({ color: 0xcc8844, metalness: 0.3, roughness: 0.8, envMap: cube, envMapIntensity: 1.5 }),
      new THREE.MeshPhysicalMaterial({ color: 0xffffff, metalness: 1, roughness: 0.2, envMap: equirect, clearcoat: 1 }),
    ];
    mats.forEach((m, i) => {
      const mesh = new THREE.Mesh(new THREE.SphereGeometry(0.42, 48, 24), m);
      mesh.position.set((i % 2) - 0.5, i < 2 ? 0.48 : -0.48, 0);
      scene.add(mesh);
    });
    const light = new THREE.DirectionalLight(0xffffff, 1); light.position.set(1, 2, 3);
    scene.add(light);
    return { scene, camera: camera(THREE, [0, 0, 3.2]) };
  },
  // r186 WebGLShadowMap on the program route: three shadow-casting lights
  // (directional, spot, point cube map), Basic-type-free PCF receivers.
  shaderlib_shadows(THREE, renderer) {
    renderer.shadowMap.enabled = true;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x101018);
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(8, 8), new THREE.MeshPhongMaterial({ color: 0xcccccc }));
    floor.rotation.x = -Math.PI / 2; floor.position.y = -0.7; floor.receiveShadow = true;
    const wall = new THREE.Mesh(new THREE.PlaneGeometry(8, 4), new THREE.MeshStandardMaterial({ color: 0x8899aa, roughness: 0.9 }));
    wall.position.set(0, 1.3, -1.6); wall.receiveShadow = true;
    const box = new THREE.Mesh(new THREE.BoxGeometry(0.6, 0.6, 0.6), new THREE.MeshLambertMaterial({ color: 0xff8844 }));
    box.position.set(-0.7, -0.2, 0); box.rotation.y = 0.5; box.castShadow = true; box.receiveShadow = true;
    const knot = new THREE.Mesh(new THREE.TorusKnotGeometry(0.3, 0.1, 80, 10), new THREE.MeshStandardMaterial({ color: 0x44aaff, roughness: 0.4 }));
    knot.position.set(0.7, 0.1, 0.2); knot.castShadow = true; knot.receiveShadow = true;
    const sun = new THREE.DirectionalLight(0xffffff, 1.5); sun.position.set(2, 4, 2); sun.castShadow = true; sun.shadow.mapSize.set(512, 512);
    const spot = new THREE.SpotLight(0xffeedd, 25, 0, Math.PI / 5, 0.3, 2); spot.position.set(-2, 3, 1.5); spot.castShadow = true; spot.shadow.mapSize.set(512, 512);
    const point = new THREE.PointLight(0x88ff88, 6, 0, 2); point.position.set(0.2, 0.9, 1.0); point.castShadow = true; point.shadow.mapSize.set(256, 256);
    scene.add(floor, wall, box, knot, sun, spot, spot.target, point, new THREE.AmbientLight(0xffffff, 0.15));
    return { scene, camera: camera(THREE, [0, 1.4, 3.6]) };
  },
  // r186 WebGLClipping on programs: a global plane, local union/intersection
  // planes, clipShadows into two program shadow maps.
  shaderlib_clipping(THREE, renderer) {
    renderer.shadowMap.enabled = true;
    renderer.localClippingEnabled = true;
    renderer.clippingPlanes = [new THREE.Plane(new THREE.Vector3(-1, 0, 0), 1.4)];
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x101018);
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(8, 8), new THREE.MeshPhongMaterial({ color: 0xcccccc }));
    floor.rotation.x = -Math.PI / 2; floor.position.y = -0.7; floor.receiveShadow = true;
    const knot = new THREE.Mesh(new THREE.TorusKnotGeometry(0.35, 0.12, 96, 12), new THREE.MeshStandardMaterial({ color: 0xffaa44, roughness: 0.4, side: THREE.DoubleSide,
      clippingPlanes: [new THREE.Plane(new THREE.Vector3(0, -1, 0), 0.15)], clipShadows: true }));
    knot.position.set(-0.6, 0.1, 0); knot.castShadow = true; knot.receiveShadow = true;
    const ball = new THREE.Mesh(new THREE.SphereGeometry(0.45, 48, 24), new THREE.MeshPhongMaterial({ color: 0x44aaff, side: THREE.DoubleSide, clipIntersection: true,
      clippingPlanes: [new THREE.Plane(new THREE.Vector3(1, 0, 0), -0.7), new THREE.Plane(new THREE.Vector3(0, 1, 0), 0)] }));
    ball.position.set(0.7, 0, 0.1); ball.castShadow = true; ball.receiveShadow = true;
    const sun = new THREE.DirectionalLight(0xffffff, 1.5); sun.position.set(2, 4, 2); sun.castShadow = true; sun.shadow.mapSize.set(512, 512);
    const spot = new THREE.SpotLight(0xffeedd, 20, 0, Math.PI / 5, 0.3, 2); spot.position.set(-2, 3, 1.5); spot.castShadow = true; spot.shadow.mapSize.set(512, 512);
    scene.add(floor, knot, ball, sun, spot, spot.target, new THREE.AmbientLight(0xffffff, 0.2));
    return { scene, camera: camera(THREE, [0, 1.2, 3.4]) };
  },
  // r186 WebGLBackground meshes on the program route: a cube background with
  // rotation/intensity behind an envMapped sphere, and (second scenario) a 2D
  // texture background with a uv transform.
  shaderlib_background_cube(THREE) {
    const scene = new THREE.Scene();
    const face = (r, g, b) => {
      const n = 32, d = new Uint8Array(n * n * 4);
      for (let i = 0; i < n * n; i++) { const k = ((i % n >> 2) + (i >> 7)) % 2 ? 1 : 0.4; d.set([r * k, g * k, b * k, 255], i * 4); }
      const t = new THREE.DataTexture(d, n, n); t.needsUpdate = true; return t;
    };
    const cube = new THREE.CubeTexture([face(255, 90, 90), face(90, 255, 90), face(90, 90, 255), face(255, 255, 90), face(90, 255, 255), face(255, 90, 255)]);
    cube.colorSpace = THREE.SRGBColorSpace; cube.needsUpdate = true;
    scene.background = cube; scene.backgroundIntensity = 0.8; scene.backgroundRotation.set(0.2, 0.6, 0);
    scene.add(new THREE.Mesh(new THREE.SphereGeometry(0.7, 48, 24), new THREE.MeshBasicMaterial({ envMap: cube })));
    return { scene, camera: camera(THREE, [0.6, 0.4, 3]) };
  },
  shaderlib_background_texture(THREE) {
    const scene = new THREE.Scene();
    const n = 64, d = new Uint8Array(n * n * 4);
    for (let i = 0; i < n * n; i++) { const x = i % n, y = i >> 6; d.set([x * 4, y * 4, ((x >> 3) + (y >> 3)) % 2 ? 220 : 40, 255], i * 4); }
    const t = new THREE.DataTexture(d, n, n); t.colorSpace = THREE.SRGBColorSpace; t.repeat.set(2, 1); t.wrapS = THREE.RepeatWrapping; t.needsUpdate = true;
    scene.background = t;
    const box = new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.8, 0.8), new THREE.MeshNormalMaterial()); box.rotation.set(0.4, 0.6, 0);
    scene.add(box);
    return { scene, camera: camera(THREE, [0, 0, 3]) };
  },
  // WebGLRenderer in-shader tone mapping (per material, before blending and
  // sRGB encoding): ACES with a transparent blend and a toneMapped:false draw.
  shaderlib_tone_mapping(THREE, renderer) {
    renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.4;
    renderer.shadowMap.enabled = true;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x203040);
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(6, 6), new THREE.MeshStandardMaterial({ color: 0xbbbbbb, roughness: 0.8 }));
    floor.rotation.x = -Math.PI / 2; floor.position.y = -0.6; floor.receiveShadow = true;
    const hot = new THREE.Mesh(new THREE.SphereGeometry(0.4, 40, 20), new THREE.MeshStandardMaterial({ color: 0xff8844, emissive: 0xff4400, emissiveIntensity: 2, roughness: 0.3 }));
    hot.position.set(-0.7, 0, 0); hot.castShadow = true;
    const glass = new THREE.Mesh(new THREE.BoxGeometry(0.7, 0.7, 0.7), new THREE.MeshPhongMaterial({ color: 0x66aaff, transparent: true, opacity: 0.5, shininess: 80 }));
    glass.position.set(0.5, 0, 0.3); glass.rotation.set(0.3, 0.5, 0); glass.castShadow = true;
    const flat = new THREE.Mesh(new THREE.PlaneGeometry(0.6, 0.6), new THREE.MeshBasicMaterial({ color: 0xffffff, toneMapped: false }));
    flat.position.set(0.9, 0.7, -0.6);
    const sun = new THREE.DirectionalLight(0xffffff, 3); sun.position.set(1, 3, 2); sun.castShadow = true; sun.shadow.mapSize.set(512, 512);
    scene.add(floor, hot, glass, flat, sun, new THREE.HemisphereLight(0x8899ff, 0x332211, 1.2));
    return { scene, camera: camera(THREE, [0, 1, 3.2]) };
  },
  // r186 WebGLCubeRenderTarget.fromEquirectangularTexture (CubeCamera faces,
  // generated mips) for an equirect background and a Phong equirect envMap.
  shaderlib_equirect_cube(THREE) {
    const scene = new THREE.Scene();
    const w = 128, h = 64, d = new Uint8Array(w * h * 4);
    for (let i = 0; i < w * h; i++) { const x = i % w, y = (i / w) | 0; d.set([(x * 2) & 255, 255 - y * 4, ((x >> 3) + (y >> 3)) % 2 ? 220 : 50, 255], i * 4); }
    const pano = new THREE.DataTexture(d, w, h); pano.mapping = THREE.EquirectangularReflectionMapping; pano.colorSpace = THREE.SRGBColorSpace;
    pano.magFilter = THREE.LinearFilter; pano.minFilter = THREE.LinearMipmapLinearFilter; pano.generateMipmaps = true; pano.needsUpdate = true;
    scene.background = pano;
    const ball = new THREE.Mesh(new THREE.SphereGeometry(0.6, 48, 24), new THREE.MeshPhongMaterial({ color: 0xffffff, envMap: pano, reflectivity: 0.8, shininess: 60 }));
    scene.add(ball, new THREE.DirectionalLight(0xffffff, 1.5), new THREE.AmbientLight(0xffffff, 0.3));
    return { scene, camera: camera(THREE, [0.8, 0.3, 2.6]) };
  },
  // Skinning on the program route (bone texture texelFetch, bindMatrix) under
  // in-shader tone mapping, casting into a program shadow map.
  shaderlib_skinning(THREE, renderer) {
    renderer.toneMapping = THREE.AgXToneMapping; renderer.shadowMap.enabled = true;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x202028);
    const height = 1.6, segments = 8, geometry = new THREE.CylinderGeometry(0.15, 0.15, height, 16, segments * 2, false);
    const position = geometry.attributes.position, skinIndex = [], skinWeight = [], v = new THREE.Vector3();
    for (let i = 0; i < position.count; i++) {
      v.fromBufferAttribute(position, i);
      const y = v.y + height / 2, w = Math.min(1, Math.max(0, y / height));
      skinIndex.push(0, 1, 0, 0); skinWeight.push(1 - w, w, 0, 0);
    }
    geometry.setAttribute('skinIndex', new THREE.Uint16BufferAttribute(skinIndex, 4));
    geometry.setAttribute('skinWeight', new THREE.Float32BufferAttribute(skinWeight, 4));
    const root = new THREE.Bone(), tip = new THREE.Bone();
    root.position.y = -height / 2; tip.position.y = height; root.add(tip);
    const mesh = new THREE.SkinnedMesh(geometry, new THREE.MeshStandardMaterial({ color: 0xff9955, roughness: 0.5 }));
    mesh.add(root); mesh.bind(new THREE.Skeleton([root, tip]));
    tip.rotation.z = 0.9; root.rotation.x = 0.3; mesh.castShadow = true;
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(5, 5), new THREE.MeshStandardMaterial({ color: 0xaaaaaa }));
    floor.rotation.x = -Math.PI / 2; floor.position.y = -0.9; floor.receiveShadow = true;
    const sun = new THREE.DirectionalLight(0xffffff, 3); sun.position.set(1.5, 3, 1); sun.castShadow = true;
    scene.add(mesh, floor, sun, new THREE.AmbientLight(0xffffff, 0.4));
    return { scene, camera: camera(THREE, [0, 0.6, 3.2]) };
  },
  // WebGLMorphtargets on the program route: position/normal morph texture,
  // influences, relative and absolute targets, under in-shader tone mapping.
  shaderlib_morph(THREE, renderer) {
    renderer.toneMapping = THREE.NeutralToneMapping;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x182030);
    const geometry = new THREE.BoxGeometry(0.8, 0.8, 0.8, 8, 8, 8), p = geometry.attributes.position;
    const sphere = [], twist = [], v = new THREE.Vector3();
    for (let i = 0; i < p.count; i++) {
      v.fromBufferAttribute(p, i);
      const s = v.clone().normalize().multiplyScalar(0.55); sphere.push(s.x, s.y, s.z);
      const a = v.y * 2; twist.push(v.x * Math.cos(a) - v.z * Math.sin(a), v.y * 1.3, v.x * Math.sin(a) + v.z * Math.cos(a));
    }
    geometry.morphAttributes.position = [new THREE.Float32BufferAttribute(sphere, 3), new THREE.Float32BufferAttribute(twist, 3)];
    const mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color: 0x66ccff, roughness: 0.35 }));
    mesh.morphTargetInfluences = [0.55, 0.3]; mesh.rotation.set(0.4, 0.6, 0); mesh.position.x = -0.55;
    const rel = new THREE.BufferGeometry().copy(geometry); rel.morphTargetsRelative = true;
    rel.morphAttributes.position = [new THREE.Float32BufferAttribute(sphere.map(x => x * 0.4), 3)];
    const mesh2 = new THREE.Mesh(rel, new THREE.MeshLambertMaterial({ color: 0xffaa66 }));
    mesh2.morphTargetInfluences = [0.8]; mesh2.position.x = 0.6; mesh2.rotation.set(0.2, -0.5, 0);
    const sun = new THREE.DirectionalLight(0xffffff, 2.5); sun.position.set(1, 2, 3);
    scene.add(mesh, mesh2, sun, new THREE.AmbientLight(0xffffff, 0.4));
    return { scene, camera: camera(THREE, [0, 0.4, 3]) };
  },
  // RectAreaLight (LTC tables) and a projected spot-light map: lights only the
  // r186 programs shade, so lit materials take the program route.
  shaderlib_area_lights(THREE) {
    THREE.RectAreaLightUniformsLib.init();
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x050508);
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(6, 6), new THREE.MeshStandardMaterial({ color: 0x808080, roughness: 0.2, metalness: 0 }));
    floor.rotation.x = -Math.PI / 2; floor.position.y = -0.6;
    const knot = new THREE.Mesh(new THREE.TorusKnotGeometry(0.3, 0.1, 96, 12), new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.3, metalness: 0.1 }));
    const red = new THREE.RectAreaLight(0xff3333, 6, 0.6, 1.4); red.position.set(-0.9, 0.1, -0.6); red.lookAt(0, 0, 0);
    const blue = new THREE.RectAreaLight(0x3355ff, 6, 0.6, 1.4); blue.position.set(0.9, 0.1, -0.6); blue.lookAt(0, 0, 0);
    const n = 16, d = new Uint8Array(n * n * 4);
    for (let i = 0; i < n * n; i++) d.set(((i % n) >> 2) % 2 ^ ((i >> 6) % 2) ? [255, 230, 120, 255] : [40, 60, 255, 255], i * 4);
    const map = new THREE.DataTexture(d, n, n); map.colorSpace = THREE.SRGBColorSpace; map.needsUpdate = true;
    const spot = new THREE.SpotLight(0xffffff, 30, 0, Math.PI / 6, 0.2, 2); spot.position.set(0.5, 2.5, 1.2); spot.map = map;
    scene.add(floor, knot, red, blue, spot, spot.target);
    return { scene, camera: camera(THREE, [0, 0.9, 2.8]) };
  },
  // MeshPhysicalMaterial extensions via ShaderLib programs (WebGL surface), and a
  // nearest-sampled float32 DataTexture read by a ShaderMaterial.
  shaderlib_physical(THREE) {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x181818);
    const mats = [
      new THREE.MeshPhysicalMaterial({ color: 0xaa2222, roughness: 0.6, clearcoat: 1, clearcoatRoughness: 0.1 }),
      new THREE.MeshPhysicalMaterial({ color: 0x223355, roughness: 0.8, sheen: 1, sheenColor: new THREE.Color(0xffaa55), sheenRoughness: 0.4 }),
      new THREE.MeshPhysicalMaterial({ color: 0xffffff, metalness: 0.2, roughness: 0.3, iridescence: 1, iridescenceIOR: 1.4, iridescenceThicknessRange: [200, 600] }),
      new THREE.MeshPhysicalMaterial({ color: 0x88cc88, roughness: 0.35, ior: 1.8, specularIntensity: 0.7, specularColor: new THREE.Color(0xff8888) }),
    ];
    mats.forEach((m, i) => {
      const mesh = new THREE.Mesh(new THREE.SphereGeometry(0.4, 40, 20), m);
      mesh.position.set((i % 2) * 1.0 - 0.5 - (i < 2 ? 0.45 : -0.45), i < 2 ? 0.45 : -0.45, 0);
      scene.add(mesh);
    });
    const data = new Float32Array(4 * 4 * 4);
    for (let i = 0; i < 16; i++) data.set([i / 15, 1 - i / 15, (i % 3) / 2, 1], i * 4);
    const ft = new THREE.DataTexture(data, 4, 4, THREE.RGBAFormat, THREE.FloatType); ft.needsUpdate = true;
    const quad = new THREE.Mesh(new THREE.PlaneGeometry(0.5, 0.5), new THREE.ShaderMaterial({ uniforms: { t: { value: ft } },
      vertexShader: 'varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
      fragmentShader: 'uniform sampler2D t; varying vec2 vUv; void main() { gl_FragColor = texture2D(t, vUv); }' }));
    quad.position.set(1.3, 0.9, 0);
    const key = new THREE.DirectionalLight(0xffffff, 2.5); key.position.set(2, 3, 4);
    scene.add(quad, key, new THREE.HemisphereLight(0x8899aa, 0x332211, 0.8), new THREE.PointLight(0x66aaff, 5, 0, 2));
    return { scene, camera: camera(THREE, [0, 0, 3.4]) };
  },
  // Per-axis normalScale on the derivative tangent frame (GLTFLoader sets y=-1 when
  // geometry has no tangents) and a flat-shaded lit mesh with no normal attribute.
  normal_scale_and_flat(THREE) {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x202028);
    const size = 32, data = new Uint8Array(size * size * 4);
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
      const nx = 0.6 * Math.sin((2 * Math.PI * x) / 8), ny = 0.6 * Math.sin((2 * Math.PI * y) / 16);
      const nz = Math.sqrt(Math.max(0, 1 - nx * nx - ny * ny)), i = (y * size + x) * 4;
      data.set([(nx * 0.5 + 0.5) * 255, (ny * 0.5 + 0.5) * 255, (nz * 0.5 + 0.5) * 255, 255], i);
    }
    const normalMap = new THREE.DataTexture(data, size, size);
    normalMap.needsUpdate = true;
    [[1, 1], [1, -1], [0.5, 2]].forEach(([sx, sy], i) => {
      const plane = new THREE.Mesh(new THREE.PlaneGeometry(0.9, 0.9), new THREE.MeshStandardMaterial({ color: 0xbbbbbb, roughness: 0.5, normalMap, normalScale: new THREE.Vector2(sx, sy) }));
      plane.position.set((i - 1) * 1.0, 0.45, 0); plane.rotation.x = -0.3;
      scene.add(plane);
    });
    const geometry = new THREE.IcosahedronGeometry(0.4, 0);
    geometry.deleteAttribute('normal');
    const flat = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color: 0x55aa77, flatShading: true }));
    flat.position.set(0, -0.6, 0); flat.rotation.set(0.4, 0.6, 0);
    const light = new THREE.DirectionalLight(0xffffff, 3); light.position.set(-1, 2, 2);
    scene.add(flat, light, new THREE.AmbientLight(0xffffff, 0.3));
    return { scene, camera: camera(THREE) };
  },
};
function camera(THREE, position = [0, 0, 3.2]) {
  const c = new THREE.PerspectiveCamera(50, 320 / 240, 0.1, 50);
  c.position.set(...position); c.lookAt(0, 0, 0);
  return c;
}
