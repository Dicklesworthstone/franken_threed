import * as THREE from 'three/webgpu';
import { SunLight } from 'three/addons/lights/SunLight.js';
import { SunLightNode } from 'three/addons/lights/SunLightNode.js';
import { run } from './runner.js';
// Addon lights ride along with the namespace; WebGPU apps register their node.
const antialias = ['native_pipeline_msaa_depth', 'native_pipeline_mrt_updates', 'native_clipping_msaa'].includes(location.hash.slice(1));
run({ ...THREE, SunLight, SunLightNode }, (canvas) => new THREE.WebGPURenderer({ canvas, antialias }));
