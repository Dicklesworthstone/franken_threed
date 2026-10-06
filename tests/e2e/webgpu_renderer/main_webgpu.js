import * as THREE from 'three/webgpu';
import { SunLight } from 'three/addons/lights/SunLight.js';
import { SunLightNode } from 'three/addons/lights/SunLightNode.js';
import { run } from './runner.js';
// Addon lights ride along with the namespace; WebGPU apps register their node.
run({ ...THREE, SunLight, SunLightNode }, (canvas) => new THREE.WebGPURenderer({ canvas, antialias: false }));
