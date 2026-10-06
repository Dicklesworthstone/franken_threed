import * as THREE from 'three/webgpu';
import { run } from './runner.js';
run(THREE, (canvas) => new THREE.WebGPURenderer({ canvas, antialias: false }));
