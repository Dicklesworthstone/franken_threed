import * as THREE from 'three';
import { SunLight } from 'three/addons/lights/SunLight.js';
import { run } from './runner.js';
run({ ...THREE, SunLight }, (canvas) => new THREE.WebGLRenderer({ canvas, antialias: false, preserveDrawingBuffer: true }));
