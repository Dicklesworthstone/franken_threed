import * as THREE from 'three';
import { run } from './runner.js';
run(THREE, (canvas) => new THREE.WebGLRenderer({ canvas, antialias: false, preserveDrawingBuffer: true }));
