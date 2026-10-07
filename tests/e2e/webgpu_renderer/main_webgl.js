import * as THREE from 'three';
import { SunLight } from 'three/addons/lights/SunLight.js';
import { RectAreaLightUniformsLib } from 'three/addons/lights/RectAreaLightUniformsLib.js';
import { run } from './runner.js';
run({ ...THREE, SunLight, RectAreaLightUniformsLib }, (canvas) => new THREE.WebGLRenderer({ canvas, antialias: false, preserveDrawingBuffer: true }));
