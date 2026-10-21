# Multiple-scattering environment lighting

The existing animation environment receiver can recover light lost between GGX
microfacets on rough surfaces. This is an opt-in extension to the default
single-scattering profile in `ANIMATION_ENVIRONMENT.md`, not a second renderer
or environment-filter implementation.

```js
// Prepare/load the environment and enable renderer.environment as before.
// This option belongs to the FRAME descriptor, not the environment factory.
scene.render({
  ...frame,
  environment: { map: environment, intensity: 1, scattering: 'multiple' },
});
```

`createGpuAnimationRenderer` consumes the same `frame.environment` descriptor.
The ordinary animation scene and glTF model paths forward that descriptor.
`scattering: 'single'`, or omitting the property, retains the original receiver
arithmetic and uniform bytes. Invalid modes reject before receiver packing.
Changing the mode reuses the same maps, bindings, shader, and 64-byte uniform;
it does not require image loading, filtering, a pipeline rebuild, or allocation
of another texture. Existing rotation, intensity, ownership, device-identity,
cancellation and disposal rules are unchanged.

## Lighting model

The multiple-scattering model uses the Fdez-Aguera repeated-bounce approximation
(JCGT 8(1), 2019, https://jcgt.org/published/0008/01/03/). For DFG coefficients
`A,B`, the single-scattering energy is `Ess = A+B`. The hemisphere-averaged
Schlick Fresnel is `Favg = F0 + (F90-F0)/21`; the additional reflected fraction is
`(F0*A + F90*B) * Favg * (1-Ess) / (1-Favg*(1-Ess))`.

Single scattering still samples the existing GGX-prefiltered specular cube.
Repeated bounces use the cosine-weighted irradiance cube, which already stores
`E/pi`; there is no extra division by pi. Dielectric and metallic lobes are
compensated separately before mixing by metalness, because compensation is
nonlinear. Only the dielectric fraction has a diffuse substrate. Its remaining
energy subtracts both its single and repeated scattering before multiplication
by base color. A perfectly reflecting white metal therefore retains unit energy
in a white furnace, including at high roughness.

The native physical-specular shader variant uses its authored dielectric F0 and
F90, including a disabled zero-specular material. The generalized Fresnel average
retains F90 rather than assuming it is always one. This, the existing cubemap
filter, and the original generic material profile do not imply Three.js PMREM
or pixel equivalence. The default single-scattering variant is not changed.

Finite quadrature and RGBA16F interpolation can make `A+B` slightly exceed one.
Only multiple mode clamps negative coefficients and normalizes excessive total
DFG energy. The repeated-bounce denominator uses a cancellation-resistant form,
and the exact zero-energy perfect-reflector limit avoids a zero division. Energy
bounds assume reflectance inputs in `[0,1]`; HDR intensity and source radiance are
not clamped or tone mapped. This is an isotropic indirect-light approximation,
not path tracing or measured convergence to arbitrary rendered scenes.

Lambert and unlit behavior, direct-light BRDFs, emission, alpha and tone mapping
are not changed. Existing material occlusion and coating composition remain at
their surrounding renderer boundaries. In particular this feature does not add
multiple-scattering compensation to direct lights or a transmission model.

## Validation

```sh
node --test tools/ingest/animation_environment_scattering.test.mjs
```

The focused tests run production admission/packing and scalarize the emitted
component-independent WGSL arithmetic, comparing it against an independent sum
of repeated bounces. They cover white-furnace energy, colored metals/dielectrics,
partial metalness, distinct irradiance/radiance, zero F90, intensity, DFG edge
cases, legacy shader branches, binding layout and snapshot lifetime. The test
scalarization is explicitly not a WGSL compiler or native GPU execution. Native
browser validation was blocked by the implementation environment's browser
policy; rendered pixels, full renderer integration and full-workspace regression
remain separate checks. No wall-time or GPU speedup is asserted.
