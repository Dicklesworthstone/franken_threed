// Shared source-valid programs for compiler lowering and rendered ESSL parity.
// Each result contains exact 8-bit channel values; no floating tolerance hides
// an extra index evaluation, a moved argument read or a changed destination.
export const lvalueCases = [
  {name: 'assignment-result', expected: [71, 20, 71, 1], body: `
    int cursor = 0; float values[4] = float[4](10., 20., 30., 40.);
    float result = (values[cursor++] = 71.);
    color = vec4(values[0], values[1], result, float(cursor));`},
  {name: 'compound-assignment-address', expected: [11, 20, 11, 1], body: `
    int cursor = 0; float values[3] = float[3](10., 20., 30.);
    float result = (values[cursor] += float(++cursor));
    color = vec4(values[0], values[1], result, float(cursor));`},
  {name: 'prefix-index', expected: [11, 20, 11, 1], body: `
    int cursor = 0; float values[3] = float[3](10., 20., 30.);
    float result = ++values[cursor++];
    color = vec4(values[0], values[1], result, float(cursor));`},
  {name: 'postfix-index', expected: [11, 20, 10, 1], body: `
    int cursor = 0; float values[3] = float[3](10., 20., 30.);
    float result = values[cursor++]++;
    color = vec4(values[0], values[1], result, float(cursor));`},
  {name: 'nested-aggregate-address', expected: [17, 17, 21, 50], declarations: 'struct Item { mat2 frame; };', body: `
    Item items[2] = Item[2](Item(mat2(10., 20., 30., 40.)), Item(mat2(50., 60., 70., 80.)));
    int item = 0, column = 0, row = 0;
    float result = (items[item++].frame[column++][row++] += 7.);
    color = vec4(result, items[0].frame[0][0], float(item + 4 * column + 16 * row), items[1].frame[0][0]);`},
  {name: 'nested-swizzle-stores', expected: [21, 55, 66, 76], body: `
    vec4 value = vec4(10., 20., 30., 40.); int cursor = 0;
    value.wyx[cursor++] = 75.; value.zy.yx = vec2(55., 66.); value.xz.x = 21.;
    color = vec4(value.xyz, value.w + float(cursor));`},
  {name: 'out-address-before-body', expected: [73, 20, 30, 2], declarations: `
    int cursor = 0; void store(out float value) { cursor = 2; value = 73.; }`, body: `
    float values[4] = float[4](10., 20., 30., 40.); store(values[cursor++]);
    color = vec4(values[0], values[1], values[2], float(cursor));`},
  {name: 'out-address-before-later-argument', expected: [62, 20, 30, 2], declarations: `
    void store(out float value, int amount) { value = 60. + float(amount); }`, body: `
    int cursor = 0; float values[4] = float[4](10., 20., 30., 40.);
    store(values[cursor++], ++cursor);
    color = vec4(values[0], values[1], values[2], float(cursor));`},
  {name: 'inout-address-and-result', expected: [12, 20, 12, 2], declarations: `
    float adjust(inout float value, int amount) { value += float(amount); return value; }`, body: `
    int cursor = 0; float values[4] = float[4](10., 20., 30., 40.);
    float result = adjust(values[cursor++], ++cursor);
    color = vec4(values[0], values[1], result, float(cursor));`},
  // Chromium 153 ANGLE/SwiftShader defers these aliased argument reads. Keep
  // its observed values separate from the ESSL 3.00 section 6.1.1 results;
  // the browser regression confirms them using raw WebGL2 as well as r186.
  {name: 'inout-value-before-later-argument', expected: [13, 20, 30, 255], sourceExpected: [73, 20, 30, 255], declarations: `
    vec4 values = vec4(10., 20., 30., 40.);
    float mutate() { values.x = 70.; return 3.; }
    void adjust(inout float value, float amount) { value += amount; }`, body: `
    adjust(values.x, mutate()); color = vec4(values.xyz, 255.);`},
  {name: 'out-nested-swizzle-address', expected: [10, 20, 77, 2], declarations: `
    int cursor = 0; void store(out float value) { cursor = 2; value = 77.; }`, body: `
    vec4 value = vec4(10., 20., 30., 40.); store(value.zyx[cursor++]);
    color = vec4(value.xyz, float(cursor));`},
  {name: 'inout-array-swizzle', expected: [13, 22, 30, 2], declarations: `
    void adjust(inout vec2 value, int amount) { value += vec2(float(amount), 3.); }`, body: `
    int cursor = 0; vec2 values[3] = vec2[3](vec2(10., 20.), vec2(30., 40.), vec2(50., 60.));
    adjust(values[cursor++].yx, ++cursor);
    color = vec4(values[0], values[1].x, float(cursor));`},
  {name: 'modf-swizzle-output', expected: [13, 50, 1, 0], body: `
    int cursor = 0; vec4 values = vec4(0.); float fraction = modf(13.5, values.zy[cursor++]);
    color = vec4(values.z, fraction * 100., float(cursor), values.y);`},
  {name: 'modf-input-before-output-address', expected: [14, 25, 1, 99], sourceExpected: [99, 0, 1, 99], declarations: `
    float value = 14.25; int cursor = 0;
    int destination() { value = 99.; return cursor++; }`, body: `
    vec2 whole = vec2(0.); float fraction = modf(value, whole[destination()]);
    color = vec4(whole.x, fraction * 100., float(cursor), value);`},
  {name: 'user-argument-value-order', expected: [26, 4, 0, 255], sourceExpected: [46, 4, 0, 255], declarations: `
    float combine(float a, float b, float c) { return a * 10. + b + c; }`, body: `
    float value = 2.; float result = combine(value, ++value, value++);
    color = vec4(result, value, 0., 255.);`},
  {name: 'builtin-argument-effect-order', expected: [30, 12, 1, 255], declarations: `
    int trace = 0;
    float first() { trace = trace * 10 + 1; return 20.; }
    float second() { trace = trace * 10 + 2; return 40.; }`, body: `
    int weight = 0; float result = mix(first(), second(), float(++weight) * .5);
    color = vec4(result, float(trace), float(weight), 255.);`},
  {name: 'constructor-argument-effect-order', expected: [15, 25, 1, 12], declarations: `
    int trace = 0;
    float first() { trace = trace * 10 + 1; return 15.; }
    float second() { trace = trace * 10 + 2; return 25.; }`, body: `
    int cursor = 0; color = vec4(first(), second(), float(++cursor), float(trace));`},
  {name: 'index-side-effect-before-value-read', expected: [81, 81, 20, 255], declarations: `
    float values[2] = float[2](10., 20.);
    int destination() { values[0] = 81.; return 0; }
    float consume(float value) { return value; }`, body: `
    float result = consume(values[destination()]); color = vec4(values[0], result, values[1], 255.);`},
  {name: 'aggregate-producer-before-index', expected: [20, 12, 0, 255], declarations: `
    int trace = 0;
    vec4 produce() { trace = trace * 10 + 1; return vec4(10., 20., 30., 40.); }
    int destination() { trace = trace * 10 + 2; return 1; }`, body: `
    float result = produce()[destination()]; color = vec4(result, float(trace), 0., 255.);`},
  {name: 'nested-assignment-results', expected: [17, 17, 17, 2], body: `
    int cursor = 0; float values[4] = float[4](10., 20., 30., 40.);
    float result = (values[cursor++] = (values[cursor++] = 17.));
    color = vec4(values[0], values[1], result, float(cursor));`},
  {name: 'multiple-output-addresses', expected: [15, 20, 30, 2], declarations: `
    void pair(inout float a, out float b) { b = a + 10.; a += 5.; }`, body: `
    int cursor = 0; vec4 values = vec4(10., 20., 30., 40.); pair(values[cursor++], values[cursor++]);
    color = vec4(values.xyz, float(cursor));`},
  {name: 'unsigned-index-decrements', expected: [19, 29, 20, 3], body: `
    uint cursor = 1u; uvec4 values = uvec4(10u, 20u, 30u, 40u);
    uint before = values[cursor++]--; uint after = --values[cursor++];
    color = vec4(float(values.y), float(after), float(before), float(cursor));`},
  {name: 'constant-aggregate-index', expected: [23, 1, 0, 255], body: `
    const int key = ivec2(1, 2)[0]; int result = 0;
    switch (1) { case key: result = 11; break; default: result = 99; }
    switch (2) { case ivec2(1, 2)[1]: result += 12; break; default: result = 98; }
    color = vec4(float(result), float(key), 0., 255.);`},
];

export function lvalueSources(fixture, stage) {
  const precision = 'precision highp float;precision highp int;';
  const vertex = `${precision}in vec3 position;void main(){gl_Position=vec4(position.xy,0.,1.);}`;
  const declarations = fixture.declarations ?? '';
  if (stage === 'vertex') return {
    vertex: `${precision}in vec3 position;flat out vec4 observedResult;${declarations}
      void main(){vec4 color;${fixture.body}observedResult=color/255.;gl_Position=vec4(position.xy,0.,1.);}`,
    fragment: `${precision}flat in vec4 observedResult;out vec4 color;void main(){color=observedResult;}`,
  };
  return {vertex, fragment: `${precision}out vec4 color;${declarations}void main(){${fixture.body}color/=255.;}`};
}
