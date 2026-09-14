/* The network, running in your browser.
 *
 * Not a library and not an inference API: this is the forward pass written out,
 * reading the weights that were trained in `sightline/train.py`. The whole model
 * is eight convolutions, a ReLU after each, an average, and two matrix
 * multiplies — small enough that shipping a runtime to execute it would be more
 * code than executing it.
 *
 * BatchNorm is absent because scripts/export_weights.py folded it into the
 * convolutions: at eval time it is an affine map with fixed constants, so it
 * can be pushed into the preceding weights exactly. Dropout is absent because
 * it does nothing at eval.
 *
 * Everything is Float32Array and flat indexing. Roughly 173 million multiply-
 * adds, which a browser does in well under a second — slow by the standards of
 * a GPU and instant by the standards of a person clicking a button.
 */
const Net = (() => {
  let manifest = null, W = null;

  async function load() {
    if (manifest) return manifest;
    const [m, buf] = await Promise.all([
      fetch("net/demandnet.json").then((r) => r.json()),
      fetch("net/demandnet.bin").then((r) => r.arrayBuffer()),
    ]);
    manifest = m;
    W = new Float32Array(buf);
    return manifest;
  }

  const take = (l) => W.subarray(l.offset, l.offset + l.count);

  /* A 3x3 convolution with stride 1 or 2, zero padding of 1, bias and ReLU.
     Written as the plain six-deep loop rather than an im2col: the shapes here
     are small, and the loop is the thing a reader can check against the maths. */
  function conv(src, cin, h, w, weights, bias, cout, stride) {
    const oh = Math.ceil(h / stride), ow = Math.ceil(w / stride);
    const dst = new Float32Array(cout * oh * ow);
    for (let o = 0; o < cout; o++) {
      const wo = o * cin * 9, base = o * oh * ow;
      for (let y = 0; y < oh; y++) {
        const iy0 = y * stride - 1;
        for (let x = 0; x < ow; x++) {
          const ix0 = x * stride - 1;
          let sum = bias[o];
          for (let c = 0; c < cin; c++) {
            const wc = wo + c * 9, sc = c * h * w;
            for (let ky = 0; ky < 3; ky++) {
              const iy = iy0 + ky;
              if (iy < 0 || iy >= h) continue;          // zero padding
              const row = sc + iy * w, krow = wc + ky * 3;
              for (let kx = 0; kx < 3; kx++) {
                const ix = ix0 + kx;
                if (ix < 0 || ix >= w) continue;
                sum += src[row + ix] * weights[krow + kx];
              }
            }
          }
          dst[base + y * ow + x] = sum > 0 ? sum : 0;   // ReLU
        }
      }
    }
    return { data: dst, c: cout, h: oh, w: ow };
  }

  function pool(t) {                                    // global average
    const out = new Float32Array(t.c), n = t.h * t.w;
    for (let c = 0; c < t.c; c++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += t.data[c * n + i];
      out[c] = s / n;
    }
    return out;
  }

  function linear(v, weights, bias, cout, relu) {
    const out = new Float32Array(cout);
    for (let o = 0; o < cout; o++) {
      let s = bias[o];
      for (let i = 0; i < v.length; i++) s += v[i] * weights[o * v.length + i];
      out[o] = relu && s < 0 ? 0 : s;
    }
    return out;
  }

  /* `rgb` is width*height*4 RGBA bytes straight out of a canvas, at the
     model's native 128 px. onBlock, if given, is handed each block's output as
     it is produced, so the page can draw the network thinking. */
  function run(rgb, px, onBlock) {
    const { input, target, layers } = manifest;
    let t = { data: new Float32Array(3 * px * px), c: 3, h: px, w: px };
    for (let i = 0, n = px * px; i < n; i++) {
      for (let c = 0; c < 3; c++) {
        t.data[c * n + i] = rgb[i * 4 + c] * input.scale + input.shift;
      }
    }

    let li = 0;
    while (layers[li].kind === "conv") {
      const cw = layers[li];
      t = conv(t.data, cw.cin, t.h, t.w, take(cw), take(layers[li + 1]),
        cw.cout, cw.stride);
      li += 2;
      // A block is finished when the next convolution belongs to a different
      // one. Reading that off the manifest rather than counting in twos keeps
      // this honest if the architecture ever stops being two convs per block.
      const next = layers[li];
      if (onBlock && (!next || next.kind !== "conv" || next.block !== cw.block)) {
        onBlock(cw.block, t);
      }
    }
    li++;                                               // the pool marker
    let v = pool(t);
    while (li < layers.length) {
      const lw = layers[li], lb = layers[li + 1];
      v = linear(v, take(lw), take(lb), lw.shape[0], lw.relu);
      li += 2;
    }
    const log10 = v[0] * target.sd + target.mu;
    return { log10, density: Math.pow(10, log10) };
  }

  return { load, run, manifest: () => manifest };
})();
