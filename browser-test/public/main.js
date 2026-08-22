// Drives a full SetFit round trip in the page: fetch a checkpoint, fine-tune
// from a handful of examples one step per frame, pack a bundle, load it back
// and classify a document long enough to chunk.
//
// The point is not accuracy -- the checkpoint is 32-wide random weights. It is
// that every piece runs at all in a browser (tokenizers without rayon threads,
// entropy through getrandom, autodiff, safetensors) and that the tab stays
// responsive while it happens.

import init, { Trainer, Classifier } from './pkg/setfit_wasm.js';

const log = (msg) => {
  document.getElementById('log').textContent += msg + '\n';
  console.log(msg);
};
const phase = (msg) => { document.getElementById('phase').textContent = msg; };
const progress = (f) => {
  document.querySelector('#bar > div').style.width = `${(f * 100).toFixed(1)}%`;
};

const result = { ok: false, errors: [], steps: [], timings: {} };
window.__RESULT__ = result;

addEventListener('error', (e) => result.errors.push(String(e.message)));
addEventListener('unhandledrejection', (e) => result.errors.push(String(e.reason)));

// A main-thread frame clock, running for the whole test. The largest interval
// between consecutive frames is how long the main thread was blocked at its
// worst -- the number that decides whether step-wise training actually keeps a
// tab usable.
const frames = [];
let heartbeat = true;
(function tick(t) {
  frames.push(t);
  if (heartbeat) requestAnimationFrame(tick);
})(performance.now());

const nextFrame = () => new Promise(requestAnimationFrame);

function frameGaps(fromIndex) {
  const gaps = [];
  for (let i = fromIndex + 1; i < frames.length; i++) gaps.push(frames[i] - frames[i - 1]);
  gaps.sort((a, b) => a - b);
  return gaps;
}

const TRAIN_REQUEST = {
  labels: ['alpha', 'omega'],
  examples: [
    { text: 'a b c', labels: [0] },
    { text: 'a c b', labels: [0] },
    { text: 'b a c', labels: [0] },
    { text: 'c a b', labels: [0] },
    { text: 'a b a', labels: [0] },
    { text: 'b c a', labels: [0] },
    { text: 'x y z', labels: [1] },
    { text: 'z y x', labels: [1] },
    { text: 'y x z', labels: [1] },
    { text: 'x z y', labels: [1] },
    { text: 'z z y', labels: [1] },
    { text: 'y y x', labels: [1] },
  ],
  multi_label: false,
  num_iterations: 4,
  head_epochs: 20,
  seed: 42,
};

// Long enough to exercise the chunker rather than a single window: the toy
// vocabulary is one token per letter, so this is ~900 tokens over several
// sentence-bounded windows.
function longDocument() {
  const sentence = (letters) => letters.join(' ') + '. ';
  let doc = '';
  for (let i = 0; i < 60; i++) {
    doc += sentence(['x', 'y', 'z', 'z', 'y', 'x', 'y', 'z', 'x', 'z', 'y', 'x', 'z', 'x', 'y']);
  }
  return doc;
}

async function main() {
  try {
    const t0 = performance.now();
    phase('loading wasm');
    await init();
    result.timings.wasm_init_ms = Math.round(performance.now() - t0);
    log(`wasm loaded in ${result.timings.wasm_init_ms} ms`);

    phase('fetching checkpoint');
    const base = './checkpoint';
    const [configJson, weights, tokenizerJson] = await Promise.all([
      fetch(`${base}/config.json`).then((r) => r.text()),
      fetch(`${base}/model.safetensors`).then((r) => r.arrayBuffer()).then((b) => new Uint8Array(b)),
      fetch(`${base}/tokenizer.json`).then((r) => r.arrayBuffer()).then((b) => new Uint8Array(b)),
    ]);
    log(`checkpoint: ${weights.length} bytes of weights, ${tokenizerJson.length} bytes of tokenizer`);

    // Construction alone is a real test: it tokenizes every example (the rayon
    // question) and seeds the head from getrandom (the entropy question).
    phase('constructing trainer');
    const tConstruct = performance.now();
    const trainer = new Trainer(configJson, weights, tokenizerJson, JSON.stringify(TRAIN_REQUEST));
    result.timings.trainer_new_ms = Math.round(performance.now() - tConstruct);
    result.total_steps = trainer.totalSteps();
    log(`trainer constructed in ${result.timings.trainer_new_ms} ms, ${result.total_steps} steps planned`);

    phase('training');
    const trainStartFrame = frames.length - 1;
    const tTrain = performance.now();
    let p;
    do {
      p = JSON.parse(trainer.step());
      progress(p.fraction);
      result.steps.push({ stage: p.stage, step: p.step, loss: p.loss });
      await nextFrame();
    } while (!p.done);
    result.timings.train_ms = Math.round(performance.now() - tTrain);

    const gaps = frameGaps(trainStartFrame);
    result.responsiveness = {
      frames_during_training: gaps.length,
      median_frame_gap_ms: Math.round(gaps[Math.floor(gaps.length / 2)] ?? 0),
      worst_frame_gap_ms: Math.round(gaps[gaps.length - 1] ?? 0),
    };
    log(`trained in ${result.timings.train_ms} ms over ${gaps.length} frames`);
    log(`frame gap: median ${result.responsiveness.median_frame_gap_ms} ms, worst ${result.responsiveness.worst_frame_gap_ms} ms`);

    phase('packing bundle');
    const bundle = trainer.finish();
    result.bundle_bytes = bundle.length;
    log(`bundle: ${bundle.length} bytes`);

    phase('classifying');
    const classifier = new Classifier(bundle);
    result.labels = classifier.labels();

    const short = TRAIN_REQUEST.examples.map((e) => {
      const pred = JSON.parse(classifier.classify(e.text));
      return { text: e.text, expected: result.labels[e.labels[0]], got: pred.labels[0], scores: pred.scores };
    });
    result.short = short;
    result.short_correct = short.filter((s) => s.expected === s.got).length;
    log(`held-in examples classified correctly: ${result.short_correct}/${short.length}`);

    const doc = longDocument();
    const tClassify = performance.now();
    const long = JSON.parse(classifier.classify(doc));
    result.timings.classify_long_ms = Math.round(performance.now() - tClassify);
    result.long = {
      bytes: doc.length,
      chunks_seen: long.chunks_seen,
      labels: long.labels,
      scores: long.scores,
      evidence: long.evidence,
    };
    log(`long document: ${doc.length} bytes -> ${long.chunks_seen} chunks -> ${long.labels.join(', ')} in ${result.timings.classify_long_ms} ms`);
    if (long.evidence.length) {
      const e = long.evidence[0];
      log(`strongest chunk: ${e.label} ${e.score.toFixed(3)} at bytes ${e.start}..${e.end}`);
    }

    result.ok = result.errors.length === 0 && long.chunks_seen > 1;
    phase(result.ok ? 'done' : 'finished with problems');
  } catch (e) {
    result.errors.push(String(e && e.stack ? e.stack : e));
    phase('failed');
    log(`FAILED: ${e}`);
  } finally {
    heartbeat = false;
    result.done = true;
    document.title = result.ok ? 'PASS' : 'FAIL';
  }
}

main();
