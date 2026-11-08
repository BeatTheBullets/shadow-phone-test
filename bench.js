// Spike (throwaway): how fast do the app's speech models run on this device?
// Shared by the browser page (index.html) and the Node check (node-check.mjs). Inputs are synthetic (noise of the
// right length): the time a model takes depends on the input's length, not on what is said.
// The aligner and the sound model are both wav2vec2-large (same size, same layers), so one file stands in for both.

const SR = 16000;
const noise = (secs) => { const a = new Float32Array(Math.round(secs * SR)); for (let i = 0; i < a.length; i++) a[i] = (Math.random() - 0.5) * 0.1; return a; };
const asrFeats = (secs) => { const t = Math.max(1, Math.round(secs * 100 / 6)); const a = new Float32Array(t * 560); for (let i = 0; i < a.length; i++) a[i] = Math.random() - 0.5; return [a, t]; };
const median = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
const now = () => (globalThis.performance ? performance.now() : Date.now());

export async function runBench({ ort, getModel, log, result }) {
  const opts = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
  const feed = {
    w2v: (s, secs) => s.run({ wav: new ort.Tensor("float32", noise(secs), [1, Math.round(secs * SR)]) }),
    asr: (s, secs) => { const [a, t] = asrFeats(secs); return s.run({ speech: new ort.Tensor("float32", a, [1, t, 560]), speech_lengths: new ort.Tensor("int32", Int32Array.from([t]), [1]) }); },
  };
  const kind = { embed: "w2v", large: "w2v", asr: "asr" };
  const label = { embed: "tone model", large: "aligner / sound model (each)", asr: "recogniser" };
  const timed = async (fn) => { const t = now(); await fn(); return now() - t; };
  const bytes = {};

  async function load(name) {
    if (!bytes[name]) { const t = now(); bytes[name] = await getModel(name); log(`${name}: downloaded ${(bytes[name].byteLength / 1e6).toFixed(0)} MB in ${((now() - t) / 1000).toFixed(1)} s`); }
    const t = now();
    const s = await ort.InferenceSession.create(bytes[name], opts);
    log(`${name}: ready in ${((now() - t) / 1000).toFixed(1)} s`);
    return s;
  }

  // 1. each model alone: a 4-second take (median of 3 after a warm-up) and a 30-second stretch (lesson building)
  const per = {};
  for (const name of ["embed", "large", "asr"]) {
    const s = await load(name);
    const run = (secs) => feed[kind[name]](s, secs);
    await run(4);
    const take = median([await timed(() => run(4)), await timed(() => run(4)), await timed(() => run(4))]);
    const long = await timed(() => run(30));
    per[name] = { take, long };
    log(`${label[name]}: 4 s of speech in ${take.toFixed(0)} ms, 30 s in ${long.toFixed(0)} ms`);
    result(`${label[name]}`, `${take.toFixed(0)} ms per 4 s take · ${long.toFixed(0)} ms per 30 s`);
    await s.release?.();
  }

  // 2. everything a take needs, all loaded at once (as the app would hold them): recogniser + aligner + sound model + tone model
  log("loading all four models at once (memory test)…");
  const all = { asr: await load("asr"), al: await load("large"), snd: await load("large"), embed: await load("embed") };
  const takes = [];
  for (let i = 0; i < 3; i++) {
    takes.push(await timed(async () => { await feed.asr(all.asr, 4); await feed.w2v(all.al, 4); await feed.w2v(all.snd, 4); await feed.w2v(all.embed, 4); }));
  }
  const takeMs = median(takes);
  log(`one 4 s take, all models: ${takeMs.toFixed(0)} ms (model time only)`);
  result("Score one take (4 s), all models loaded", `${(takeMs / 1000).toFixed(2)} s`);

  // 3. lesson building, sustained: recogniser + aligner over 5 minutes of audio in 30 s pieces, extrapolated to 20 min
  log("lesson building: 5 minutes of audio in 30 s pieces…");
  const t = now();
  for (let i = 0; i < 10; i++) { await feed.asr(all.asr, 30); await feed.w2v(all.al, 30); log(`  ${(i + 1) * 30} s of audio done`); }
  const fiveMin = now() - t;
  result("Build a lesson: 5 min of audio", `${(fiveMin / 1000).toFixed(0)} s`);
  result("Build a lesson: 20 min episode (×4)", `${(fiveMin * 4 / 60000).toFixed(1)} min`);
  for (const s of Object.values(all)) await s.release?.();
  log("done");
}
