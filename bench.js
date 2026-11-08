// Spike (throwaway): how fast do the app's speech models run on this device?
// Shared by the browser page (index.html) and the Node check (node-check.mjs). Inputs are synthetic (noise of the
// right length): the time a model takes depends on the input's length, not on what is said.
// The aligner and the sound model are both wav2vec2-large (same size, same layers), so one file stands in for both.
// The test is split into stages so the page can reload between them: a browser never gives back the memory a
// model used, so running everything in one go measures the page, not the phone.

const SR = 16000;
const noise = (secs) => { const a = new Float32Array(Math.round(secs * SR)); for (let i = 0; i < a.length; i++) a[i] = (Math.random() - 0.5) * 0.1; return a; };
const asrFeats = (secs) => { const t = Math.max(1, Math.round(secs * 100 / 6)); const a = new Float32Array(t * 560); for (let i = 0; i < a.length; i++) a[i] = Math.random() - 0.5; return [a, t]; };
const median = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
const now = () => (globalThis.performance ? performance.now() : Date.now());

export const STAGES = [
  { id: "embed", label: "Tone model" },
  { id: "large", label: "Aligner / sound model (each)" },
  { id: "asr", label: "Recogniser" },
  { id: "take", label: "Score one take (4 s), all four models loaded" },
  { id: "lesson", label: "Build a lesson (recogniser + aligner loaded)" },
];

export async function runStage(id, { ort, getModel, log, result }) {
  const opts = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
  const feed = {
    w2v: (s, secs) => s.run({ wav: new ort.Tensor("float32", noise(secs), [1, Math.round(secs * SR)]) }),
    asr: (s, secs) => { const [a, t] = asrFeats(secs); return s.run({ speech: new ort.Tensor("float32", a, [1, t, 560]), speech_lengths: new ort.Tensor("int32", Int32Array.from([t]), [1]) }); },
  };
  const kind = { embed: "w2v", large: "w2v", asr: "asr" };
  const timed = async (fn) => { const t = now(); await fn(); return now() - t; };
  const label = STAGES.find((s) => s.id === id).label;

  async function load(name) {   // the model's bytes are dropped as soon as the session holds the model
    const t = now();
    const s = await ort.InferenceSession.create(await getModel(name), opts);
    log(`${name}: ready in ${((now() - t) / 1000).toFixed(1)} s`);
    return s;
  }

  if (kind[id]) {   // one model alone: a 4-second take (median of 3 after a warm-up) and a 30-second stretch
    const s = await load(id);
    const run = (secs) => feed[kind[id]](s, secs);
    await run(4);
    const take = median([await timed(() => run(4)), await timed(() => run(4)), await timed(() => run(4))]);
    log(`${label}: 4 s take ${take.toFixed(0)} ms; now 30 s…`);
    const long = await timed(() => run(30));
    log(`${label}: 30 s in ${long.toFixed(0)} ms`);
    result(label, `${take.toFixed(0)} ms per 4 s take · ${long.toFixed(0)} ms per 30 s`);
    await s.release?.();
  } else if (id === "take") {   // everything a take needs, held at once as the app would
    const all = { asr: await load("asr"), al: await load("large"), snd: await load("large"), embed: await load("embed") };
    log("all four models loaded");
    await feed.asr(all.asr, 4); await feed.w2v(all.al, 4); await feed.w2v(all.snd, 4); await feed.w2v(all.embed, 4);
    const takes = [];
    for (let i = 0; i < 3; i++) {
      takes.push(await timed(async () => { await feed.asr(all.asr, 4); await feed.w2v(all.al, 4); await feed.w2v(all.snd, 4); await feed.w2v(all.embed, 4); }));
    }
    result(label, `${(median(takes) / 1000).toFixed(2)} s`);
    for (const s of Object.values(all)) await s.release?.();
  } else if (id === "lesson") {   // lesson building, sustained: recogniser + aligner over 5 min of audio in 30 s pieces
    const asr = await load("asr"), al = await load("large");
    const t = now();
    for (let i = 0; i < 10; i++) { await feed.asr(asr, 30); await feed.w2v(al, 30); log(`  ${(i + 1) * 30} s of audio done`); }
    const fiveMin = now() - t;
    result("Build a lesson: 5 min of audio", `${(fiveMin / 1000).toFixed(0)} s`);
    result("Build a lesson: 20 min episode (×4)", `${(fiveMin * 4 / 60000).toFixed(1)} min`);
    await asr.release?.(); await al.release?.();
  }
}
