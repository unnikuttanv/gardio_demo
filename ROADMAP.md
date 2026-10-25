# Roadmap

Planned work, roughly in priority order. Effort estimates are rough.

## Now — signal-processing depth

- [ ] **Validate Pan-Tompkins against MIT-BIH annotations** (~5 h): script that runs QRS detection on the public MIT-BIH records and reports sensitivity / positive predictivity per record; add a regression test on a small cached subset.
- [ ] **Frequency-domain HRV** (~4 h): LF, HF, LF/HF via Lomb-Scargle on the RR series, added to `FeatureSet`.
- [ ] **Signal-quality index** (~4 h): flag noisy / flat-line segments before feature extraction so the report can say "insufficient quality" instead of reporting bad numbers.
- [ ] **Rhythm irregularity features** (~5 h): RR-interval irregularity metrics (e.g. RMSSD-based and entropy-based) surfaced as observations, not diagnoses.

## Next — reliability

- [ ] **Agent / report tests** (~3 h): unit tests for `ecg_llm/report/` using the deterministic mock LLM — schema conformance of `EcgReport`, trend summaries, prompt construction.
- [ ] **CI** (~2 h): GitHub Actions running ruff + pytest (fully offline) on every push.
- [ ] **API hardening** (~3 h): input limits on `/analyze`, structured error responses, OpenAPI examples.

## Later — interfaces & deployment

- [ ] **Cohort trends view in Gradio** (~4 h): per-user multi-day HRV trend charts next to the generated report.
- [ ] **Hugging Face Space deployment** (~3 h): publish the Gradio demo.
- [ ] **Multi-lead support** (~6 h): accept 12-lead records and pick / fuse leads for detection.
- [ ] Docs: annotated walkthrough of the pipeline with plots at each stage.
