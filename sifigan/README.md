SiFiGAN offline inference
========================

The inference generator is adapted from the official [SiFiGAN project](https://github.com/chomeyama/SiFiGAN), revision `cdb20daeb4da4a126077819091706ee90522c7a8`. Its MIT license is included in LICENSE.

Put a standard SiFiGAN checkpoint (`.pkl`, `.pth` or `.ckpt`) and its matching statistics (`.joblib`) together under `models/pretraineds/sifigan`, `models/sifigan/custom` or `logs`. Refresh the existing Inference tab and select the checkpoint in Voice Model / Vocoder. Index, embedder and speaker controls are hidden automatically. RMVPE, parselmouth, pitch adjustment, splitting, batch processing and audio export use the existing inference controls.

The default configuration matches the official 24 kHz LibriTTS-R + NUS-48E model. The official model and statistics archive is linked in the upstream README. For another standard SiFiGAN model, put its resolved training configuration in `sifigan.yaml` or `config.yaml` beside the checkpoint. A `stats` entry can select the matching statistics file. Keep `alpha`, `fft_size` and feature dimensions consistent with its feature extraction settings. Unsupported generator variants and mismatched weights are rejected.

Audio is analyzed with WORLD CheapTrick and D4C using the selected pitch extractor. Mel cepstra and coded aperiodicity are standardized with the saved statistics before synthesis. Pitch adjustment changes the excitation and pitch-dependent dilation while retaining the analyzed spectral envelope. The mel-cepstrum conversion uses the SPTK frequency transform; its BSD license is included in LICENSE-SPTK. The transform matrix construction is adapted from [diffsptk](https://github.com/sp-nitech/diffsptk); its Apache license and attribution are in LICENSE-diffsptk.

This integration is available only for standalone offline inference.
