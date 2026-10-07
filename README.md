<p align="center">
  <img src="assets/applio_mascot.png" alt="Applio mascot" width="180">
</p>

# Applio Fork

This project is a fork of [Applio](https://github.com/IAHispano/Applio), with changes focused on dataset preprocessing, inference, training, normalization, and a cleaner WebUI experience.

Rectified Flow follows DiffSinger's acoustic model with ContentVec in place of
phonemes: a transformer content encoder over ContentVec frames at their native
50 Hz, expanded to mel frames before speaker, pitch, key-shift and speed
conditioning, a LYNXNet2 ATanGLU backbone, shallow flow and a ConvNeXt auxiliary
mel decoder. Muon/AdamW uses betas `(0.9, 0.999)`, Muon weight decay `0.1`,
AdamW weight decay `0`, and no EMA. Random pitch and time augmentation keep their
key-shift and speed embeddings. `flow.model.use_spk_id` defaults to `false`,
matching DiffSinger; set it to `true` before starting an experiment to enable
speaker conditioning.

F0 is extracted when training binarizes the dataset, exactly like DiffSinger's
acoustic binarizer, with the **Pitch extractor** chosen next to the content
embedder at extraction (`flow.pitch_extractor`; `--pitch-extractor` overrides it
on the command line). `parselmouth` uses
autocorrelation on the 44.1 kHz audio at the mel hop, 65-1100 Hz, voicing
threshold 0.6. `rmvpe` uses DiffSinger's RMVPE code (from
[yxlllc/RMVPE](https://github.com/yxlllc/RMVPE)) with the 230917 model, which
the app downloads to `rvc/models/predictors/rmvpe.pt`. It runs at 16 kHz and is
resampled to the mel hop. Unvoiced frames are interpolated in log frequency.
Time stretching re-extracts F0 at the stretched hop and pitch shifting scales
it. Like DiffSinger, binarization skips clips with no voiced frames. The
extractor is fixed when an experiment starts. File conversion with
`parselmouth` or `rmvpe` reproduces the training extraction on the input file at
the model's sample rate, not on the 16 kHz copy used for content features; other
methods are interpolated and resampled to the mel hop. Realtime follows the
selected pitch method the same way, extracting Parselmouth and RMVPE F0 from the
full-rate input stream.

As in DiffSinger, the vocoder gets the same interpolated F0 as the flow, and
conversion does not post-process unvoiced frames. The interpolated F0 does not
tell the flow which frames are breaths, noise or whispers; models trained with
breathiness/voicing conditioning learn that from those curves, the way RVC
learns it from zero F0. Models trained without it can render unvoiced input as
sung vowels.

**Breathiness / voicing conditioning** (training tab, on by default;
`--variance-embeds` / `--no-variance-embeds` on the command line) conditions the
flow on DiffSinger's breathiness and voicing
(`use_breathiness_embed` and `use_voicing_embed`), which tell the model how
noisy and how voiced each frame of the source is, the job DiffSinger's AP and SP
phonemes do. Binarization splits each clip into harmonic and aperiodic parts with
DiffSinger's harmonic-noise separator (`flow.hnsep`: `vr`, the default, or
`world`), takes the RMS of each in dB at the mel hop and smooths it with a 60 ms
sine window (`breathiness_smooth_width`, `voicing_smooth_width`). Each curve is
scaled by 1/96 into its own linear embedding. Pitch-shifted copies keep the
original curves and time-stretched copies resample them, as in DiffSinger.
Conversion extracts the same curves from the full-rate source before pitch
shifting. VR runs in 15 s windows with 1 s crossfades so long files fit in GPU
memory, and realtime separates only the newest audio with 1 s (VR) or 0.3 s
(WORLD) of context and reuses the rest. The app downloads the VR model
(`hnsep_240512`) to `rvc/models/predictors/hnsep/vr` when first needed.
Experiments and checkpoints made before this keep both embeddings off; turn the
option off to resume them. Turning it off matches DiffSinger's default. Enabling
it while fine-tuning a checkpoint without it starts both embeddings at zero, so
the fine-tune begins from the checkpoint's exact output.

The flow samples with RK2 (DiffSinger's midpoint `rk2`) for 10 steps, the same
20 network evaluations as Euler at 20 steps but closer to the converged
trajectory, which keeps breath noise from coming out over-smoothed. Checkpoints
saved with the old Euler/20 default use RK2/10 when loaded. The realtime Flow
steps setting counts RK2 steps, each costing two evaluations.

Realtime can run the flow on the newest audio only (Flow window, on by default
in the realtime GUI). The content encoder still sees the whole context, as
seed-vc does, but the aux decoder and the flow steps run on the frames
the vocoder needs plus 0.5 s. The backbone is convolutional, so frames further back
do not change the output; on a GTX 1660 Ti this took a realtime-preset flow from
203 ms to 141 ms per 250 ms block. Turning it off runs the flow over the full
context as before.

Shortcut training is optional (**Shortcut (few-step) flow** in the training tab,
`--shortcut` on the command line; off by default, which keeps the DiffSinger flow
unchanged). It follows Frans et al., "One Step Diffusion via Shortcut Models"
(2024) and their reference code: the backbone also takes the step size as log2 of
the step count through its own embedding, initialised to zero so a converted
checkpoint starts out identical. One in eight clips per batch (`shortcut_bootstrap_every`)
is trained to make one jump of 2d equal to two jumps of d, with both jumps taken
by an EMA copy of the model (`shortcut_ema` 0.999) and the remaining clips use the
usual flow loss at the finest of 128 steps (`shortcut_steps`). Step levels are drawn
uniformly per clip rather than spread over the batch as in the reference code,
which with small batches trained only the one-step jump (kvfrans/shortcut-models#11).
The step grid covers the shallow-flow range from `t_start` to 1. Shortcut flows
sample with Euler in a power of two of steps, 8 by default, and down to 1 in the
realtime Flow steps setting; each update costs about 15% more. New shortcut flows
trained from scratch decay the learning rate twice as slowly (`decay_step` 10000
instead of 5000) over twice as many updates (`max_updates` 200000), so they reach
at 200000 updates the learning rate a standard flow reaches at 100000, about
6.9e-6. An existing flow can be fine-tuned into a shortcut flow by enabling
Shortcut with a pretrained checkpoint; fine-tunes keep the fine-tuning schedule.

Training binarizes the dataset like DiffSinger: originals and their augmented
copies go into `logs/<model>/binary/train.data`, held-out clips into
`valid.data`, each with a `.meta` file of clip lengths used for batching. The
binary data is rebuilt only when the clips or the data and augmentation
settings change. ContentVec features stay FP32. As in DiffSinger, held-out
clips are binarized in the main process, originals use
`flow.augmentation_workers` worker processes (0 processes them in the main
process, the default), and augmented copies are computed in the main process.

The recipe defaults live in `rvc/rectified/config.py` and are written to
`logs/<model>/rectified_config.json` when training starts. Edit the experiment
config to change the model, optimizer, scheduler, batching, validation or
checkpoint policy. Architecture changes require a new experiment. Training
budgets, precision, validation and retention settings can change on resume.
Rectified Flow uses the shared precision setting under Settings > Training >
Precision. Command-line training uses `flow.precision` unless `--precision` is
provided. Blank numeric overrides in the WebUI use the experiment config.
Fine-tuning from a voice export starts from a lower learning rate with warmup
and disabled augmentation.

Rectified Flow uses `lightning.pytorch` with `lightning~=2.3.0`, matching the local
DiffSinger trainer. Lightning owns automatic backward, AMP loss scaling, gradient
clipping, gradient accumulation, optimizer steps, per-step LR scheduling and DDP.
The shared `fp32`, `fp16` and `bf16` choices map to `32-true`, `16-mixed` and
`bf16-mixed`. Validation loss and previews run in FP32, including the initial
sanity validation. Training logs use `training/*` and `validation/*` in
`logs/<model>/flow/lightning_logs/latest`.
The config exposes accelerator, devices, nodes, strategy, accumulation, sanity
validation, separate validation batch limits and length sorting. Multi-GPU
training uses Lightning DDP with NCCL on Linux and Gloo on Windows/CPU.
Experiments default to 100000 Lightning training steps (200000 for new shortcut
flows), FP16 mixed precision,
validation/preview/checkpoints every 4000 updates, up to 10 validation plots,
vocoder previews and 8 recent resumable checkpoints plus voice exports.
Checkpoints saved from step 60000 at 10000-step multiples are retained
permanently. Resumable training states are `flow/model_ckpt_steps_<step>.ckpt`
and `flow/last.ckpt`; voice exports are `_flow_<epoch>e_<step>s.pth` files.
`--fresh` archives previous checkpoints and exports inside the flow folder.
As in DiffSinger, Lightning counts optimizer attempts when FP16 overflow skips
an update. Reflow inference uses `flow.model.sampling_steps`.

The training tab's optional **Realtime** switch (`--preset realtime`) starts a
new experiment with a narrower, deeper model: 256 hidden channels and 6 content
encoder layers, a 512-channel LYNXNet2 backbone with 12 layers and a 384-channel
aux decoder with 8 layers. It has about 37M parameters instead of 67M. The
switch must match the experiment or pretrained checkpoint when resuming or
fine-tuning.

The training tab's optional **Fused Linear + SoftSignGLU kernels** switch
uses DiffSinger's Triton forward and elementwise backward kernels with cuBLAS
gradient matrix multiplications, and is off by default. Enabling it when an
experiment starts sets `flow.model.backbone_args.glu_type` to `softsign_glu`, which
is saved with the experiment and checkpoints. Experiments and pretrained
checkpoints that use another activation refuse the switch instead of changing
their activation. Disabling the switch later turns off fused kernels and keeps the
saved activation. Evaluation and inference use eager kernels.
Fused CUDA FP16/BF16 training requires a working Triton installation; CPU/FP32 and
unsupported GPUs use the eager path. The port in `rvc/rectified/kernels` is
adapted from [DiffSinger](https://github.com/openvpi/DiffSinger) under Apache 2.0;
its license is included in that directory.

 ## Credits

- [Applio](https://github.com/IAHispano/Applio)
- [FireRedVAD](https://github.com/FireRedTeam/FireRedVAD)
- [ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN)
- [DiffSinger](https://github.com/openvpi/DiffSinger)
