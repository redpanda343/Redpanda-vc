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

F0 is extracted with Parselmouth exactly like DiffSinger's acoustic binarizer:
autocorrelation on the 44.1 kHz audio at the mel hop, 65-1100 Hz, voicing
threshold 0.6, with unvoiced frames interpolated in log frequency. Time
stretching re-extracts F0 at the stretched hop and pitch shifting scales it.
The same F0 feeds the flow and the vocoder during training, file conversion and
realtime. Like DiffSinger, extraction skips clips with no voiced frames.

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
Experiments default to 100000 Lightning training steps, FP16 mixed precision,
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
gradient matrix multiplications, and is off by default. Enabling it overrides
`flow.model.backbone_args.glu_type` with `softsign_glu`, including for resume and
fine-tuning. The effective activation is saved with the experiment and
checkpoints. Disabling the switch later turns off fused kernels and keeps the
saved activation. Evaluation and inference use eager kernels.
Fused CUDA FP16/BF16 training requires a working Triton installation; CPU/FP32 and
unsupported GPUs use the eager path. The port in `rvc/rectified/kernels` is
adapted from [DiffSinger](https://github.com/openvpi/DiffSinger) under Apache 2.0;
its license is included in that directory.

 ## Credits

- [Applio](https://github.com/IAHispano/Applio)
- [FireRedVAD](https://github.com/FireRedTeam/FireRedVAD)
- [UTMOSV2](https://github.com/sarulab-speech/UTMOSv2)
- [ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN)
- [DiffSinger](https://github.com/openvpi/DiffSinger)
