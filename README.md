<p align="center">
  <img src="assets/applio_mascot.png" alt="Applio mascot" width="180">
</p>

# Applio Fork

This project is a fork of [Applio](https://github.com/IAHispano/Applio), with changes focused on dataset preprocessing, inference, training, normalization, and a cleaner WebUI experience.

New Rectified Flow experiments use the `standard-v5` preset: a DiffSinger-style
transformer condition encoder with ContentVec input, LYNXNet2 ATanGLU backbone,
shallow flow and ConvNeXt auxiliary mel decoder. Muon/AdamW uses betas
`(0.9, 0.98)`, Muon weight decay `0.1`, AdamW weight decay `0`, and no EMA.
Random pitch and time augmentation keep their key-shift and speed embeddings.
New experiments default to `flow.model.use_spk_id: false`, matching DiffSinger.
These models have no speaker embedding table and ignore the target speaker ID.
Set `flow.model.use_spk_id` to `true` before starting a new experiment to enable
speaker conditioning. Fine-tuning and resume retain the checkpoint's setting.
Existing `standard-v1` through `standard-v4` experiments retain their saved
architecture, activation and speaker setting.

All recipe settings are exposed in `rvc/configs/rectified/44100_standard.json`
and copied to `logs/<model>/rectified_config.json`. Edit the experiment config to
change the model, optimizer, scheduler, batching, validation or checkpoint policy.
Architecture changes require a new experiment. Training budgets, precision,
validation and retention settings can change on resume. The WebUI precision
selector supports FP32, FP16 and BF16; blank numeric overrides use the config.
New experiments default to 100000 successful updates, FP16 mixed precision,
validation/preview/checkpoints every 4000 updates, up to 10 validation plots,
vocoder previews and 8 recent resumable checkpoints plus voice exports.
Checkpoints saved from step 60000 at
10000-step multiples are retained permanently. The fine-tuning template retains
its lower learning rate, warmup and disabled augmentation.
DiffSinger's `K_step` fields are DDPM-only; reflow inference uses
`flow.model.sampling_steps` instead. Phoneme stretch embeddings require phoneme
alignment and have no equivalent for frame-level ContentVec.

The training tab's optional **Fused Linear + SoftSignGLU kernels** switch
replaces `torch.compile` for Rectified Flow. It uses DiffSinger's Triton forward
and elementwise backward kernels with cuBLAS gradient matrix multiplications,
and is off by default. Set `flow.model.backbone_args.glu_type` to
`softsign_glu` for a new experiment before enabling it. Evaluation and inference
use eager kernels.
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
