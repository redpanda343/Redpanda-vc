<p align="center">
  <img src="assets/applio_mascot.png" alt="Applio mascot" width="180">
</p>

# Applio Fork

This project is a fork of [Applio](https://github.com/IAHispano/Applio), with changes focused on dataset preprocessing, inference, training, normalization, and a cleaner WebUI experience.

New Rectified Flow experiments use the `standard-v3` preset: a DiffSinger-style
transformer condition encoder with ContentVec input, LYNXNet2 SoftSignGLU backbone,
shallow flow and ConvNeXt auxiliary mel decoder. Muon/AdamW uses betas
`(0.9, 0.98)`, Muon weight decay `0.1`, AdamW weight decay `0`.
Random pitch and time augmentation keep their key-shift and speed embeddings.
Existing `standard-v1` and `standard-v2` experiments retain ATanGLU.

The training tab's optional **Fused Linear + SoftSignGLU kernels** switch
replaces `torch.compile` for Rectified Flow. It uses DiffSinger's Triton forward
and elementwise backward kernels with cuBLAS gradient matrix multiplications,
and is off by default. Evaluation and inference use eager SoftSignGLU.
CUDA FP16/BF16 training requires a working Triton installation; CPU/FP32 and
unsupported GPUs use the eager path. The port in `rvc/rectified/kernels` is
adapted from [DiffSinger](https://github.com/openvpi/DiffSinger) under Apache 2.0;
its license is included in that directory.

 ## Credits

- [Applio](https://github.com/IAHispano/Applio)
- [FireRedVAD](https://github.com/FireRedTeam/FireRedVAD)
- [UTMOSV2](https://github.com/sarulab-speech/UTMOSv2)
- [ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN)
- [DiffSinger](https://github.com/openvpi/DiffSinger)
