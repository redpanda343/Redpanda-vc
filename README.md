<p align="center">
  <img src="assets/applio_mascot.png" alt="Applio mascot" width="180">
</p>

# Applio Fork

This project is a fork of [Applio](https://github.com/IAHispano/Applio), with changes focused on dataset preprocessing, inference, training, normalization, and a cleaner WebUI experience.

For manual or cloud setup, install `requirements.txt`, then install SwiftF0
separately with `python -m pip install --no-deps swift-f0==0.3.0`. SwiftF0 is
intentionally excluded from `requirements.txt` because its upstream dependency
on CPU `onnxruntime` can overwrite `onnxruntime-gpu`. The standard installers
already use this separate installation. When Swift is selected, a missing
SwiftF0 package is installed with `--no-deps` in the current Python environment
before pitch extraction workers start. CUDA support is checked first; a CPU-only
ONNX Runtime installation is reported rather than modified automatically.

New Rectified Flow experiments use the `standard-v2` preset: a DiffSinger-style
transformer condition encoder with ContentVec input, LYNXNet2 ATanGLU backbone,
shallow flow and ConvNeXt auxiliary mel decoder. Muon/AdamW uses betas
`(0.9, 0.98)`, Muon weight decay `0.1`, AdamW weight decay `0`, and no EMA.
Random pitch and time augmentation keep their key-shift and speed embeddings.
Existing `standard-v1` experiments retain their saved architecture. EMA is removed
from all training runs.
Start a new experiment to use the new architecture.

 ## Credits

- [Applio](https://github.com/IAHispano/Applio)
- [FireRedVAD](https://github.com/FireRedTeam/FireRedVAD)
- [UTMOSV2](https://github.com/sarulab-speech/UTMOSv2)
- [ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN)
- [DiffSinger](https://github.com/openvpi/DiffSinger)
