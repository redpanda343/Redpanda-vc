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

 ## Credits

- [Applio](https://github.com/IAHispano/Applio)
- [FireRedVAD](https://github.com/FireRedTeam/FireRedVAD)
- [UTMOSV2](https://github.com/sarulab-speech/UTMOSv2)
- [ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN)
