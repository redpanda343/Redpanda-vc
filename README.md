<p align="center">
  <img src="assets/applio_mascot.png" alt="Applio mascot" width="180">
</p>

# redpanda-vc

A fork of [Applio](https://github.com/IAHispano/Applio) that adds the option to train Rectified Flow models that use ContentVec, Beatrice models, and NSF-HiFiGAN / PC-NSF-HiFiGAN vocoders.

The Rectified Flow model uses the same architecture as [DiffSinger](https://github.com/openvpi/DiffSinger)'s acoustic model, with ContentVec features in place of phonemes.

The realtime GUI (`run-realtime-gui.bat`) supports RVC, Beatrice and Rectified Flow models.

The WebUI realtime tab supports multiple GPUs through isolated inference workers. Set Inference Device to Automatic to assign each new stream to an available GPU, or select a specific GPU. Separate browser tabs or users can run streams on different GPUs. Each stream stays on one GPU to preserve its audio state; additional GPUs increase concurrent capacity. Beatrice uses the CPU.

Browser realtime audio uses a binary WebSocket connection, including with `--share`, and falls back to HTTP streaming when WebSockets are unavailable. Audio queues are bounded to avoid accumulating delay on slow connections. Install the updated requirements before launching.

## Credits

- [Applio](https://github.com/IAHispano/Applio)
- [BigVGAN](https://github.com/NVIDIA/BigVGAN) by NVIDIA (MIT)
- [Beatrice Trainer](https://huggingface.co/fierce-cats/beatrice-trainer)
- [ContentVec](https://github.com/auspicious3000/contentvec/)
- [DiffSinger](https://github.com/openvpi/DiffSinger)
- [FireRedVAD](https://github.com/FireRedTeam/FireRedVAD)
- [MeanVC2](https://github.com/ASLP-lab/MeanVC2) by ASLP-lab for the optional MeanFlow training method.
- [NSF-BigVGAN](https://github.com/PlayVoice/BigVGAN) by PlayVoice (MIT)
- [OpenVPI PC-NSF-HiFiGAN](https://github.com/openvpi/vocoders)
- [Pupu-Vocoder](https://github.com/sizigi/AliasingFreeNeuralAudioSynthesis) by Yicheng Gu et al. (MIT), with the sinc filters of [julius](https://github.com/adefossez/julius) (MIT)
- [Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- [RMVPE](https://github.com/yxlllc/RMVPE)
- [RVC-Realtime-GUI](https://github.com/niel-blue/RVC-Realtime-GUI/)
- [ShiromiyaG](https://github.com/ShiromiyaG), for the original idea of the rectified flow models
- [SingingVocoders](https://github.com/openvpi/SingingVocoders)
- [tgm_hifigan](https://github.com/mrtigermeat/tgm_hifigan) by tigermeat (CC BY-NC 4.0)
- [vocal-remover](https://github.com/yxlllc/vocal-remover)
