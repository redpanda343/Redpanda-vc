<p align="center">
  <img src="assets/applio_mascot.png" alt="Applio mascot" width="180">
</p>

# redpanda-vc

A fork of [Applio](https://github.com/IAHispano/Applio) that adds the option to train Rectified Flow models that use ContentVec, Beatrice models, and NSF-HiFiGAN / PC-NSF-HiFiGAN vocoders.

The Rectified Flow model uses the same architecture as [DiffSinger](https://github.com/openvpi/DiffSinger)'s acoustic model, with ContentVec features in place of phonemes.

The realtime GUI (`run-realtime-gui.bat`) supports RVC, Beatrice and Rectified Flow models.

## Credits

- [DiffSinger](https://github.com/openvpi/DiffSinger)
- [MeanVC2](https://github.com/ASLP-lab/MeanVC2) by ASLP-lab for the optional MeanFlow training method.
- [RVC-Realtime-GUI](https://github.com/niel-blue/RVC-Realtime-GUI/)
- [Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- [Applio](https://github.com/IAHispano/Applio)
- [FireRedVAD](https://github.com/FireRedTeam/FireRedVAD)
- [Beatrice Trainer](https://huggingface.co/fierce-cats/beatrice-trainer)
- [OpenVPI PC-NSF-HiFiGAN](https://github.com/openvpi/vocoders)
- [SingingVocoders](https://github.com/openvpi/SingingVocoders)
- [tgm_hifigan](https://github.com/mrtigermeat/tgm_hifigan) by tigermeat (CC BY-NC 4.0)
- [ContentVec](https://github.com/auspicious3000/contentvec/)
- [RMVPE](https://github.com/yxlllc/RMVPE)
- [vocal-remover](https://github.com/yxlllc/vocal-remover)
