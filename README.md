<p align="center">
  <img src="assets/applio_mascot.png" alt="Applio mascot" width="180">
</p>

# Applio Fork

This project is a fork of [Applio](https://github.com/IAHispano/Applio), with changes focused on dataset preprocessing, inference, training, normalization, and a cleaner WebUI experience.

New Rectified Flow experiments use the `standard-v2` preset: a DiffSinger-style
transformer condition encoder with ContentVec input, LYNXNet2 ATanGLU backbone,
shallow flow and ConvNeXt auxiliary mel decoder. Muon/AdamW uses betas
`(0.9, 0.98)`, Muon weight decay `0.1`, AdamW weight decay `0`.
Random pitch and time augmentation keep their key-shift and speed embeddings.

 ## Credits

- [Applio](https://github.com/IAHispano/Applio)
- [FireRedVAD](https://github.com/FireRedTeam/FireRedVAD)
- [UTMOSV2](https://github.com/sarulab-speech/UTMOSv2)
- [ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN)
- [DiffSinger](https://github.com/openvpi/DiffSinger)
