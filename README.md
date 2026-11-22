<p align="center">
  <img src="assets/applio_mascot.png" alt="Applio mascot" width="180">
</p>

# redpanda-vc

A fork of [Applio](https://github.com/IAHispano/Applio) for training RVC voice models and running batch and realtime voice conversion. The desktop realtime GUI launches with `run-realtime-gui.bat`.

The WebUI realtime tab supports multiple GPUs through isolated inference workers. Set Inference Device to Automatic to assign each new stream to an available GPU, or select a specific GPU. Separate browser tabs or users can run streams on different GPUs. Each stream stays on one GPU to preserve its audio state. Additional GPUs increase concurrent capacity.

Browser realtime audio uses a binary WebSocket connection, including with `--share`, and falls back to HTTP streaming when WebSockets are unavailable. Audio queues are bounded to avoid accumulating delay on slow connections. Install the updated requirements before launching.

## Credits

- [Applio](https://github.com/IAHispano/Applio)
- [ContentVec](https://github.com/auspicious3000/contentvec/)
- [FireRedVAD](https://github.com/FireRedTeam/FireRedVAD)
- [Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- [RMVPE](https://github.com/yxlllc/RMVPE)
- [RVC-Realtime-GUI](https://github.com/niel-blue/RVC-Realtime-GUI/)
- [tgm_hifigan](https://github.com/mrtigermeat/tgm_hifigan) by tigermeat (CC BY-NC 4.0)
