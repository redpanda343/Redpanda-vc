<p align="center">
  <img src="assets/applio_mascot.png" alt="Applio mascot" width="180">
</p>

# redpanda-vc

A fork of [Applio](https://github.com/IAHispano/Applio) for training RVC voice models and running batch and realtime voice conversion. The desktop realtime GUI launches with `run-realtime-gui.bat`.

The WebUI realtime tab supports multiple GPUs through isolated inference workers. Set Inference Device to Automatic to assign each new stream to an available GPU, or select a specific GPU. Separate browser tabs or users can run streams on different GPUs. Each stream stays on one GPU to preserve its audio state. Additional GPUs increase concurrent capacity.

Browser realtime audio uses a binary WebSocket connection, including with `--share`, and falls back to HTTP streaming when WebSockets are unavailable. Audio queues are bounded to avoid accumulating delay on slow connections. Install the updated requirements before launching.

## Causal Voice Changer

**Training → Causal Voice Changer** accepts an audio file or folder and fine-tunes the upstream decoder. Set a voice name and the number of steps. Start again with the same name and a higher step target to resume. Exported `.cvc` voices are written to `runs/models/voice-<name>/voices/`.

In the browser's **Real-time** tab, refresh the Causal Voice Changer voice list or paste a voice path. In the desktop realtime GUI, browse for the voice file. Both accept `.cvc`, `.causalvc.json` and `.nextvc.json` voices and load their saved pitch, gain, gate and index settings. Use **Pitch range** to adjust the Hz offset, floor and ceiling while running. Causal Voice Changer uses its own pitch tracker, encoder and streaming state at 48 kHz. The desktop audio devices need to support 48 kHz.

The default pretrained files download on first use. To reuse an existing Causal Voice Changer weights folder, set `CAUSALVC_WEIGHTS` to its path. The upstream project marks the default base and its fine-tunes for noncommercial use.

## Credits

- [Applio](https://github.com/IAHispano/Applio)
- [Causal Voice Changer v0.4.0](https://github.com/mojowozo/Causal-Voice-Changer/tree/50dce9236854fef17f550571413367217e235155) by Mojobones (MIT)
- [ContentVec](https://github.com/auspicious3000/contentvec/)
- [FireRedVAD](https://github.com/FireRedTeam/FireRedVAD)
- [Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- [RMVPE](https://github.com/yxlllc/RMVPE)
- [RVC-Realtime-GUI](https://github.com/niel-blue/RVC-Realtime-GUI/)
- [tgm_hifigan](https://github.com/mrtigermeat/tgm_hifigan) by tigermeat (CC BY-NC 4.0)
