<p align="center">
  <img src="desktop/ui/redpanda-aero.png" alt="Redpanda VC mascot" width="160">
</p>

# Redpanda VC

A desktop app and browser WebUI for RVC and Casual Voice Changer training, file conversion, and realtime voice conversion. The Tauri (Rust) app renders the interface in the system WebView and runs the Python voice engines locally.

## Desktop

Open `Redpanda.exe` on Windows or `redpanda` on Linux. The executable includes the UI and engine source. When launched standalone, it prepares its portable folder layout (`runtime/`, `assets/`, `datasets/`, `models/`, and `logs/`) beside the executable and bootstraps portable Python 3.12 + PyTorch via `uv` into `runtime/` on first run (auto-detecting NVIDIA CUDA or CPU). A build inside a source checkout reuses that checkout directly.

- **Inference** converts an audio file or folder using a `.pth` or `.cvc` voice.
- **Training** runs the full RVC workflow (preprocessing, feature extraction, training, and index building), or resumes an existing prepared run when the dataset folder is left blank. RVC precision (`fp16`, `bf16`, `fp32`) and model author live under **Advanced settings**. Switch to **Casual Voice Changer** to fine-tune a `.cvc` voice.
- **Realtime** accepts RVC and CVC voices with optional Xiph RNNoise input noise reduction. Choose input, output, and optional monitor devices. CVC uses a 48 kHz audio route and loads pitch, gain, gate, and index settings saved with the voice. Legacy `.causalvc.json` and `.nextvc.json` voices work with their accompanying weights.
- **TensorBoard** displays live training metrics directly inside the app, with a button to open the same dashboard in your browser.

The HiFi-GAN training weights download in the background at startup. The CVC pretrained base downloads when first needed. Existing CVC weights can be reused by setting `CAUSALVC_WEIGHTS` to the weights folder. The upstream project marks the default base and its fine-tunes for noncommercial use.

### Build from source

Use Rust stable, the Visual Studio C++ Build Tools, and WebView2 on Windows. On Debian or Ubuntu, install `build-essential`, `pkg-config`, `libwebkit2gtk-4.1-dev`, `libgtk-3-dev`, `libxdo-dev`, `libssl-dev`, and `librsvg2-dev` to build. The Linux Python worker also uses PortAudio for realtime and FFmpeg for audio tools.

~~~sh
cargo build --release --locked --manifest-path desktop/Cargo.toml
~~~

The binary is written to `desktop/target/release/`. For source checkouts, `run-install.bat` / `run-install.sh` set up the local Python environment.

## Browser WebUI

Run `run-web.bat` on Windows, `run-web.sh` on Linux/macOS, or `python core.py web` to serve the exact same interface in any browser (zero Gradio, pure Python HTTP + SSE + same-port TensorBoard proxy):

~~~sh
python core.py web --host 127.0.0.1 --port 7860
~~~

For remote servers, bind `--host 0.0.0.0 --no-open --auth user:password`. File browsing and downloads are automatically restricted to the workspace when serving remotely.

## CLI

Run these commands from the checkout or the desktop data folder with the installed environment active.

~~~sh
python core.py --help
python core.py train --help
python core.py infer --help
python -m causalvc.build --help
~~~

RVC runs live in `logs/`. CVC exports live in `logs/causalvc/` (standalone) or `runs/models/voice-<name>/voices/` (source checkout). Start training again under the same name with a higher step target to resume.

## Credits

- [Applio](https://github.com/IAHispano/Applio)
- [Causal Voice Changer v0.4.0](https://github.com/mojowozo/Causal-Voice-Changer/tree/50dce9236854fef17f550571413367217e235155) by Mojobones (MIT)
- [ContentVec](https://github.com/auspicious3000/contentvec/)
- [FireRedVAD](https://github.com/FireRedTeam/FireRedVAD)
- [Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- [RMVPE](https://github.com/yxlllc/RMVPE)
- [RNNoise](https://github.com/xiph/rnnoise)
- [RVC-Realtime-GUI](https://github.com/niel-blue/RVC-Realtime-GUI/)
