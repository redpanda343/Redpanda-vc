# Experimental rectified-flow training

This ports ShiroRVC's 44.1 kHz rectified-flow training recipe. The flow predicts
128-bin log mel spectrograms from content, F0, loudness, breathiness and speaker.
An existing OpenVPI NSF-HiFiGAN checkpoint renders training previews and is
recorded in the exported model. The vocoder is frozen; this trains the flow,
not a new vocoder. RVC latent NSF-HiFiGAN generator checkpoints do not accept
these mels and cannot be used as the vocoder.

Use an OpenVPI NSF-HiFiGAN checkpoint with 44100 Hz sample rate, hop 512,
FFT/window 2048, 128 mel bins, fmin 40 and fmax 16000. Its accompanying
`config.json`, when provided, must remain beside it. The loader checks the
mel settings and loads generator weights strictly. Other vocoders are rejected.

In the WebUI, open **Train > Rectified Flow**. Enter a new model name and
dataset folder, preprocess, extract features, then enter the NSF-HiFiGAN
checkpoint path and start training. The page shows the job status and live log.
The stop button ends the current rectified job; training resumes from the last
saved epoch, so unsaved steps are lost. The original trainer remains under
**Train > RVC**.

Alternatively, run these commands from the repository root with its Python environment active.
Use a new experiment name so existing RVC training data is preserved.

```powershell
python -m rvc.train.preprocess.preprocess logs/my-flow "C:\path\to\dataset" 44100 4 Automatic False False 0.0 10.0 0.3 none WAV
python -m rvc.train.extract.extract logs/my-flow rmvpe 4 0 44100 contentvec 0 v2 --rectified
python -m rvc.rectified.train_flow --model-name my-flow --vocoder "C:\path\to\nsf-hifigan\model.ckpt" --batch-size 4 --precision fp32 --epochs 100 --save-every 10
```

Extraction reuses the existing content and F0 extraction. The `--rectified`
option skips the RVC generator config and mute samples. Use `spin-v2` instead
of `contentvec` only when training from scratch or using a matching flow pretrain.

The default configuration is `rvc/configs/rectified/44100.json`. To customize a
run, copy it to `logs/my-flow/rectified_config.json` before training. The trainer
creates that copy automatically on its first run. It retains Shiro's full model
dimensions, pitch/time augmentation, shallow-flow objective, auxiliary decoder,
Muon/AdamW parameter split and speaker dropout. The WebUI reads the saved **Settings > Precision** selection at each start or
resume. Click **Update precision** to save your choice. FP16 uses CUDA autocast
and gradient scaling; BF16 uses CUDA autocast on supported GPUs. CPU runs and
unsupported BF16 GPUs fall back to FP32 with a log message. The CLI supports
`--precision fp32`, `fp16`, or `bf16`, and defaults to FP32. Model weights,
optimizer state, Muon matrix operations, held-out evaluation and audio previews
remain FP32 for stability. TF32 stays disabled. Changing the setting does not
alter a job already running.

Training uses a single device, selected with `--device cuda:0` or `--device cpu`.
`flow.segment_frames`, `flow.num_workers` and batch size control memory use.
The default crop is 400 mel frames. Short clips are padded and masked.

`logs/my-flow/flow/checkpoint.pth` contains the model, optimizer, EMA, config,
epoch and step for automatic resume at the next epoch. Increase `--epochs` to
extend a run. Use `--fresh` to deliberately restart the same experiment;
this replaces its resumable checkpoint. Use a new experiment to preserve it.
`--pretrained-flow path.pth` starts from a compatible Shiro flow export or
training checkpoint. It retains Shiro's single-speaker fine-tuning freeze policy.

EMA exports are saved as `my-flow_flow_<epoch>e_<step>s.pth`. TensorBoard in
the same directory records flow loss, auxiliary mel loss, gradient norm, learning
rate, held-out loss when configured, and NSF-HiFiGAN audio previews. Exports
retain Shiro's rectified-flow format. The ordinary RVC conversion UI cannot
load these exports; conversion integration is outside this training-only port.
