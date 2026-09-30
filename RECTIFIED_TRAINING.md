# Experimental rectified-flow training

This ports ShiroRVC's 44.1 kHz rectified-flow training recipe. The flow predicts
128-bin log mel spectrograms from content, F0, loudness, breathiness and speaker.
An optional OpenVPI NSF-HiFiGAN checkpoint renders audio previews and is
recorded in the exported model. Without one, previews show mel images only. The vocoder is frozen; this trains the flow,
not a new vocoder. RVC latent NSF-HiFiGAN generator checkpoints do not accept
these mels and cannot be used as the vocoder.

Use an OpenVPI NSF-HiFiGAN checkpoint with 44100 Hz sample rate, hop 512,
FFT/window 2048, 128 mel bins, fmin 40 and fmax 16000. Its accompanying
`config.json`, when provided, must remain beside it. The loader checks the
mel settings and loads generator weights strictly. Other vocoders are rejected.

In the WebUI, open **Train > Rectified Flow**. Enter a new model name and
dataset folder, preprocess, extract features, then enter the NSF-HiFiGAN
checkpoint path if audio previews are wanted, and start training. The page shows the job status and live log.
The stop button ends the current rectified job; training resumes from the last
saved epoch, so unsaved steps are lost. The original trainer remains under
**Train > RVC**.

The WebUI enables **Pretrained** by default. On the first start of a new
ContentVec experiment, it downloads the [Shiro ContentVec flow pretrained](https://huggingface.co/shiromiya/ShiroRVC-Resources/blob/main/Rectified_pretrains/pretrain_flow_contentvec.pth)
to `rvc/models/pretraineds/rectified/pretrain_flow_contentvec.pth`. The download
is pinned to a verified revision, checked with SHA-256, and reused for later runs.
Enable **Custom pretrained** to upload a compatible `.pth` checkpoint or enter
its path. Disable **Pretrained** to train a new experiment from scratch.
Existing experiments still resume from their saved checkpoint regardless of
these controls. Use a new model name to start over and preserve an existing run.
The default pretrained requires ContentVec; other embedders need a compatible
custom pretrained or training from scratch.

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
optimizer state, held-out evaluation and audio previews remain FP32. Muon
Newton-Schulz matrix iterations follow the effective training precision: FP16
when FP16 is selected, BF16 when supported and selected, and FP32 otherwise.
CPU runs and unsupported BF16 requests use FP32 iterations. Normalization is
computed in FP32 and the matrix update is converted back to FP32 afterward,
as in Shiro. AdamW calculations remain FP32. TF32 stays disabled. Changing the setting does not
alter a job already running.

Training accepts `--device cuda:0`, `--device cpu`, or multiple GPUs with
`--device cuda:0,cuda:1`. The WebUI Device field accepts the same values.
Multiple GPUs use DDP with one process per GPU and a distributed sampler.
Batch size and data-loader workers are per GPU; global batch size is batch size
times GPU count. Only rank zero writes checkpoints, TensorBoard logs, and
previews. Linux CUDA runs use NCCL. Windows uses Gloo with model synchronization
and gradient averaging through CPU memory to avoid native CUDA Gloo collectives;
this adds CPU transfer overhead. Single-device training
retains its existing path. GPU IDs must be unique and available.
Training loss and gradients are weighted by valid frames across ranks, so
short-clip padding does not give a GPU a disproportionate contribution.
Resume checkpoints keep the existing unwrapped model and optimizer formats.
BF16 falls back to FP32 on all ranks if any selected GPU cannot support it.
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

The flow implementation is synchronized with the local ShiroRVC update: LYNXNet2
uses the revised modulation arithmetic and a full-precision input projection.
Sampling also supports `churn` and caller-supplied churn noise. Muon batches
same-shaped matrices and AdamW uses grouped updates. RedPanda selects the
Newton-Schulz iteration dtype from the effective training precision, while
Shiro chooses FP16/BF16 automatically by GPU capability. Changing precision
on resume also changes Muon's iteration dtype without changing checkpoint
parameter or optimizer-state formats.
Model dimensions, the flow-matching objective, and the 400-frame crop are unchanged.

The vocoder path is optional. Without it, TensorBoard records mel previews and
exports have an empty vocoder reference. With it, previews include the original
recording, the flow output, and the real mel rendered by the vocoder. Dataset
previews use a non-mute clip of at least two seconds, capped at ten seconds.
Fine-tuning previews default to every 500 steps. Validation records loss at each
of five flow times and auxiliary mel loss; conditioning weight norms are logged
for diagnosis.

`--compile` (or the WebUI checkbox) compiles only the training backbone when CUDA
and Triton are available. Preview and evaluation paths remain eager. Compilation
runtime errors are surfaced. Existing checkpoint names and resume rules are retained.
Shiro's separate vocoder trainer, NSF-BigVGAN vocoder, and
conversion interface are not part of this flow training update.

`python -m rvc.rectified.openvpi input.ckpt output.pth` converts an OpenVPI
checkpoint into a compatible rectified vocoder export.
