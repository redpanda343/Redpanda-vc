# Experimental rectified-flow training

This extends ShiroRVC's 44.1 kHz rectified-flow training recipe. New models use
dual-timestep training, separate voicing and tension conditioning, and RK4 sampling.
The flow predicts 128-bin log mel spectrograms from content, F0, loudness,
breathiness, voicing, tension and speaker. Existing models keep their saved recipe.
The default OpenVPI NSF-HiFiGAN vocoder renders audio previews and is
recorded in the exported model. Mel-only previews can be selected instead. The vocoder is frozen; this trains the flow,
not a new vocoder. RVC latent NSF-HiFiGAN generator checkpoints do not accept
these mels and cannot be used as the vocoder.

Use an OpenVPI NSF-HiFiGAN checkpoint with 44100 Hz sample rate, hop 512,
FFT/window 2048, 128 mel bins, fmin 40 and fmax 16000. Its accompanying
`config.json`, when provided, must remain beside it. The loader checks the
mel settings and loads generator weights strictly. Other vocoders are rejected.

In the WebUI, open **Train > Rectified Flow**. Enter a new model name and
dataset folder, preprocess, extract features, choose the audio preview vocoder,
and start training. The page shows the job status and live log.
The stop button ends the current rectified job; training resumes from the last
saved epoch, so unsaved steps are lost. The original trainer remains under
**Train > RVC**.

The WebUI defaults to the **Quality** recipe with **Pretrained** disabled so a
new model can learn the additional conditioning inputs from scratch. To fine-tune
a quality model, enable **Pretrained** and **Custom pretrained** and select a
checkpoint trained with the same inputs and dual-timestep setting. The original
Shiro pretrained is not compatible with the quality recipe.

For an original Shiro model, select **Legacy Shiro** and enable **Pretrained**.
On the first start of a new ContentVec legacy experiment, it downloads the
[Shiro ContentVec flow pretrained](https://huggingface.co/shiromiya/ShiroRVC-Resources/blob/main/Rectified_pretrains/pretrain_flow_contentvec.pth)
to `rvc/models/pretraineds/rectified/pretrain_flow_contentvec.pth`. The download
is pinned to a verified revision, checked with SHA-256, and reused for later runs.
Enable **Custom pretrained** to upload a compatible `.pth` checkpoint or enter
its path. Disable **Pretrained** to train a new experiment from scratch.
Existing experiments still resume from their saved checkpoint regardless of
these controls. Use a new model name to start over and preserve an existing run.
The default pretrained requires ContentVec; other embedders need a compatible
custom pretrained or training from scratch.

**Audio preview vocoder** defaults to **Default NSF-HiFiGAN**. On start,
it downloads only `pc_nsf_hifigan_44.1k_hop512_128bin_vocoder.pth` from
[Shiro's vocoders](https://huggingface.co/shiromiya/ShiroRVC-Resources/tree/main/vocoders)
to `rvc/models/pretraineds/rectified/`. This 56.6 MB file is the converted
inference generator from the larger `.ckpt`; the larger checkpoint is not
needed or downloaded. The revision and SHA-256 are verified, and the file is
reused on later starts. Choose **Custom NSF-HiFiGAN** to enter a compatible
OpenVPI `.ckpt` or converted `.pth` path. Choose **Mel previews only** to skip
the vocoder download and audio rendering. This selection is independent of
the flow pretrained toggle and also applies when resuming a run.

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
creates that copy automatically on its first run. It retains Shiro's full backbone
dimensions, pitch/time augmentation, auxiliary decoder,
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

If an FP16 forward pass produces a non-finite loss, the trainer retries that
batch once with autocast disabled (FP32). All DDP ranks make the same decision
and replay their original random draws, preserving the sampled noise, times,
and dropout. Successful retries update the optimizer and EMA normally; later
batches still use FP16. Gradient scaling continues to handle backward overflows.
If the FP32 retry also fails, training stops and reports any non-finite input
fields or model parameters across ranks. This does not repair corrupted data
or already corrupted weights.

Training accepts `--device cuda:0`, `--device cpu`, or multiple GPUs with
`--device cuda:0,cuda:1`. The WebUI Device field accepts the same values.
Multiple GPUs use DDP with one process per GPU and a distributed sampler.
Batch size and data-loader workers are per GPU; global batch size is batch size
times GPU count. Data-loader workers explicitly use `spawn`, as recommended
by [PyTorch's DDP documentation](https://docs.pytorch.org/docs/2.11/generated/torch.nn.parallel.DistributedDataParallel.html)
to avoid NCCL/fork deadlocks on Linux. CPU synchronization uses monitored
barriers to report missing ranks. NCCL initialization explicitly binds each
rank to its selected GPU. Startup vocoder loading, preview reference preparation
and output setup also propagate rank-zero errors to the other ranks. Only rank
zero writes checkpoints, TensorBoard logs, and previews. Rank-zero preview, validation and checkpoint work uses a separate
CPU/Gloo control group with a one-hour timeout, so an idle GPU does not enqueue
an NCCL barrier while this work runs. Training collectives retain their ten-minute
timeout. Stage start/finish messages identify slow work, and rank-zero exceptions
in these stages are sent to the other ranks. A stuck process or CUDA kernel can
still time out; this does not repair such failures. Linux CUDA runs use NCCL. Windows uses Gloo with model synchronization
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
retain Shiro's rectified-flow format. Select a flow export in the ordinary
**Inference** tab; the loader detects it and uses the flow and NSF-HiFiGAN
vocoder instead of the RVC synthesizer. Single and Batch conversion support
flow models. The flow's speaker count populates the speaker selector.

Inference runs the flow and vocoder in FP32. New quality models use 16 RK4 steps
(64 backbone evaluations). Legacy checkpoints retain 16 Euler steps. It reuses
the tab's F0 extractor, pitch shift, ContentVec or spin-v2 selection, optional
index retrieval and protection, audio splitting, and output processing. The
flow generates at its configured sample rate (44.1 kHz for this recipe).
Content, F0, energy, breathiness, voicing and tension use the training recipe's frame alignment.

Inference automatically uses the vocoder path recorded in the exported model.
The vocoder weights are stored separately from the flow checkpoint. If its
default vocoder path belongs to another machine or no vocoder was recorded,
the default NSF-HiFiGAN is located locally or downloaded and verified
automatically. Missing custom vocoders must be restored or replaced through
the inference API's optional `rectified_vocoder_path` argument; incompatible
mel settings are rejected. Training checkpoints also load using their EMA
weights when available.

The flow implementation is synchronized with the local ShiroRVC update: LYNXNet2
uses the revised modulation arithmetic and a full-precision input projection.
Sampling also supports `churn` and caller-supplied churn noise. Muon batches
same-shaped matrices and AdamW uses grouped updates. RedPanda selects the
Newton-Schulz iteration dtype from the effective training precision, while
Shiro chooses FP16/BF16 automatically by GPU capability. Changing precision
on resume also changes Muon's iteration dtype without changing checkpoint
parameter or optimizer-state formats.
The backbone dimensions and 400-frame crop are unchanged. The quality recipe
adds two conditioning projections and dual-timestep training.

The vocoder path is optional. Without it, TensorBoard records mel previews and
exports have an empty vocoder reference. With it, previews include the original
recording, the flow output, and the real mel rendered by the vocoder. Dataset
previews use a non-mute clip of at least two seconds, capped at ten seconds.
Fine-tuning previews default to every 500 steps. Validation records loss at each
of five flow times and auxiliary mel loss; conditioning weight norms are logged
for diagnosis.

`--compile` (or the WebUI checkbox) compiles only the training backbone with
PyTorch Inductor. `requirements.txt` pins Triton 3.6.0 for PyTorch 2.11.0 on Linux;
Triton provides Python 3.12 wheels for Linux x86_64 and aarch64. Official Triton
supports Linux and NVIDIA compute capability 8.0 or newer. Unsupported platforms,
non-CUDA devices, older NVIDIA GPUs and unavailable Triton imports train uncompiled
and report the reason in the job log. Compilation starts on the first training
step, including the backward graph, so startup takes longer. The WebUI uses the
`default` compile mode; the CLI also accepts `reduce-overhead` and `max-autotune`,
which can use more GPU memory through CUDA graphs. Preview and evaluation paths
remain eager. Compilation runtime errors are surfaced. Existing checkpoint names
and resume rules are retained.

Compiled forwards use a scoped `backward_pass_autocast="off"` setting to match
the trainer's backward pass outside autocast. This also covers FP32 overflow
retries without changing the eager training path or the global compiler setting.
See [PyTorch compiled autograd semantics](https://docs.pytorch.org/docs/2.11/user_guide/torch_compiler/torch.compiler_backward.html).

Compatibility references: [PyTorch 2.11.0 Triton pin](https://github.com/pytorch/pytorch/blob/v2.11.0/.ci/docker/triton_version.txt),
[Triton installation](https://triton-lang.org/main/getting-started/installation.html),
[Triton 3.6.0 hardware support](https://github.com/triton-lang/triton/blob/v3.6.0/README.md#compatibility),
and [PyTorch 2.11 torch.compile](https://docs.pytorch.org/docs/2.11/generated/torch.compile.html).
Shiro's separate vocoder trainer, NSF-BigVGAN vocoder, and
separate conversion interface are not part of this flow training update.

`python -m rvc.rectified.openvpi input.ckpt output.pth` converts an OpenVPI
checkpoint into a compatible rectified vocoder export.

## Quality recipe

The default `rvc/configs/rectified/44100.json` enables `dual_timestep`, `voicing`
and `tension` under `flow.model`, with `sampling_method: "rk4"` and
`sampling_steps: 16`. Those settings travel with the exported checkpoint and
control both previews and ordinary Single/Batch inference. The Python sampler
also accepts explicit `method="euler"`, `"heun"` or `"rk4"` and `steps` overrides.
RK4 evaluates the backbone four times per step and is slower than Euler at the
same step count. An audible improvement is not guaranteed.

Dual-timestep training assigns a second independently sampled flow time to 25%
of frames on average. Both the noisy mel mixture and the backbone time embedding
use each frame's selected time, including adaptive normalization. Padding remains
excluded from losses. RedPanda's existing logit-normal time distribution is
retained; the uniform-time distribution from DiffSinger is not copied.
Validation and inference use a common time across frames.

Voicing and tension follow DiffSinger's WORLD harmonic-analysis definitions.
Voicing is harmonic RMS loudness in dB, divided by 96. Tension is the logit of
the non-fundamental harmonic RMS fraction, scaled by 0.1. Extraction uses
`pyworld==0.3.5`, fixed local dither for repeatability, finite silence handling,
and the existing 100 Hz alignment and 60 ms smoothing. Existing breathiness
conditioning is unchanged. WORLD analysis uses its required double-precision CPU
arrays internally; the extracted inputs and FP32 model remain float32.
Training, validation, previews and conversion call the same extractor. Pitch
augmentation and conversion pitch shifts retain the source voicing/tension
curves, and time stretching aligns them to the adjusted mel hop.

These features add CPU extraction work per clip and two small learned projections.
Install the updated requirements before training or converting a quality model.
Models with the new inputs require a newly trained compatible pretrained model.
Legacy models do not invoke WORLD extraction. Missing new conditioning inputs
raise an error instead of silently replacing them with invented curves.

Use a new experiment name for the quality recipe. The CLI defaults to this recipe;
`--recipe legacy` creates a new experiment with the original architecture and
Euler sampler. Existing `rectified_config.json` files always take precedence,
and resume still verifies the saved configuration. The trainer rejects pretrained
checkpoints with mismatched voicing, tension or dual-timestep settings.

These are experimental quality features. Functional checks and short synthetic
training runs do not establish better voice similarity or audio quality. Compare
properly trained models on held-out real audio before choosing a new pretrained.
