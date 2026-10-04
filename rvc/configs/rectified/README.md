# Rectified-flow configuration

`44100_standard.json` contains the main settings for a new experiment. Training saves the same compact format to `logs/<model>/rectified_config.json`.

Like DiffSinger's base and experiment configs, `preset: standard-v1` supplies shared defaults. Audio and mel settings, model architecture, conditioning, optimizer details, and other defaults live in `rvc/rectified/config.py`. This cleanup preserves the existing training recipe, including its learning rate and augmentation settings.

The editable config keeps the learning rate and decay, batch limits, augmentation, worker count, preview and validation intervals, holdout size, and inference sampler. `model.sampling_method` supports `euler`, `rk2`, `rk4`, and `rk5`.

You can override other defaults using their existing nested `data` or `flow` keys. Nested objects inherit unspecified settings. For example, `flow.grad_clip` overrides gradient clipping and `flow.model.backbone_args.channels` overrides backbone width. An explicit `null` replaces a default object rather than inheriting it. Model compatibility checks still apply.

Checkpoints always store fully expanded settings, so they do not depend on preset defaults. Existing full configs remain supported without injecting new architecture defaults. Saving an existing experiment uses the compact format only when it resolves to exactly the same settings, including custom overrides. Resume compares the expanded settings.

The preset is versioned. Changes to its defaults should use a new preset name and retain `standard-v1` for existing experiments.
