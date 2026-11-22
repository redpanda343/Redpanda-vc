import torch


def load_voice_checkpoint(path):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except RuntimeError as error:
        if "mmap can only be used with files saved with" not in str(error):
            raise
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    if isinstance(checkpoint, dict):
        config = checkpoint.get("config")
        weights = checkpoint.get("weight")
        if (
            isinstance(config, (list, tuple))
            and len(config) >= 3
            and isinstance(weights, dict)
            and isinstance(weights.get("emb_g.weight"), torch.Tensor)
        ):
            return checkpoint
    raise ValueError("Select an exported RVC voice model (.pth).")
