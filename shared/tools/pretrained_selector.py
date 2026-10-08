import os


def pretrained_selector(vocoder, sample_rate, version="v2"):
    if version not in {"v1", "v2"}:
        raise ValueError(f"Unsupported RVC version: {version}")
    if version == "v1":
        if vocoder != "HiFi-GAN" or int(sample_rate) != 40000:
            raise ValueError("RVC v1 pretrained models require HiFi-GAN at 40000 Hz.")
        base_path = os.path.join(
            "models", "pretraineds", "hifi-gan", "v1"
        )
    else:
        base_path = os.path.join("models", "pretraineds", f"{vocoder.lower()}")

    path_g = os.path.join(base_path, f"f0G{str(sample_rate)[:2]}k.pth")
    path_d = os.path.join(base_path, f"f0D{str(sample_rate)[:2]}k.pth")

    if os.path.exists(path_g) and os.path.exists(path_d):
        return path_g, path_d
    else:
        return "", ""
