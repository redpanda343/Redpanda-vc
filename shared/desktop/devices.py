"""Lightweight PortAudio device discovery without importing PyTorch or NumPy."""

import os
import re

os.environ.setdefault("SD_ENABLE_ASIO", "1")

import sounddevice as sd


def friendly_device_label(api_name, device_name, direction):
    if "voicemeeter" in device_name.lower():
        if api_name == "ASIO":
            return f"[Voicemeeter ASIO] {device_name}"
        arrow = "→ RVC input" if direction == "input" else "← RVC output"
        return f"[Voicemeeter] {device_name} {arrow}"
    api_label = "WASAPI" if api_name == "Windows WASAPI" else api_name
    return f"[{api_label}] {device_name}"


def asio_channel_choices(count, prefix, mono):
    choices = {}
    if mono:
        for channel in range(count):
            choices[f"{prefix} {channel + 1}"] = [channel]
    for first in range(0, count - 1, 2):
        choices[f"{prefix} {first + 1}+{first + 2}"] = [first, first + 1]
    if count % 2:
        choices[f"{prefix} {count}"] = [count - 1]
    return choices


def device_sort_key(label):
    lower_label = label.lower()
    if "voicemeeter" in lower_label:
        priority = 0
    elif label.startswith("[ASIO]"):
        priority = 1
    elif label.startswith("[WASAPI]"):
        priority = 2
    else:
        priority = 3
    parts = re.split(r"(\d+)", lower_label)
    return priority, [int(part) if index % 2 else part for index, part in enumerate(parts)]


def audio_routes(all_drivers=False, rescan=False):
    if rescan:
        sd._terminate()
        sd._initialize()
    hostapis = sd.query_hostapis()
    inputs, outputs = {}, {}
    for index, device in enumerate(sd.query_devices()):
        api = hostapis[device["hostapi"]]["name"]
        if os.name == "nt" and not all_drivers and api not in {"ASIO", "Windows WASAPI"}:
            continue
        for direction, routes in (("input", inputs), ("output", outputs)):
            count = int(device[f"max_{direction}_channels"])
            if not count:
                continue
            label = friendly_device_label(api, device["name"], direction)
            if api == "ASIO":
                choices = asio_channel_choices(count, "In" if direction == "input" else "Out", direction == "input")
                for channel_name, selectors in choices.items():
                    routes[f"{label}: {channel_name}"] = (index, selectors)
            else:
                routes[label] = (index, None)
    if os.name == "nt" and not all_drivers and (not inputs or not outputs):
        return audio_routes(all_drivers=True)
    host = next((api for api in hostapis if api["name"] == "Windows WASAPI"), None)
    defaults = (host["default_input_device"], host["default_output_device"]) if host else sd.default.device
    result = {}
    for key, routes, default in (("inputs", inputs, defaults[0]), ("outputs", outputs, defaults[1])):
        labels = sorted(routes, key=device_sort_key)
        result[key] = labels
        result[f"default_{key}"] = next((label for label in labels if routes[label][0] == default), labels[0] if labels else "")
    return result, inputs, outputs
