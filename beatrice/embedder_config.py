import json
from pathlib import Path

LOOKAHEAD_CHOICES = (0, 10, 20, 25)


def default_config():
    return json.loads(Path(__file__).with_name('embedder_defaults.json').read_text(encoding='utf-8'))
