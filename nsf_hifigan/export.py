import argparse
from pathlib import Path

from nsf_hifigan.checkpoints import export_vocoder, latest_checkpoint
from nsf_hifigan.config import ROOT, experiment_paths


def main():
    parser = argparse.ArgumentParser(description='Export a trained NSF-HiFiGAN vocoder in the OpenVPI format.')
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--checkpoint', help='Training checkpoint to export (default: the latest one).')
    parser.add_argument('--output', help='Destination folder (default: models/pretraineds/rectified/trained/<name>_<steps>s).')
    args = parser.parse_args()
    checkpoint = args.checkpoint or latest_checkpoint(experiment_paths(ROOT / 'logs' / args.model_name)['output'])
    if checkpoint is None:
        raise SystemExit(f'No vocoder checkpoint found for {args.model_name}. Train it first.')
    exported = export_vocoder(checkpoint, Path(args.output) if args.output else None, args.model_name)
    print(f'Exported {checkpoint} to {exported.parent}. Select it as a Rectified Flow vocoder.', flush=True)


if __name__ == '__main__':
    main()
