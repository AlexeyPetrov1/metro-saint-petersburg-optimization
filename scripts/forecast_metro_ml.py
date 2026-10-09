"""Export frozen-model station entries for the existing metro simulator."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metro_ml import FrozenMetroForecaster, INPUTS, prepare_inputs, station_mapping


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--as-of', required=True, help='Decision time with timezone, e.g. 2026-09-30T09:00:00+03:00')
    parser.add_argument('--output', type=Path, help='JSON destination; defaults to stdout')
    parser.add_argument('--stations', nargs='+', help='Simulator IDs, e.g. L1_01 L1_07; defaults to all 19')
    parser.add_argument('--diagnostic', action='store_true', help='Allow dates within the frozen model training period')
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    try:
        required = ['lgbm_final/meta.json', 'data/interim/spb_line1_hourly.parquet',
            'data/interim/spb_line1_15min.parquet']
        if not all((INPUTS / name).is_file() for name in required):
            prepare_inputs()
        model = FrozenMetroForecaster(allow_training_period=args.diagnostic)
        bundle = model.forecast_bundle(model.load_simulator_frame(),
            args.stations or list(station_mapping()), args.as_of)
        text = json.dumps(bundle, ensure_ascii=False, indent=2) + '\n'
        if args.output is None:
            print(text, end='')
        else:
            output = args.output.resolve()
            if output.exists() and output.read_bytes() != text.encode('utf-8'):
                raise ValueError(f'existing forecast differs; choose another output: {output}')
            output.parent.mkdir(parents=True, exist_ok=True)
            if not output.exists():
                output.write_text(text, encoding='utf-8')
            print(json.dumps(dict(status='complete', output=str(output), rows=len(bundle['rows']),
                evaluation_mode=bundle['evaluation_mode']), ensure_ascii=False))
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
