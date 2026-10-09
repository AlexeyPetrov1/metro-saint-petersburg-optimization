import argparse
import json
from pathlib import Path
import sys

from .build import check, prepare
from .pipeline import SourceError


def main():
    parser = argparse.ArgumentParser(description='Prepare or check metro data without changing source files.')
    parser.add_argument('command', choices=['prepare', 'check'])
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root / 'data/processed/metro').resolve()
    try:
        result = (prepare if args.command == 'prepare' else check)(root, output)
    except (SourceError, FileNotFoundError) as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    raise SystemExit(main())
