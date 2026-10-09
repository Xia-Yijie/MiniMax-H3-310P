"""Run CPU regressions and expose failure traces as GitHub annotations."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def annotation_escape(text):
    return text.replace('%', '%25').replace('\r', '%0D').replace('\n', '%0A')


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    for test, trace in result.errors + result.failures:
        print('::error::' + annotation_escape(f'{test}\n{trace}'), flush=True)
    raise SystemExit(0 if result.wasSuccessful() else 1)
