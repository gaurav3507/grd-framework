"""E13 under amendment 1 (notes/e13_amendment_1.md) for the rebuilt K562 and RPE1 files.

Runs experiments/e13_power_sign.py unchanged, except that its reproduction tolerance
TOL (Step 0d --verify and the Part B check against the stored E12 T2) is 1e-6 instead
of 1e-8. Same flags as e13_power_sign.py. Norman is refused: it uses the original file
and keeps 1e-8, so run it with e13_power_sign.py directly.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import e13_power_sign as M  # noqa: E402

AMENDED_TOL = 1e-6


def main():
    if "norman" in sys.argv:
        raise SystemExit("STOP: Norman keeps the 1e-8 tolerance; use experiments/e13_power_sign.py")
    M.TOL = AMENDED_TOL
    M.main()


if __name__ == "__main__":
    main()
