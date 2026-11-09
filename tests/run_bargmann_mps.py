#!/usr/bin/env python3
"""Run the shared Bargmann suite on macOS Metal (MPS), defaulting to float32.

Keep run_bargmann_cuda.py alongside this file when copying the scripts.
"""

from run_bargmann_cuda import main

if __name__ == '__main__':
    raise SystemExit(main(default_device='mps', default_output='bargmann_mps_report.json'))
