"""Relay Planetary Computer SAS tokens to weka for the MODIS build jobs.

The token endpoint answers 429 to the cluster's shared egress address while
other jobs use the Planetary Computer, so a machine outside the cluster (a
laptop on the VPN) fetches an anonymous token every ``--every`` minutes and
writes it, with its expiry, to the relay file that ``snow.modis`` reads first.
The write goes through a Beaker session that mounts weka (the
``beaker-weka-shell`` helper script, ``--shell``).

The tokens are anonymous, read-only and expire after about an hour; the file
is written with mode 600.

Usage (laptop)::

    python scripts/era5_supervised/v0/snow/token_relay.py --every 15
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SAS_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/modiseuwest/modis-061-cogs"
RELAY_FILE = "/weka/dfive-default/rslearn-eai/datasets/snow/raw/modis/sas_token.json"
SHELL = Path.home() / ".claude/skills/beaker-weka-shell/scripts/weka_shell.sh"


def relay_once(shell: Path, path: str) -> float:
    """Fetch one token and write it to the relay file.

    Args:
        shell: ``weka_shell.sh``.
        path: Relay file on weka.

    Returns:
        The token's expiry (epoch seconds).
    """
    body = requests.get(SAS_URL, timeout=60).json()
    expiry = datetime.fromisoformat(body["msft:expiry"].replace("Z", "+00:00")).timestamp()
    payload = base64.b64encode(json.dumps(dict(token=body["token"], expiry=expiry)).encode())
    command = (f"umask 077; mkdir -p $(dirname {path}); echo {payload.decode()} | base64 -d"
               f" > {path}.tmp && mv {path}.tmp {path} && echo relayed")  # fmt: skip
    # ``script`` gives the Beaker CLI the pseudo-terminal it needs.
    out = subprocess.run(
        ["script", "-q", "/dev/null", str(shell), "--", "bash", "-c", command],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=300,
    )  # fmt: skip
    if "relayed" not in out.stdout:
        raise RuntimeError(f"relay write failed: {out.stdout[-500:]} {out.stderr[-500:]}")
    return expiry


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--every", type=float, default=15.0, help="minutes between tokens")
    ap.add_argument("--path", default=RELAY_FILE)
    ap.add_argument("--shell", default=str(SHELL))
    ap.add_argument("--hours", type=float, default=12.0, help="stop after this long")
    args = ap.parse_args()
    stop = time.time() + args.hours * 3600
    while time.time() < stop:
        try:
            expiry = relay_once(Path(args.shell), args.path)
            logger.info("relayed a token valid until %s", datetime.fromtimestamp(expiry))
        except (requests.RequestException, RuntimeError, subprocess.TimeoutExpired) as exc:
            logger.warning("relay failed: %s", exc)
            time.sleep(60)
            continue
        time.sleep(args.every * 60)


if __name__ == "__main__":
    main()
