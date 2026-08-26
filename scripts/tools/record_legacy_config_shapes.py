r"""Record the parameter manifest for a legacy checkpoint config fixture.

Run this from a checkout where the config still deserializes NATIVELY -- i.e. a commit
that predates the field removal, where no patching is needed. That is what makes the
resulting fixture non-circular: the expectation comes from the code the checkpoint was
trained with, not from HEAD.

    git worktree add /tmp/wt <commit>
    cd /tmp/wt && python scripts/tools/record_legacy_config_shapes.py \\
        <config.json> tests/fixtures/legacy_configs/<name>.shapes.json

The config may be a full checkpoint config.json or just its "model" subtree. The model
is built on the meta device, so this costs no memory regardless of model size.
"""

import json
import subprocess  # nosec
import sys
from pathlib import Path

import torch

from olmoearth_pretrain.nn.flexi_vit import EncoderConfig


def main() -> int:
    """Build the config on the meta device and write its parameter manifest."""
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    config_dict = json.loads(src.read_text())
    encoder_config = config_dict.get("model", config_dict)["encoder_config"]

    # Deliberately NOT patched: if this raises, you are on a commit that has already
    # dropped the field, and the manifest would record the wrong architecture.
    with torch.device("meta"):
        model = EncoderConfig.from_dict(encoder_config).build()

    commit = subprocess.check_output(  # nosec
        ["git", "rev-parse", "--short", "HEAD"], text=True
    ).strip()
    dst.write_text(
        json.dumps(
            {
                "recorded_at_commit": commit,
                "note": (
                    f"Parameter manifest built from this config at {commit}, where it "
                    f"deserializes natively. HEAD must rebuild the identical parameter "
                    f"set after patch_legacy_encoder_config."
                ),
                "shapes": {k: list(v.shape) for k, v in model.state_dict().items()},
            },
            indent=1,
            sort_keys=True,
        )
    )
    print(f"recorded {len(model.state_dict())} tensors at {commit} -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
