"""Deprecated entry point for the studio corpus CSV path.

Prefer:

    python -m olmoearth_pretrain.dataset_creation.create_windows \
        --ds_path ... --corpus_csv ... [--verify_s2]

This shim simply forwards to the unified CLI.
"""

from .__main__ import main

if __name__ == "__main__":
    main()
