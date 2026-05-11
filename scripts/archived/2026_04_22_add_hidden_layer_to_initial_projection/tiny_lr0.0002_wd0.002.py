"""Tiny hidden projection sweep: lr=0.0002, weight_decay=0.002."""

import tiny

tiny.LEARNING_RATE = 0.0002
tiny.WEIGHT_DECAY = 0.002


if __name__ == "__main__":
    tiny.run()
