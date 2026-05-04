"""Tiny hidden projection sweep: lr=0.0001, weight_decay=0.002."""

import tiny

tiny.LEARNING_RATE = 0.0001
tiny.WEIGHT_DECAY = 0.002


if __name__ == "__main__":
    tiny.run()
