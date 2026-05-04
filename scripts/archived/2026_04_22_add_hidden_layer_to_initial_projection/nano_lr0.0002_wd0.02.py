"""Nano hidden projection sweep: lr=0.0002, weight_decay=0.02."""

import nano

nano.LEARNING_RATE = 0.0002
nano.WEIGHT_DECAY = 0.02


if __name__ == "__main__":
    nano.run()
