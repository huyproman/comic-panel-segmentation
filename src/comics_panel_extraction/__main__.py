import sys
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

from comics_panel_extraction.cli import main

if __name__ == "__main__":
    sys.exit(main())
