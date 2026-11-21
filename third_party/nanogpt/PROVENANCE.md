# NanoGPT source

`model.py` is an unmodified copy of Andrej Karpathy's MIT-licensed NanoGPT implementation.

- Repository: https://github.com/karpathy/nanoGPT
- Commit: `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`
- Original file: `model.py`
- License: accompanying `LICENSE`, Copyright (c) 2022 Andrej Karpathy.

No pretrained model weights are downloaded or used. The benchmark constructs a small model from scratch.

Text data: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt (public Shakespeare text); downloaded to `data/tinyshakespeare.txt`. Exact SHA256 and split lengths are saved with each experiment.
