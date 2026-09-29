# Expert-v0 checkpoint load fix

PyTorch 2.6 changed `torch.load` to default to `weights_only=True`.

The existing Windows checkpoint stores `pathlib.WindowsPath` objects in its
saved config. The evaluator now keeps the safer weights-only loader and
allowlists only pathlib path classes required by that checkpoint format.

The already-created held-out test cache is valid and will be reused on the next
evaluation run.
