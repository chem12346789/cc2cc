"""All points within 0.1 Bohr, smoothly pooled into 100 latent channels."""

from importlib import import_module


class Model(import_module(
    "cc2cc.utils.model.transformer+dense_mix_e3nn_4_no_cube"
).Model):
    def __init__(self):
        super().__init__(neighborhood_size=100)
        self.radius_cutoff = 0.1
