"""Smooth radius pooling with bounded edge-encoding activations."""

import torch
from e3nn import o3
from torch.utils.checkpoint import checkpoint

from cc2cc.utils.model.model_utils.model_e3nn_correct_no_cube import E3nn as BaseE3nn


class E3nn(BaseE3nn):
    def radius_cutoff(self, edge_vec, radius):
        return (1 - edge_vec.square().sum(-1, keepdim=True) / radius**2).clamp_min(0).pow(3)

    def pool(self, features, edges, radius):
        weights = self.radius_cutoff(edges, radius)
        radii = torch.linalg.vector_norm(edges, dim=-1, keepdim=True)
        sh = o3.spherical_harmonics(
            self.irreps_sh, edges, normalize=True, normalization="component"
        )
        hidden = self.tp1(features, sh, self.radial(radii))
        return (hidden * weights).sum(-2, keepdim=True), weights.sum(-2, keepdim=True)

    def forward(self, features, edges, radius):
        if features.shape[:-1] != edges.shape[:-1] or features.shape[-1] != self.input_level:
            raise ValueError("Radius features and edges must have matching point dimensions.")
        numerator = features.new_zeros((*features.shape[:-2], 1, self.tp1.irreps_out.dim))
        denominator = features.new_zeros((*features.shape[:-2], 1, 1))
        chunk_size = max(1, 32768 // max(1, features.shape[0]))
        for start in range(0, features.shape[-2], chunk_size):
            stop = min(start + chunk_size, features.shape[-2])
            value, weight = checkpoint(
                self.pool, features[..., start:stop, :], edges[..., start:stop, :],
                radius, use_reentrant=False,
            )
            numerator = numerator + value
            denominator = denominator + weight
        return self.readout(numerator / denominator)
