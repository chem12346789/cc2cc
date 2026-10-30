"""All points within 0.1 Bohr, smoothly pooled into 100 latent channels."""

from importlib import import_module

import torch
from torch.utils.checkpoint import checkpoint

from cc2cc.utils.model.model_utils.model_e3nn_correct_no_cube_radius import E3nn

class Model(import_module(
    "cc2cc.utils.model.transformer+dense_mix_e3nn_4_no_cube"
).Model):
    def __init__(self):
        super().__init__(neighborhood_size=100)
        for i in range(1, self.input_level + 1):
            previous = getattr(self, f"conv{i}")
            with torch.random.fork_rng(devices=[]):
                encoder = E3nn(
                    self.cube_type, self.cube_size, self.input_level, self.lmax
                )
            encoder.load_state_dict(previous.state_dict())
            setattr(self, f"conv{i}", encoder)
        self.radius_cutoff = 0.1

    def predict_encoded(self, encoded, center):
        mixing = torch.softmax(self.mixing_weight(encoded.flatten(1)), dim=-1)
        neighborhood = self.densenet(self.predictor(encoded).flatten(1))
        experts = torch.cat(
            (neighborhood, self.densenet_center(center), center), dim=-1
        )
        return (mixing * experts).sum(dim=-1, keepdim=True)

    def forward(self, x, relative_positions, *, neighborhoods=None):
        if neighborhoods is None:
            if x.ndim != 3 or x.shape[1] != self.input_level:
                raise ValueError("Expected radius density features with shape (N, 4, K).")
            if relative_positions.shape != (x.shape[0], x.shape[2], 3):
                raise ValueError("Radius edges must match density feature point dimensions.")
            encoded = torch.cat([
                getattr(self, f"conv{i}")(
                    x.permute(0, 2, 1), relative_positions, self.radius_cutoff
                ) for i in range(1, self.input_level + 1)
            ], dim=-2)
            return self.predict_encoded(encoded, x[:, :, 0])

        outputs = []
        ranges = list(neighborhoods.ranges(32768))
        range_index = 0
        for center_start in range(0, len(x), 256):
            center_stop = min(center_start + 256, len(x))
            pooled = []
            while range_index < len(ranges) and ranges[range_index][1] <= center_start:
                range_index += 1
            for index in range(range_index, len(ranges)):
                start, stop = ranges[index]
                if start >= center_stop:
                    break
                start, stop = max(start, center_start), min(stop, center_stop)
                if start >= stop:
                    continue

                def aggregate(features, coords, start=start, stop=stop):
                    indices = neighborhoods.query(start, stop)
                    local = torch.as_tensor(indices, dtype=torch.int64, device=features.device)
                    edges = coords[local] - coords[start:stop, None, :]
                    padding = local == local[:, :1]
                    padding[:, 0] = False
                    outside = edges.new_tensor((2 * self.radius_cutoff, 0, 0))
                    edges = torch.where(padding[..., None], outside, edges)
                    totals = [None] * self.input_level
                    weights = None
                    edge_width = max(1, 32768 // (stop - start))
                    for edge_start in range(0, local.shape[1], edge_width):
                        edge_stop = edge_start + edge_width
                        gathered = features[local[:, edge_start:edge_stop]]
                        for i in range(self.input_level):
                            value, weight = getattr(self, f"conv{i + 1}").pool(
                                gathered, edges[:, edge_start:edge_stop], self.radius_cutoff
                            )
                            totals[i] = value if totals[i] is None else totals[i] + value
                        weights = weight if weights is None else weights + weight
                    return torch.cat([value / weights for value in totals], dim=-2)

                pooled.append(checkpoint(
                    aggregate, x, relative_positions, use_reentrant=False
                ))
            hidden = torch.cat(pooled)
            encoded = torch.cat([
                getattr(self, f"conv{i + 1}").readout(hidden[:, i:i + 1])
                for i in range(self.input_level)
            ], dim=-2)
            outputs.append(self.predict_encoded(encoded, x[center_start:center_stop]))
        return torch.cat(outputs)
