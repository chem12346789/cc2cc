# Distance- and angle-aware e3nn weights

## Implementations

- [model_e3nn_correct3.py](../cc2cc/utils/model/model_utils/model_e3nn_correct3.py)
  uses spacing-normalized radial weights and a linear scalar readout.
- [model_e3nn_correct_f.py](../cc2cc/utils/model/model_utils/model_e3nn_correct_f.py)
  combines radial weights, quadratic angular invariants, and a nonlinear
  scalar head.
- [transformer+dense_mix_e3nn_4_correct_f.py](../cc2cc/utils/model/transformer+dense_mix_e3nn_4_correct_f.py)
  combines four e3nn branches with a Transformer and dense predictions.

See [distance-aware weights](e3nn_distance_aware_weights.md) for the difference
between shared tensor-product weights and distance-dependent weights.

## Distance-dependent weights

For each grid displacement `r_i` from the cube center, the radial input is
`rho_i = ||r_i|| / EDGE_LEN`. On the default three-point-per-axis grid, the
distinct inputs are `0`, `1`, `sqrt(2)`, and `sqrt(3)`.

The shared radial network is:

```text
Linear(1, 16) -> SiLU -> Linear(16, tp1.weight_numel)
```

It produces external tensor-product weights for each point:

```python
weights = self.radial(self.radii)
f_hidden = self.tp1(f_in, self.sh, weights)
f_hidden = f_hidden.mean(dim=-2, keepdim=True)
```

- `shared_weights=False` and `internal_weights=False` let the tensor product
  use these generated weights.
- Equal radii receive equal weights; trainable parameters are shared across
  points rather than independently assigned to grid indices.
- Distance dependence works even at `lmax=0`.
- Normalization measures radius in grid-spacing units, not absolute physical
  distance. Changing `EDGE_LEN` alone does not change the radial inputs.

## Angular features and scalar invariants

Normalized spherical harmonics encode directions up to `lmax`.
The combined implementation uses parity `(-1)^l`: `0e`, `1o`, `2e`, `3o`, etc.

- `f_lin = self.readout(f_hidden)` maps only the hidden `0e` features into
  scalar features. A linear map to scalars cannot retain higher-order angular
  components.
- `f_quad = self.tensor_square(f_hidden)` contracts pairs of hidden features
  into `0e` invariants. Because this happens after pooling, the quadratic
  terms include interactions between different grid points and can encode
  relative angles.
- At `lmax=0`, the quadratic path is still nonlinear but has no directional
  information. Angular sensitivity requires `lmax >= 1`.

The contractions and readout are O(3)-invariant when features and geometry
transform together. For a fixed Cartesian cube, rotations/reflections that
permute its grid points preserve the output; arbitrary rotations require
rotating the geometry too, not merely permuting fixed-grid input values.

## Nonlinear merging

Both readouts produce `cube_size` scalar features. They are concatenated in
the order `[f_quad, f_lin]` and passed through:

```text
Linear(2 * cube_size, cube_size) -> SiLU -> Linear(cube_size, cube_size)
```

This allows output-specific mixing and nonlinear interactions between the
two paths. The head receives only `0e` scalars, so ordinary scalar
activations preserve invariance. Applying an ordinary MLP directly to
higher-order components of `f_hidden` would not preserve it.

The previous two-coefficient linear mixer could be absorbed into the learned
readout weights; it added no representational capacity. The nonlinear head
is more expressive, but improved physical accuracy has not been established.

## Four-branch mixed model

| Branch | `lmax` | Included angular orders |
| --- | --- | --- |
| `conv1` | 0 | 0 |
| `conv2` | 1 | 0, 1 |
| `conv3` | 2 | 0, 1, 2 |
| `conv4` | 3 | 0, 1, 2, 3 |

Each branch receives **all four input features**, not a single isolated input
channel. Its output is one learned feature row of length `cube_size`.
The model's `self.lmax=3` records the maximum branch order.

- Mixed-model input: `[batch, 4, EDGE_SIZE**3]`.
- e3nn input after permutation: `[batch, EDGE_SIZE**3, 4]`.
- Each branch output: `[batch, 1, cube_size]`.
- Concatenated branch output: `[batch, 4, cube_size]`.
- Final prediction: `[batch, 1]`.

The final softmax gate has six weights: one for the Transformer/dense cube
prediction, one for the dense center prediction, and four for the raw center
features.

## Device, precision, and checkpoints

- The e3nn modules initialize geometry and parameters on CPU with explicit
  `torch.float64`; registered geometry buffers move with the model.
- Transformer and DenseNet layers currently use PyTorch's default dtype.
  Under the usual float32 default, the mixed model initially has mixed dtypes.
  Convert the complete model before using float64 inputs:
  `model.to(device=target_device, dtype=torch.float64)`.
- The [model wrapper](../cc2cc/utils/ModelClass.py) calls `.double()` when
  `precision="float64"`. Its other precision path does not make all parameters
  float32 and can leave this model with incompatible dtypes.
- Keep inputs consistent with the model's device and dtype. Do not hard-code
  CPU for new tensors created during a GPU forward pass.
- Float64 is the precision baseline. Float32 is an optimization to validate
  against output, derivative, and downstream physical tolerances.
  Converting float32 constants to float64 cannot recover lost precision.
- Earlier shared-weight/linear-mixer checkpoints and the previous all-zero
  branch-order configuration are not directly compatible with this architecture.
  Use checkpoints produced for the matching architecture.

## Validation

[Targeted tests](../tests/test_model_e3nn_correct_f.py) cover normalized radii,
output shapes, distance sensitivity at `lmax=0`, angular sensitivity for
equal-radius inputs, nonlinear merging, cube rotations/reflections, gradient
flow, and the mixed model's `[0, 1, 2, 3]` branch orders.

```bash
python -m unittest tests.test_model_e3nn_correct_f -q
```

The seven targeted tests passed on CPU in float64 after the branch-order
change. GPU behavior, float32 physical accuracy, and training performance
were not validated; no training was launched.
