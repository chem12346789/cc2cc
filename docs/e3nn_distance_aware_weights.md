# Distance-aware weights in the e3nn model

## Distance information versus parameter count

Both ordinary tensor-product weights and a radial network are trainable.
The difference is not simply the number of parameters: it is whether the
weights depend on distance.

Let `f_i` be the input features at grid point `i`, and let
`r_i = ||x_i - x_center||` be its distance from the cube center.

| Approach | Weights at point `i` | Consequence |
| --- | --- | --- |
| Shared learned tensor-product weights | `W_i = W` | All points use the same weights. More shared parameters alone do not supply distance information. |
| Radial network | `W_i = MLP_theta(r_i)` | A shared trainable function generates distance-dependent weights. |
| Independent learned weights per point | `W_i = theta_i` | Weights depend on grid indices rather than geometry. Unconstrained weights generally break rotational/reflection symmetry. |
| Independent learned weights per radial shell | `W_i = theta_shell(i)` | Points at the same radius share weights. This is a simpler distance-aware option for a fixed grid. |

## What the original implementation uses

The original [E3nn implementation](../cc2cc/utils/model/model_utils/model_e3nn_correct.py)
constructs displacements from the cube center, not all pairwise displacements.
Its spherical harmonics use `normalize=True`, which removes the radial
magnitude for nonzero displacements. There is no separate radial weighting.

The [original mixed model](../cc2cc/utils/model/transformer+dense_mix_e3nn_4_correct.py)
sets `lmax=0`, so its spherical-harmonic feature is constant and carries no
direction information either. Shared tensor-product weights, mean aggregation,
and a linear readout make that e3nn branch depend on mean input features,
not on which neighboring point is nearer or farther.
The mixed model separately processes the center value, but that is not a
neighbor-distance encoding.

## What `self.radial(self.radii)` learns

The [distance-aware implementation](../cc2cc/utils/model/model_utils/model_e3nn_correct2.py)
uses:

```python
weights = self.radial(self.radii)
f_hidden = self.tp1(f_in, self.sh, weights)
f_hidden = f_hidden.mean(dim=-2, keepdim=True)
return self.readout(f_hidden)
```

- `self.radii` contains fixed point-to-center distances in the same units as
  `EDGE_LEN`. It is a buffer, not a trainable parameter.
- `self.radial` is a trainable `Linear(1, 16) -> SiLU -> Linear(16, P)` network,
  where `P = self.tp1.weight_numel`.
- The generated `weights` have shape `[EDGE_SIZE**3, P]`. They are outputs of
  the network, not independent parameters stored for each point.
- Backpropagation through the tensor product updates the radial network's
  parameters. The scalar readout also retains its own trainable parameters.
- `internal_weights=False` means the tensor product receives weights from
  outside; it does not mean those weights cannot be learned.
- `shared_weights=False` permits a different weight vector for each point.
  The radial network itself is shared across all points and input samples.

Equal radii always produce equal weight vectors. Different radii can produce
different vectors, but learning may also make them similar or ignore distance.
At `lmax=0`, this gives a learned radial weighting before averaging, rather
than an unweighted mean followed by shared linear maps.

## The four shells of the default cube

For a centered `3 x 3 x 3` grid with positive spacing `a = EDGE_LEN`:

| Location | Number of points | Distance from center |
| --- | --- | --- |
| Center | 1 | `0` |
| Face centers | 6 | `a` |
| Edge centers | 12 | `sqrt(2) * a` |
| Corners | 8 | `sqrt(3) * a` |

Thus, the radial network produces only four distinct weight vectors on this
grid. Learning one vector per shell is a valid alternative, not an implemented
option in the current module.

For `P` tensor-product weights, the original shared weights require `P`
parameters, independent point weights require `27 * P`, independent shell
weights require `4 * P`, and the current radial MLP requires `32 + 17 * P`.
These counts exclude the readout. The MLP is not necessarily the smallest
choice; its benefit is sharing a smooth function of radius.

## Symmetry, limitations, and use

- Radius is unchanged by rotations and reflections. Sharing radial weights
  preserves compatibility with the tensor product's symmetry. On the fixed
  cubic grid, the relevant exact spatial symmetries are the octahedral ones.
- Distances are to the center, not between every pair of points. Points at
  the same radius cannot be distinguished by the radial network alone.
- Increasing `lmax` does not add distance information. In this particular
  implementation, the retained linear scalar readout cannot use higher-order
  hidden irreps, so increasing `lmax` alone also does not make the scalar output
  angularly sensitive. That requires additional equivariant interactions.
- The radial function can be evaluated at unseen distances, but accuracy at
  those distances is not guaranteed. The module currently constructs a fixed
  grid and does not accept per-sample coordinates.
- Enabling the new module requires changing the model's import to
  `from cc2cc.utils.model.model_utils.model_e3nn_correct2 import E3nn`.
  Constructor arguments and input/output shapes are retained, but original
  checkpoints are not interchangeable; retraining is required.
- [Targeted tests](../tests/test_model_e3nn_correct2.py) cover distance
  sensitivity, cubic symmetry checks, gradients, shapes, finite center
  features, checkpoint round-trip, and unsupported geometry.

More parameters increase capacity; radial conditioning supplies a specific
dependence on distance.
