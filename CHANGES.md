# Changelog

## 4D (3D + time) convolution operators and configurable padding mode

This release adds a set of 4D operators that implement time-aware convolutions
by decomposing them into stacks of 3D convolutions applied per timestep, registers them
with the dimensionality helpers, and threads a new `padding_mode` argument through the
`PlainConvUNet` construction chain.

### Added

**New module `building_blocks/operations_4D.py`**

| Class | Purpose |
|---|---|
| `Conv4D` | Full 4D convolution as a sum of three 3D convolutions applied at `t-1`, `t` and `t+1`. Only the center branch carries the bias. Temporal kernel size must be 1 or 3; with `kernel_size[0] == 1` only the center branch runs. |
| `Conv4DHypercross` | "Hypercross" variant: the center branch uses the full spatial kernel while the `t±1` branches use `1×1×1` kernels, giving a cross-shaped 4D receptive field at much lower parameter cost. |
| `Conv3Din4D` | Purely spatial convolution applied independently per timestep; no temporal mixing. |
| `NormOp4D` | Wraps `InstanceNorm3d` and applies it per timestep, concatenating along the time axis. |
| `TranspConv4D` | Wraps `ConvTranspose3d`, applied per timestep; temporal upsampling is done by repeating each output `stride[0]` times. |

Shared conventions for all of the above:
- Input layout is `(B, C, T, X, Y, Z)`; the time axis is `dim=2`.
- `stride`, `kernel_size` and `padding` accept scalars (broadcast to 4 entries) or
  4-element sequences; entry `[0]` is temporal, `[1:]` is passed to the underlying 3D op.
- New `padding_mode="time_cyclic"` value: recorded as `self.time_padding_mode = "cyclic"`
  and downgraded to `"zeros"` for the wrapped 3D convolution.
- Temporal neighbours are gathered with wrap-around indexing (`x[:, :, -1]` and
  `t % x.size(2)`), i.e. **circular/periodic boundary conditions in time** — appropriate
  for cyclic acquisitions (e.g. cardiac phases).

### Changed

**`building_blocks/helper.py` — dimension 4 registered**
- `convert_dim_to_conv_op(4)` → `Conv4D`; error message updated to "Only 1, 2, 3, and 4".
- `convert_conv_op_to_dim()` returns `4` for `Conv4DHypercross`, `Conv4D` and `Conv3Din4D`.
- `get_matching_convtransp()` accepts `dimension=4` and returns `TranspConv4D`.
- `maybe_convert_scalar_to_list()` broadcasts to 4 entries for the new conv ops.
- `get_matching_pool_op()`: removed a duplicated `convert_conv_op_to_dim()` call; the
  assertion still restricts pooling to dimensions 1–3 (no 4D pooling).

**`padding_mode` plumbed through the UNet stack**
- `ConvDropoutNormReLU`, `StackedConvBlocks`, `PlainConvEncoder`, `UNetDecoder` and
  `PlainConvUNet` all gained a `padding_mode: str = "zeros"` parameter, forwarded down to
  the conv constructor. Default preserves previous behaviour.
