- All stems receive and return input as `[n, c, h, w]` format.
- All stems have the func: get_shape which does a forward pass and returns shape.
- Stem `expansions` get multiplied with the `out_channels`.
- `expansions` in other blocks like trunk, head, work on `in_channels`.
- T2T arch works well with narrow and deep transformer blocks.
    - `embed_sz=64`, `n_layers=12`
- CCT arch works well with wide and shallow(er) transformer blocks.
    - `embed_sz=128`, `n_layers=4`
