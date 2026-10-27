# Third-party model attribution

The model code is split into modules so that upstream attribution remains
visible. This notice does not assign a new license to the whole repository.

## Centaur

`dsfn/centaur.py` contains the sensor adaptation of the autoencoder and
temporal-attention classifier from
[sustainable-computing/Centaur](https://github.com/sustainable-computing/Centaur),
commit `9a8d77f33e6e734b4cadbd1ca3c38a7b633e37ff`.
The upstream MIT license is reproduced in [`centaur/LICENSE`](centaur/LICENSE).
It was retrieved from that commit's `LICENSE.md`; its copyright notice is
preserved verbatim. The adaptation changes input geometry and the output class
count for the UAV tasks while retaining the separate two-stage model.

## Discounted Belief Fusion

`dsfn/cnn_dbf.py` contains the numerical fusion and evidential-loss adaptation
of [bezirganyan/DBF_uncertainty](https://github.com/bezirganyan/DBF_uncertainty),
associated with Bezirganyan et al., AISTATS 2025, PMLR 258:3142–3150.
The upstream GPL version 3 license is reproduced in [`dbf/LICENSE`](dbf/LICENSE),
retrieved from `main/LICENSE` on 2026-10-27. The experimental source records that
its upstream source was read on 2026-10-19 without an immutable commit pin; no
unverified commit identifier is claimed here.

The adaptation uses the DSFN-style CNN/patch frontend, independent linear
evidence heads, capped-exponential evidence, double-precision fusion and loss,
and class weighting for the evidential cross-entropy term. The original
directed conflict normalization and epsilon placements are retained. These
details describe the evaluated adaptation, not an exact reproduction of the
upstream benchmark. Keep this attribution and the GPL license with copies of
the adapted module.

## Other baselines and experiment provenance

`dsfn/baseline_models.py` retains the six baseline implementations used in the
revision experiments: ResNet1D, TCN, InceptionTime, LSTM-FCN, DRSN and MRA-CNN.
Their architecture names and source comments are preserved. The local
experiment records do not establish additional upstream license grants for
this file, so this notice does not invent any.

`dsfn/models.py` is extracted from the frozen residual CNN/GTD/CRCF definitions
in `basic_models_20261020.py` and `basic_no_scalar_models_20261020.py`. It retains
checkpoint parameter names and the final 130-input scoring network. The
training objective defaults to the revised manuscript's auxiliary and balance
weights of 0.1 each; the older 0.05/0.01 experiment defaults are not used.
