# Implementation provenance and verification

The revision replaces the dynamic CNN–Transformer encoder with three
fixed-depth residual dilated CNN branches. It retains gated feature-level
denoising (GTD) and confidence/consistency-based fusion (CRCF). The public
modules were extracted from the actual experiment source snapshots, with
machine-specific paths, experiment scheduling, and implicit checkpoint loading
removed. Importing the model package does not train a model or load data.

## Source mapping

| Public module | Experiment source |
| --- | --- |
| `dsfn/models.py` | `basic_models_20261020.py` and `basic_no_scalar_models_20261020.py` |
| `dsfn/baseline_models.py` | the frozen `source/baseline_models.py` used by those experiments |
| `dsfn/cnn_dbf.py` | numerical fusion and loss definitions in `cnn_dbf_baseline.py` |
| `dsfn/centaur.py` | `centaur_sensor_baseline_models_20261021.py`, with the three-class initialization from `uavfd_centaur_train.py` |

The `basic_` prefix is a historical source filename, not a change to the
datasets reported in the revised manuscript. The final UAV-FD worker imports
the no-scalar model definitions and supplies three output classes. The public
factory accepts the output class count explicitly.

The final base objective uses class-weighted cross-entropy plus auxiliary and
batch-weight-balance terms with coefficients **0.1 and 0.1**, as in the revised
experiment worker `uavfd_full_train.py`. Earlier 0.05/0.01 defaults from the
historical library are not the public defaults. The GTD loss preserves the
affected sample/modality normalization and the distinct denoising and identity
terms of the recorded source.

See [third-party attribution](../third_party/README.md) for Centaur and DBF
licenses and the scope of their adaptations.

## The final 130-input fusion scorer

The final scorer receives the 128-dimensional pooled feature, the auxiliary
classifier's maximum softmax probability, and mapped cross-modal cosine
consistency. Its dimensions are **130 → 64 → 1**. It has no dynamic-layer
selector, entropy-based stability input, or Transformer layer.

An earlier pure-CNN experiment kept an additional constant-one column in a
131-input first layer. The finalized source removes that column by folding its
weight into the bias at initialization:

```text
W_new = W_old[:, :130]
b_new = b_old + W_old[:, 130]
```

This preserves the initial affine function of the historical constant input.
The public factory retains the same temporary initialization draws and mapping
to match the finalized source's deterministic initial tensors. The returned
model stores only the 130-input layer. This is not a claim that an independently
trained 131-input model and a 130-input model have identical optimization
trajectories. Loading a historical 131-input checkpoint requires an explicit,
audited conversion; the factory does not silently convert checkpoint files.

For six output classes, the complete CNN–GTD–CRCF model stores **526,361**
parameters. For three output classes it stores **524,813**. The GTD base is
frozen during its separate training stage. Complete-input inference uses
correction scales `(0, 0.25, 1)`; a zero scale skips that adapter. When an entire
modality is zero-masked, the unavailable branch is excluded and all GTD adapters
are bypassed. Fusion weights are normalized over the available modalities.

## Migration checks performed

The model migration checks used fresh, synthetic tensors and PyTorch 2.5.1+cpu.
They did not load fitted weights or experimental datasets and did not run an
optimizer or repeat the formal training experiments.

Separate pipeline integration tests run one short epoch per stage on generated
signals for DSFN and Centaur, then save, reload, and evaluate all 22 conditions.
These disposable tests check that training and evaluation interfaces work
together. They are not the manuscript experiments and their metrics are not
reported as research results.

During extraction, the package was compared directly with the preserved
experiment source definitions at seed 42:

- For both six and three classes, every initial base-model tensor and every
  GTD-wrapper state tensor matched exactly by name, shape, and value.
- For complete input and each of the three single-modality masks, DSFN logits
  and all returned evidence fields matched the original implementation exactly.
- For both RflyMAD and UAV-FD, Centaur's autoencoder and classifier initial
  tensors matched the original dataset-specific construction exactly, including
  the six-class-to-three-class output-layer replacement for UAV-FD.
- All seven non-Centaur comparison models produced finite logits with the
  requested class count.

The source-to-package comparisons were migration checks against local frozen
source snapshots, not a reproduction of the paper's trained results. The
portable regression checks committed in `tests/test_model_protocol.py` use no
private source files and cover the parameter count, scorer width, initial GTD
identity, zero-scale skipping, masking weights and bypass behavior, rejection of
all-missing samples, output shapes, and the frozen base during GTD backward.

Run them from the repository root:

```bash
python -m unittest discover -s tests -p "test_model_protocol.py" -v
```

These checks establish implementation and interface consistency. They do not
establish dataset/checkpoint provenance, statistical significance, or bitwise
reproduction of GPU training. Reproducing the reported metrics also requires
the documented data partitions, preprocessing, noise realization, training
recipe, and checkpoint selection rule.
