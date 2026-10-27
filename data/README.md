# Data preparation for the revised protocol

Obtain source logs from the dataset providers:

- [RflyMAD and its processing tools](https://rfly-openha.github.io/documents/4_resources/dataset.html).
- [UAV-FD, Zenodo record 7648996](https://doi.org/10.5281/zenodo.7648996).

Raw logs, arrays and checkpoints are not distributed here. Earlier window-level
`data/paper_split` files belong to the original submission and must not be used for
the revision. Git history retains that version.

## Canonical flight import

Convert each complete flight log into `<log_id>.npz` with:

| Key | Meaning |
| --- | --- |
| `signals` | finite `float32 [time,9]` in Gyro XYZ, Accel XYZ, Velocity NED order |
| `label` | integer scalar for a flight with one diagnostic class |
| `labels` | alternatively, integer `[time]` labels aligned with `signals` |
| `log_id` | optional scalar string, checked against the manifest |
| `group_id` | optional scalar string, checked against the manifest |

The importer does not synchronize, interpolate, rotate or resample raw streams.
Use the dataset's processing tools first. In particular, the registered UAV-FD
comparison uses aligned 350 Hz channels; a different conversion is not an exact
replication of the prepared paper data.

Make `flight_manifest.csv` **before extracting windows**:

```csv
log_id,group_id,split
example_train_flight,example_case_1,train
example_validation_flight,example_case_2,val
example_test_flight,example_case_3,test
```

These examples describe the format, not the paper's actual assignments. Keep all
associated logs of a flight case in one group and partition. For UAV-FD retain
the registered fault-motor-position/log grouping in [PROTOCOL.md](../docs/PROTOCOL.md),
rather than interpreting the split as geographic sites.

```bash
python prepare_paper_data.py --dataset rflymad \
  --flights-dir /path/to/canonical_flights --manifest /path/to/flight_manifest.csv \
  --output-dir data/rflymad

python prepare_paper_data.py --dataset uavfd \
  --flights-dir /path/to/canonical_flights --manifest /path/to/flight_manifest.csv \
  --output-dir data/uavfd

python prepare_paper_data.py --output-dir data/uavfd --verify-only
```

Every window contains 1,024 samples. Training stride is 512 for both datasets.
RflyMAD validation/test strides are 1,024. UAV-FD validation stride is 512 and test
stride is 1,024. Windows crossing label transitions are excluded and their number
is recorded in `dataset.json`.

Outputs are `windows.npy [N,9,1024]`, `labels.npy [N]`, `groups.npy [N]`,
`splits.json`, `dataset.json`, a copied manifest and `window_metadata.csv`.
Existing nonempty output directories are never overwritten.

## Validation and normalization

`dsfn.data.load_dataset(root)` checks that each window occurs in one partition,
and that no flight-case group or metadata log crosses partitions. Metadata is
checked for nonoverlapping test windows and nonoverlapping RflyMAD validation
windows. Means, population standard deviations and inverse-frequency class
weights are fitted only on training windows.

The paper used 3,207/203/210 RflyMAD and 555/123/68 UAV-FD windows for
train/validation/test. Matching counts alone does not establish matching flight
IDs, preprocessing or labels. Supply the registered assignments and conversion to
reproduce reported results. The importer cannot recover them from previously
segmented or randomly shuffled arrays.

## Official UAV-FD MAT conversion

`prepare_uavfd.py` ports the archived 350 Hz alignment and event/PWM filtering.
It does not concatenate disjoint airborne segments before windowing. For the
final test set it keeps alternating windows within each continuous chain of valid
512-step windows and restarts after a gap. This is not equivalent to testing
`start_sample % 1024 == 0`. The resulting counts must be 555/123/68.

```bash
python prepare_uavfd.py --raw-dir /path/to/official_18_mat_files --output-dir data/uavfd
```

The default is offline; add `--download` to fetch missing official MAT files.
The released `manifests/uavfd_flights.csv` also supplies compatible `log_id`,
`group_id`, and `split` columns for the generic canonical importer.

The MAT converter records `test_noise_original_count`, `test_noise_positions`,
and `test_noise_seed` in `dataset.json`. Evaluation regenerates the original
134-window Gaussian bank and then selects the 68 corresponding rows, matching
the final test-only re-evaluation rather than drawing a new 68-window bank.
