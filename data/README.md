# Paper data

Place the processed `train_data.npy` and `train_labels.npy` files used by the
manuscript in a source directory, then run:

```bash
python prepare_paper_data.py --source-dir /path/to/processed/dataset
```

The command verifies the expected files and copies them to `data/paper/`.
Do not rename or reorder the arrays. The repository includes the small manifest
and fixed protocol files used by the training and evaluation entry points.

The processed arrays are derived from the Real-Sensors and Real-No Fault
subsets available on the
[official RflyMAD dataset page](https://rfly-openha.github.io/documents/4_resources/dataset.html).
The large arrays, trained checkpoints and generated result files are not stored
in Git.
