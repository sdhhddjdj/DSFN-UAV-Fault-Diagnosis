# Data

The RflyMAD dataset is not redistributed in this repository. Download the
sensor and no-fault archives from the official RflyMAD dataset page:

https://rfly-openha.github.io/documents/4_resources/dataset.html

Use `prepare_rflymad.py` to convert the original flight logs into 9-channel,
1,024-step windows. The script groups complete flight logs before creating the
train, validation, and test partitions. This prevents windows from the same
flight appearing in more than one partition.

Generated arrays and split files should be stored under `data/processed/`.
That directory is excluded from Git because the dataset must be obtained from
its original publisher and the generated files are large.
