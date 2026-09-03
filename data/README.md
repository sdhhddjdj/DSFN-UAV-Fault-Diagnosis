# Data

The RflyMAD dataset is not redistributed in this repository. Download the
sensor and no-fault archives from the official RflyMAD dataset page:

https://rfly-openha.github.io/documents/4_resources/dataset.html

Use `prepare_rflymad.py` to convert the original flight logs into 9-channel,
1,024-step windows with a stride of 512. By default, the script reproduces the
manuscript's stratified window-level 80%/10%/10% split. A separate
`--split-mode flight` option is available for grouped-split studies but was not
used to obtain the results reported in the manuscript.

Generated arrays and split files should be stored under `data/processed/`.
That directory is excluded from Git because the dataset must be obtained from
its original publisher and the generated files are large.
