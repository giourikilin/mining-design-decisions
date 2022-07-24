# This file contains information on how to run the programs in this directory.

## get_jira_models.py
This script can be used to gather issue text data for training Word2Vec and Doc2Vec models.
Some code is inside comments, which can be used to configure the data collecting and training process of the models.
Things that can be configured include:
- either collect issue data from a certain set of project, or collect data for the issues provided in an issue labels file.
- use lemmatization, stemming or none
- use ontology classes and/or lexical triggers
- set the vector size
- set the minimum count (used for training the models)
- output filename

This script does not require any parameters in order to run it.

## issuedata_extractor.py
This script can be used to collect the raw data of issues. It collects the text data, such as the summary, description and the comments.
It also collects issue properties (the full list can be found inside the script).
It can either collect data from a certain set of projects or it can gather the data for the issue keys provided in a label list.

This script also does not require any parameters in order to run it.

## preprocess_to_json.py
After running `issuedata_extractor.py`, the data has to be put in a format that can be used by the `dl_manager`.
This script can be used for that. In the main function you should specify the input file (collected using the previous script),
the output filename, the file with the labels of the issues, and the name of the dataset.
The type of formatting should also be specified (markers or keep). This has to be changed manually in the script.

In order to provide support for the cross-dataset benchmark, two datasets have to be preprocessed simultaneously.
We used the naming `study` for this, since the two datasets are from two different studies in our case. In the code we
therefore refer to the name of the `study` and we also use the `--test-study` parameter in the `dl_manager` for this reason.

After running the `preprocess_to_json.py` script, the generated output files (JSON lists of issues) have to be combined for the cross-dataset benchmark.
This combined file is used as input to `dl_manager`. NOTE: this is only required for the cross-dataset benchmark.

This script also does not require any parameters in order to run it.