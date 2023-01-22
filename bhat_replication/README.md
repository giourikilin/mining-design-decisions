# Bhat Replication

---

This module describes our code we used for replicating the work of Bhat et al.

The important files in this module are:

- [classifiers.py](classifiers.py)
- [data_downloader.py](data_downloader.py)
- [ml.sh](ml.sh) / [ml.cmd](ml.cmd)
- [prepare_data.py](prepare_data.py)
- [requirements.txt](requirements.txt)
- [text_preprocessor.py](text_preprocessor.py)

All of these files will be further explained in the following text.

---

## Preparing Data

The first step in re-creating the experiments of Bhat et al., is 
acquiring their dataset. This can be done by running 

```shell 
python data_downloader.py
```

This will generate multiple files: `todo.txt`, `reduced.txt`, `reduced2.txt`,
and `issues.csv`. The former 3 files are debugging output and can be removed.
`issues.csv` contains the actual dataset.

An input file can also be generated in order to work with our dataset. 
This can be done by running 

```shell 
python prepare_data.py 
```

The output file will be named `ebse_issues_raw.csv`.

---

## Generating Feature Vectors 

Feature vectors are generated using the `text_preprocessor.py` script. 
By running `python text_preprocessor.py -h` you get the following help message:

```shell 
usage: text_preprocessor.py [-h] [--input-files INPUT_FILES [INPUT_FILES ...]] [-n N] [--myresources MYRESOURCES] [--cleanup-formatting] [--proper-multi-input]

options:
  -h, --help            show this help message and exit
  --input-files INPUT_FILES [INPUT_FILES ...]
                        Specify the path of the input csv file
  -n N                  Specify n in the n-grams
  --myresources MYRESOURCES
  --cleanup-formatting
  --proper-multi-input  If given, train TF/IDF on first input and apply transform on second
```

The `--input-files` argument is used to specify the input file(s) which are to
be converted to feature vectors. Normally, only a single file is given. 
This would result in three files being generated: `detection_data.json`, 
`classifying_data.json`, and `detect_and_classify__data.json`. 

Multiple `--input-files` may also be given. When `k` files are given, the first
`k-1` will be used as training data, while the final file will be used for testing
data. By default, the TF/IDF and CountVectorizer are trained on all `k` files,
instead of the `k-1` training data files. To change this, specify the 
`--proper-multi-input` flag. When using multiple input files, only a 
`detection_data_*.json` file is generated, where the `*` denotes a portion of 
the filename used to denote the source input files.

The `-n` flag is used to specify the value of `n` to use when generating `n`-grams. 
The `--cleanup-formatting` flag is used to enable the removal of formatting. 
This flag should always be given, unless explicitly testing without the removal
of formatting.

---

## Running the classifiers 

Run `ml.sh up` or `ml.cmd up` in order to start a local Spark cluster with a 
single worker. This Spark cluster will be used to run the machine learning code.

The machine learning process can be started by running `ml.sh submit` or 
`ml.cmd submit`. This will start the machine learning process with the settings 
specified in the [Dockerfile](Dockerfile). The arguments in the Dockerfile 
are passed to `classifiers.py`. The following arguments can be specified:

- `--algorithm [ALGORITHM]`: the algorithm to use. This must be one of
  `svm`, `dtree`, `logreg`, `bayes', or `onevsrest`.
- `--file [FILE]`: The input file to use. Use `detection_data.json` for detection,
    `classyifing_data.json` for classification as done by Bhat et al.,
    and `detect_and_classify__data.json` for classification by us
  (i.e. classification with the additional class "not design")
- `--split-size`: this parameter is ignored 
- `--max-from-class [MAX]`: maximum amount of samples allowed from any class. 
  Used for class-balancing as done by Bhat et al.
- `--benchmark [N]`: if given, enable a benchmark. This will cause `--algorithm` 
  to be ignored. In benchmark mode, all algorithms will be tested `N` times.

Note that, even when not running in benchmark mode, the script always performs 
10-fold cross-validation, unless running with an input `--file` generated using 
`text_preprocessor.py` with multiple `--input-files`,

Normally, the results of the test run are stored in `results/model_results.txt`.
When running in benchmark mode, the results are stored in 
`results/kfold_benchmark.json`.


