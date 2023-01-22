import collections
import json
import statistics
import os.path

import scipy.stats
import texttable

from ..config import conf
from .util import *


def run_stat_command():
    files = conf.get('run_analysis.compare-stats.files')
    trim = conf.get('run_analysis.compare-stats.trim')
    patience = conf.get('run_analysis.compare-stats.patience')
    min_index = conf.get('run_analysis.compare-stats.min_index')
    min_delta = conf.get('run_analysis.compare-stats.min_delta')
    attribute = conf.get('run_analysis.compare-stats.attribute')
    trim_attributes = conf.get('run_analysis.compare-stats.trimming-attribute')
    min_delta = fix_min_delta(min_delta, trim_attributes)
    # First, load files
    kfold_runs = {}
    for filename in sorted(files):
        with open(filename) as file:
            data = json.load(file)
        if trim and 'voting' not in filename:
            data = [trim_run_result(run,
                                    patience,
                                    min_index,
                                    min_delta,
                                    get_early_stopping_settings(run),
                                    trim_attributes) for run in data]
        elif 'voting' in filename:
            data = transform_voting_data(data)
        kfold_runs[os.path.split(filename)[1]] = data
    # Collect data
    results = {}

    for key, kfold in kfold_runs.items():
        if '|' in attribute:
            parent, child = attribute.split('|')
            # mean = statistics.mean([run[parent][child][-1] for run in kfold])
            # std = statistics.stdev([run[parent][child][-1] for run in kfold])
            data = [run[parent][child][-1] for run in kfold]
        else:
            # mean = statistics.mean([run[attribute][-1] for run in kfold])
            # std = statistics.stdev([run[attribute][-1] for run in kfold])
            data = [run[attribute][-1] for run in kfold]
        results[key] = data

    runs = list(results)
    project_mapping = {run: str(i) for i, run in enumerate(runs, start=1)}

    for run_name, run_number in project_mapping.items():
        data = results[run_name]
        mu = statistics.mean(data)
        std = statistics.stdev(data)
        p = scipy.stats.shapiro(data).pvalue
        print(f'{run_number.rjust(3)}. {run_name}: mean = {mu} -- std = {std} -- normality: p = {p}')

    table = texttable.Texttable()
    table.header(['P/P'] + [project_mapping[r] for r in runs])
    for row_project in runs:
        row = [project_mapping[row_project]]
        for col_project in runs:
            row.append(str(_test_attr(results[row_project], results[col_project])))
        table.add_row(row)

    print(table.draw())


def _test_attr(data_a, data_b):
    #return scipy.stats.ttest_ind(data_a, data_b, equal_var=False).pvalue
    if data_a == data_b:
        return 1
    return scipy.stats.wilcoxon(data_a, data_b).pvalue

