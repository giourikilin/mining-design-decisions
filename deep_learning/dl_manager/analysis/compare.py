import collections
import json
import statistics
import os.path

from ..config import conf
from .util import *


def run_comparison_command():
    files = conf.get('run_analysis.compare.files')
    trim = conf.get('run_analysis.compare.trim')
    patience = conf.get('run_analysis.compare.patience')
    min_index = conf.get('run_analysis.compare.min_index')
    min_delta = conf.get('run_analysis.compare.min_delta')
    attributes = conf.get('run_analysis.compare.attributes')
    trim_attributes = conf.get('run_analysis.compare.trimming-attribute')
    max_attribute = conf.get('run_analysis.compare.max-attribute')
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
    results = collections.defaultdict(list)

    if max_attribute != 'None':
        # find best run for selected attribute
        max_idxs = []
        for key, kfold in kfold_runs.items():
            value = max([run[max_attribute][-1] for run in kfold])
            max_idxs.append([run[max_attribute][-1] for run in kfold].index(value))

        for attribute in attributes:
            for (key, kfold), max_idx in zip(kfold_runs.items(), max_idxs):
                if '|' in attribute:
                    parent, child = attribute.split('|')
                    value = [run[parent][child][-1] for run in kfold][max_idx]
                else:
                    value = [run[attribute][-1] for run in kfold][max_idx]
                results[key].append(value)
    else:
        for attribute in attributes:
            for key, kfold in kfold_runs.items():
                if '|' in attribute:
                    parent, child = attribute.split('|')
                    value = statistics.mean([run[parent][child][-1] for run in kfold])
                else:
                    value = statistics.mean([run[attribute][-1] for run in kfold])
                results[key].append(value)

    # Generate sort key
    objectives = get_objectives()
    sorting_key = []
    for attribute in attributes:
        if '|' in attribute:
            attribute, category = attribute.split('|')
            objectives[f'{attribute}|{category}'] = objectives[attribute]
        sorting_key.append(objectives[attribute])
    # Now, sort the stuff
    rows = [
        [v if obj == 'max' else -v for obj, v in zip(sorting_key, value)] + [key]
        for key, value in results.items()
    ]
    rows.sort()
    # Finally, print everything
    longest_name = len(max(rows, key=lambda x: len(x[-1]))[-1])
    for *values, name in rows:
        formatted = ' | '.join(f'{attr}: {value if objectives[attr] == "max" else -value:.4f}'
                               for attr, value in zip(attributes, values))
        print(f'{name.ljust(longest_name)} -- {formatted}')
