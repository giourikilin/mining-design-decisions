import json
import statistics

from ..config import conf

from .util import *


def run_summarize_command():
    filename = conf.get('run_analysis.summarize.file')
    perform_trimming = conf.get('run_analysis.summarize.trim')
    selection_only = conf.get('run_analysis.summarize.short')
    patience = conf.get('run_analysis.summarize.patience')
    min_index = conf.get('run_analysis.summarize.min_index')
    min_delta = conf.get('run_analysis.summarize.min_delta')
    trim_attributes = conf.get('run_analysis.summarize.trimming-attribute')
    min_delta = fix_min_delta(min_delta, trim_attributes)
    with open(filename) as file:
        runs = json.load(file)
    if perform_trimming:
        runs = [trim_run_result(run,
                                patience,
                                min_index,
                                min_delta,
                                get_early_stopping_settings(run),
                                trim_attributes) for run in runs]
    print('-' * 72)
    attributes = (
        sorted(get_plottable_metrics())
        if not selection_only else
        ('accuracy', 'f-score')
    )
    nested_keys = ('class-precision', 'class-recall', 'class-f-score')
    nested_keys += tuple(f'train-{x}' for x in nested_keys) + tuple(f'val-{x}' for x in nested_keys)
    for attribute in attributes:
        if attribute not in nested_keys:
            data = [run[attribute][-1] for run in runs]
            _show_averages(attribute, data)
        else:
            categories = tuple(runs[0][attribute])
            for category in categories:
                data = [run[attribute][category][-1] for run in runs]
                _show_averages(f'{attribute} ({category})', data)


def _show_averages(attribute, data):
    print(attribute.title())
    print(f' * Mean: {statistics.mean(data)}')
    print(f' * Standard Deviation: {statistics.stdev(data)}')
    print(f' * Median: {statistics.median(data)}')
    try:
        print(f' * Geometric Mean: {statistics.geometric_mean(data)}')
    except statistics.StatisticsError:
        print(f' * Geometric Mean: undefined')
    print(f' * LaTeX Ready: {statistics.mean(data):.4f} \\pm {statistics.stdev(data):.5f}')
    print('-' * 72)