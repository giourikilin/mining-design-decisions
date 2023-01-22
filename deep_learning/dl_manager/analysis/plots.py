import json
import os.path
import statistics

import matplotlib.pyplot as pyplot

from ..config import conf
from .util import *


def run_plot_attributes_command(show_plot=True):
    filename = conf.get('run_analysis.plot-attributes.file')
    include_early_stopping = conf.get('run_analysis.plot-attributes.trim')
    patience = conf.get('run_analysis.plot-attributes.patience')
    min_index = conf.get('run_analysis.plot-attributes.min_index')
    min_delta = conf.get('run_analysis.plot-attributes.min_delta')
    attributes = conf.get('run_analysis.plot-attributes.attributes')
    trim_attributes = conf.get('run_analysis.plot-attributes.trimming-attribute')
    min_delta = fix_min_delta(min_delta, trim_attributes)
    with open(filename) as file:
        runs = json.load(file)
    fig, axes = prompt_plot_arrangement(len(runs))
    for run, ax in zip(runs, axes):
        x_values = list(range(len(run['loss'])))
        for attribute in attributes:
            keys = attribute.split('|')
            current = run
            for key in keys:
                current = current[key]
            ax.plot(x_values, current, label=attribute)
        if include_early_stopping:
            index = get_trimming_index_multiple_attributes(
                run,
                patience,
                min_index,
                min_delta,
                get_early_stopping_settings(run),
                trim_attributes)
            if index is None:
                index = len(x_values) - 1
            ax.axvline(index)
        ax.legend(loc='lower right')
    fig.savefig('output.png')
    if show_plot:
        pyplot.show()


def run_bar_plot_command(show_plot=True):
    files = conf.get('run_analysis.plot.files')
    trim = conf.get('run_analysis.plot.trim')
    patience = conf.get('run_analysis.plot.patience')
    min_index = conf.get('run_analysis.plot.min_index')
    min_delta = conf.get('run_analysis.plot.min_delta')
    attributes = conf.get('run_analysis.plot.attributes')
    include_maxima = conf.get('run_analysis.plot.include-maxima')
    tolerance = conf.get('run_analysis.plot.tolerance')
    trim_attributes = conf.get('run_analysis.plot.trimming-attribute')
    plot_type = conf.get('run_analysis.plot.plot-type')
    min_delta = fix_min_delta(min_delta, trim_attributes)
    # First, load data
    kfold_runs = {}
    for filename in sorted(files):
        with open(filename) as file:
            data = json.load(file)
        if trim:
            data = [trim_run_result(run,
                                    patience,
                                    min_index,
                                    min_delta,
                                    get_early_stopping_settings(run),
                                    trim_attributes) for run in data]
        kfold_runs[get_number(os.path.split(filename)[1])] = data
        #kfold_runs[os.path.split(filename)[1].removeprefix('conv2d_metadata_combination_test__').removesuffix('.json')] = data
    # Get axes
    fig, axes = prompt_plot_arrangement(len(attributes))
    # Plot data per attribute
    objectives = get_objectives()
    x_labels = sorted(kfold_runs)
    x_values = list(range(1, len(x_labels) + 1))
    for ax, attribute in zip(axes, attributes):
        y_values = []
        for x_value in x_labels:
            kfold = kfold_runs[x_value]
            values = []
            for run in kfold:
                if '|' in attribute:
                    parent_key, child_key = attribute.split('|')
                    values.append(run[parent_key][child_key][-1])
                else:
                    values.append(run[attribute][-1])
            y_values.append(statistics.mean(values))
        if plot_type == 'line':
            x_values = [int(x) for x in x_labels]
            ax.plot(x_values, y_values)
        elif plot_type == 'bar':
            ax.bar(x_values, y_values)
            ax.set_xticks(x_values)
            ax.set_xticklabels(x_labels)
        ax.set_title(attribute)
        if include_maxima:
            if objectives[attribute] == 'max':
                maxima = find_local_maxima_within_epsilon(x_values,
                                                          y_values,
                                                          epsilon=tolerance)
            else:
                maxima = find_local_minima_within_epsilon(x_values,
                                                          y_values,
                                                          epsilon=tolerance)
            for x, y in maxima:
                ax.axvline(x, c='r')
                ax.text(x + 0.5, y / 2, f'({x}, {y:.3f})', rotation='vertical')
                #setattr(best_x.setdefault(x, Triple()), Triple.conv[attribute], y)
    fig.savefig('output.png')
    if show_plot:
        pyplot.show()


def find_local_maxima_within_epsilon(x_axis, y_axis, *, epsilon):
    global_max_y = max(y_axis)
    n = len(x_axis)
    maxima = []
    padded_x = [x_axis[0] - 1] + x_axis + [x_axis[-1] - 1]
    padded_y = [-float('inf')] + y_axis + [-float('inf')]
    for i in range(1, n + 1):
        if padded_y[i-1] < padded_y[i] and padded_y[i+1] < padded_y[i]:
            if abs(global_max_y - padded_y[i]) > epsilon:
                continue
            maxima.append((padded_x[i], padded_y[i]))
    return maxima


def find_local_minima_within_epsilon(x_axis, y_axis, *, epsilon):
    minima = find_local_maxima_within_epsilon(x_axis, [-y for y in y_axis], epsilon=epsilon)
    return [(x, -y) for x, y in minima]


def get_number(x: str):
    return int(''.join(filter(str.isdigit, x)))
