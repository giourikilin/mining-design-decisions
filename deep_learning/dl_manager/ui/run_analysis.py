
import collections
import json
import os.path
import statistics

import matplotlib.pyplot as pyplot


##############################################################################
##############################################################################
# Command Implementation
##############################################################################


def run_average_command(filename: str,
                        perform_trimming: bool,
                        selection_only: bool,
                        patience: int,
                        min_index: int):
    with open(filename) as file:
        runs = json.load(file)
    if perform_trimming:
        runs = [trim_run_result(run, patience, min_index) for run in runs]
    print('-' * 72)
    attributes = (
        sorted(runs[0])
        if not selection_only else
        ('accuracy', 'precision', 'recall', 'f-score')
    )
    nested_keys = ('class-precision', 'class-recall', 'class-f-score')
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
    print('-' * 72)


def run_plot_attributes_command(filename: str,
                                attributes: list[str],
                                include_early_stopping: bool,
                                patience: int,
                                min_index: int):
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
            index = get_trimming_index(run, patience, min_index)
            if index is None:
                index = len(x_values) - 1
            ax.axvline(index)
        ax.legend(loc='lower right')
    pyplot.show()


def run_bar_plot_command(files: list[str],
                         attributes: list[str],
                         trim: bool,
                         patience: int,
                         min_index: int,
                         include_maxima: bool,
                         tolerance: float):
    # First, load data
    kfold_runs = {}
    for filename in sorted(files):
        with open(filename) as file:
            data = json.load(file)
        if trim:
            data = [trim_run_result(run, patience, min_index) for run in data]
        kfold_runs[get_number(os.path.split(filename)[1])] = data
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
        ax.bar(x_values, y_values)
        ax.set_title(attribute)
        ax.set_xticks(x_values)
        ax.set_xticklabels(x_labels)
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
    return fig, axes


def run_comparison_command(files: list[str],
                           attributes: list[str],
                           trim: bool,
                           patience: int,
                           min_index: int):
    # First, load files
    kfold_runs = {}
    for filename in sorted(files):
        with open(filename) as file:
            data = json.load(file)
        if trim:
            data = [trim_run_result(run, patience, min_index) for run in data]
        kfold_runs[os.path.split(filename)[1]] = data
    # Collect data
    results = collections.defaultdict(list)
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


def get_objectives():
    return {
        'fp': 'min',
        'fn': 'min',
        'tp': 'max',
        'tn': 'max',
        'accuracy': 'max',
        'precision': 'max',
        'recall': 'max',
        'f-score': 'max',
        'loss': 'min',
        'val-loss': 'min',
        'train-loss': 'min',
        'class-precision': 'max',
        'class-recall': 'max',
        'class-f-score': 'max'
    }

##############################################################################
##############################################################################
# Utility Functionality
##############################################################################


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


def prompt_plot_arrangement(n: int):
    nrows = n
    ncols = 1
    fig, axes = pyplot.subplots(nrows=nrows, ncols=ncols, squeeze=False)
    flattened = []
    for row in axes:
        for col in row:
            flattened.append(col)
    return fig, flattened


def trim_run_result(run_results, patience, min_index):
    if (index := get_trimming_index(run_results, patience, min_index)) is not None:
        return _trim_recursive(run_results, index)
    return run_results


def get_trimming_index(run_results, patience, min_index):
    trimming_attribute = 'val-loss'
    values = run_results[trimming_attribute]
    traces = [values] + [values[i:] for i in range(patience, patience + 1)]
    for index, chain in enumerate(zip(*traces)):
        if index <= min_index:
            continue
        prev = chain[0]
        for current in chain[1:]:
            if current < prev:
                break
            prev = current
        else:
            return index
    return None


def _trim_recursive(obj, index):
    if isinstance(obj, list):
        return obj[:index+1]
    return {
        key: _trim_recursive(val, index) for key, val in obj.items()
    }



