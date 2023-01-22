import statistics

import matplotlib.pyplot as pyplot
import collections
import operator

##############################################################################
##############################################################################
# Constants
##############################################################################


def _base_objectives():
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
        'class-precision': 'max',
        'class-recall': 'max',
        'class-f-score': 'max'
    }


def get_objectives():
    train = {f'train-{key}': value for key, value in _base_objectives().items()}
    val = {f'val-{key}': value for key, value in _base_objectives().items()}
    test = _base_objectives()
    return train | test | val


def get_plottable_metrics() -> set[str]:
    return set(get_objectives())


def get_trimmable_metrics() -> set[str]:
    return get_plottable_metrics() | {'predictions'}

##############################################################################
##############################################################################
# Voting Integration
##############################################################################


def transform_voting_data(voting_data):
    # voting data = [
    # {
    #   "test": {}, "val": {}, "train": {}
    # }
    # ]
    return [_transform_voting_result(result) for result in voting_data]


def _transform_voting_result(voting_result):
    _compute_voting_attributes(voting_result)
    final_result = {}
    for key, results in voting_result.items():
        for sub_key, value in results.items():
            if key != 'test':
                final_key = f'{key}-{sub_key}'
            else:
                final_key = sub_key
            final_result[final_key] = [value]
    return final_result


def _compute_voting_attributes(data):
    attrs = ('precision', 'recall', 'f-score')
    keys = ('Existence', 'Property', 'Executive', 'Non-Architectural')
    for prefix in ('train', 'val', 'test'):
        for attr in attrs:
            #data_points.append(statistics.mean([run[sub_section[prefix]][key][attr] for key in keys]))
            #data[f'{prefix}{attr}'] = statistics.mean(data_points)
            data[prefix][attr] = statistics.mean(data[prefix][key][attr] for key in keys)



##############################################################################
##############################################################################
# Early-stopping analysis
##############################################################################


def trim_run_result(run_results,
                    patience,
                    min_index,
                    min_deltas,
                    settings,
                    trimming_attributes=('val-loss',)):
    index = get_trimming_index_multiple_attributes(run_results,
                                                   patience,
                                                   min_index,
                                                   min_deltas,
                                                   settings,
                                                   trimming_attributes)
    if index is not None:
        return _trim_recursive(run_results, index)
    return run_results


def get_trimming_index_multiple_attributes(run_results,
                                           patience,
                                           min_index,
                                           min_deltas,
                                           settings,
                                           trimming_attributes):
    if settings['stopped-early']:
        offset = settings['patience']
        index = len(run_results['val-loss']) - offset
        return index
    indices = []
    for min_delta, trimming_attribute in zip(min_deltas, trimming_attributes):
        index = get_trimming_index(run_results,
                                   patience,
                                   min_index,
                                   min_delta,
                                   trimming_attribute)
        if index is not None:
            indices.append(index)
    if not indices:
        return None
    return min(indices)


def get_trimming_index(run_results,
                       patience,
                       min_index,
                       min_delta,
                       trimming_attribute='val-loss'):
    # Because of the reverse nature of the check in
    # the loop body, we need gt for min and lt for max.
    cmp = (operator.gt
           if get_objectives()[trimming_attribute] == 'min'
           else operator.lt)
    values = run_results[trimming_attribute]
    if len(values) <= min_index:
        return None
    current_minimum = min_index
    while True:
        for i in range(patience):
            lookahead = current_minimum + i
            if lookahead == len(values):
                return None
            if cmp(values[lookahead], values[current_minimum]):
                continue
            if abs(values[lookahead] - values[current_minimum]) < min_delta:
                continue
            current_minimum = lookahead
            break
        else:
            break
    return current_minimum


def _trim_recursive(obj, index):
    if isinstance(obj, list):
        if not obj:
            return obj
        return obj[:index+1]
    return {
        key: (_trim_recursive(val, index)
              if key in get_trimmable_metrics()
              else val)
        for key, val in obj.items()
    }


def prompt_plot_arrangement(n: int):
    raw = input(f'How to arrange {n} plots [nrows, ncols]? ')
    nrows, ncols = [int(x) for x in raw.split()]
    fig, axes = pyplot.subplots(nrows=nrows, ncols=ncols, squeeze=False)
    flattened = []
    for row in axes:
        for col in row:
            flattened.append(col)
    return fig, flattened


def get_early_stopping_settings(run):
    try:
        return run['early-stopping-settings']
    except KeyError:
        return {'stopped-early': False}


def fix_min_delta(min_delta, trimming_attributes):
    if len(min_delta) != len(trimming_attributes):
        return [min_delta[0]] * len(trimming_attributes)
    return min_delta
