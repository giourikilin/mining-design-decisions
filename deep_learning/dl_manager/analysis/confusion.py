import json

import seaborn
import matplotlib.pyplot as pyplot
from sklearn.metrics import confusion_matrix

from ..config import conf
from .util import *


def run_confusion_matrix_command():
    filename = conf.get('run_analysis.confusion.file')
    trim = conf.get('run_analysis.confusion.trim')
    patience = conf.get('run_analysis.confusion.patience')
    min_index = conf.get('run_analysis.confusion.min_index')
    min_delta = conf.get('run_analysis.confusion.min_delta')
    trim_attributes = conf.get('run_analysis.confusion.trimming-attribute')
    min_delta = fix_min_delta(min_delta, trim_attributes)
    with open(filename) as file:
        runs = json.load(file)
    if trim:
        runs = [trim_run_result(run,
                                patience,
                                min_index,
                                min_delta,
                                get_early_stopping_settings(run),
                                trim_attributes) for run in runs]
    if 'classes' not in runs[0]:
        # Legacy version; make a simple confusion matrix
        matrices = []
        for run in runs:
            data = [
                [run['tp'][-1], run['fn'][-1]],
                [run['fp'][-1], run['tn'][-1]]
            ]
            print('Arch -> Arch', run['tp'][-1])
            run['classes'] = ['Architectural', 'Non-Architectural']
            matrices.append(data)
    else:
        matrices = []
        for run in runs:
            class_dict = {int(k): v for k, v in run['classes'].items()}
            truth = [class_dict[int(x)] for x in run['truth']]
            predicted = [class_dict[int(x)] for x in run['predictions'][-1]]
            classes = [x for x in run['classes'].values()]
            matrices.append(confusion_matrix(truth, predicted, labels=classes))
    fig, axes = prompt_plot_arrangement(len(matrices))
    #seaborn.color_palette("viridis", as_cmap=True)
    for ax, matrix, run in zip(axes, matrices, runs):
        plot = seaborn.heatmap(matrix,
                               ax=ax,
                               annot=True,
                               xticklabels=run['classes'],
                               yticklabels=run['classes'],
                               cmap='viridis')
        plot.set(xlabel='Predicted', ylabel='Actual')
    pyplot.show()
