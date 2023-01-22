import math

import numpy

from .classifiers import OutputEncoding
from .feature_generators import OutputMode
from . import metrics

from .config import conf


def check_adaboost_requirements():
    output_mode = conf.get('run.output_mode')
    if output_mode == OutputMode.Classification3:
        raise NotImplementedError('Adaboost not implemented for Classification3')


def initialize_weights(labels) -> numpy.ndarray:
    return numpy.ones(labels.shape) / len(labels)


def compute_error(y_true: numpy.ndarray,
                  y_pred: numpy.ndarray,
                  weights: numpy.ndarray) -> float:
    y_true, y_pred = _prepare_arrays(y_true, y_pred)
    return (y_true != y_pred).dot(weights).sum() / weights.sum()


def compute_classifier_weight(error: float, k: int) -> float:
    return math.log((1 - error) / error) + math.log(k - 1)


def update_weights(y_true: numpy.ndarray,
                   y_pred: numpy.ndarray,
                   weights: numpy.ndarray,
                   alpha: float) -> numpy.ndarray:
    y_true, y_pred = _prepare_arrays(y_true, y_pred)
    correction = numpy.exp(alpha * (y_true != y_pred))
    return _normalize(numpy.multiply(weights, correction))


def _normalize(array):
    norm = numpy.linalg.norm(array)
    return array / norm


def _prepare_arrays(y_true: numpy.ndarray,
                    y_pred: numpy.ndarray) -> (numpy.ndarray, numpy.ndarray):
    output_mode = OutputMode.from_string(conf.get('run.output_mode'))
    if output_mode.output_encoding == OutputEncoding.Binary:
        y_pred = y_pred.ravel()
    else:
        assert output_mode.output_encoding == OutputEncoding.OneHot
        # Convert to 1D arrays
        y_pred = numpy.argmax(y_pred, axis=1)
        y_true = numpy.argmax(y_true, axis=1)
    return y_true, y_pred


def compute_final_classifications(alphas, number_of_labels, *predictions):
    # First, compute 1D arrays
    converted_predictions = []
    output_mode = OutputMode.from_string(conf.get('run.output_mode'))
    for prediction_set in predictions:
        if output_mode.output_encoding == OutputEncoding.Binary:
            converted_predictions.append(
                metrics.round_binary_predictions(prediction_set)
            )
        else:   # One-hot
            converted_predictions.append(
                metrics.onehot_indices(prediction_set)
            )

    stacked_predictions = numpy.vstack(
        converted_predictions
    ).transpose()
    bitmaps = [stacked_predictions == k
               for k in range(number_of_labels)]
    weighted_maps = [numpy.multiply(alphas, bitmap)
                     for bitmap in bitmaps]
    weighted_sums = [weighted_map.sum(axis=1)
                     for weighted_map in weighted_maps]
    matrix = numpy.vstack(weighted_sums).transpose()
    predictions = matrix.argmax(axis=1)
    return predictions


