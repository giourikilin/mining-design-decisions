from sklearn.model_selection import StratifiedKFold as _StratifiedKFold
import numpy


class StratifiedKFold:

    def __init__(self, k):
        self.__k = k
        self.__kfold = _StratifiedKFold(n_splits=self.__k)

    def split(self, x, y):
        labels = self.__simplify_labels(y)
        return self.__kfold.split(x, labels)

    def __simplify_labels(self, labels):
        return list(self.__simplify_labels_iter(labels))

    def __simplify_labels_iter(self, labels):
        label_mapping = {}
        next_number = 0
        for label in labels:
            key = self.__to_tuple(label)
            if key not in label_mapping:
                label_mapping[key] = next_number
                next_number += 1
            yield label_mapping[key]

    def __to_tuple(self, x):
        if not isinstance(x, (list, tuple, numpy.ndarray)):
            return x
        return tuple(self.__to_tuple(y) for y in x)
