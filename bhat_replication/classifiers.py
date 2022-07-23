"""
Machine Learning Module based on Bhat et al ([1]).

This module allows selecting from a number of
machine learning algorithms.
Additionally, it is also possible to configure two
other parameters: the n in ngrams for the feature
vector generation, and the test/train data split.

[1] M. Bhat, K. Shumaiev, A. Biesdorf, U. Hohenstein,
    F. Matthes, A. Lopes, R. de Lemos,
    "Automatic Extraction of Design Decisions from Issue Management Systems:
     A Machine Learning Based Approach",
    Software Architecture, 2017, 138-154
"""

##############################################################################
##############################################################################
# Imports
##############################################################################

import argparse
import collections
import csv
import functools
import json
import os
import random
import typing

import numpy

from sklearn.metrics.pairwise import polynomial_kernel

import libsvm.svmutil

import pyspark
import pyspark.sql.functions
import pyspark.ml.linalg
import pyspark.ml
import pyspark.ml.classification
import pyspark.ml.feature
import pyspark.ml.evaluation

import collections
import functools
import random
import warnings


def stratified_kfold(k: int, labels: list) -> list:
    # Generate 1 additional split for the test set
    folds = _get_stratified_splits(k, labels)
    indices = frozenset(range(0, k))
    for test_set_index in range(0, k):
        training_set_indices = indices - {test_set_index}
        test_set = folds[test_set_index]
        training_set = functools.reduce(
            list.__add__,
            (folds[index] for index in training_set_indices),
            []
        )
        yield training_set, test_set


def _get_stratified_splits(k: int, labels: list) -> list:
    # Compute Label Indices
    indices_per_label = collections.defaultdict(list)
    for index, label in enumerate(labels):
        indices_per_label[label].append(index)
    # Now, compute the optimal amount of items from each
    # label per fold, where any "remaining" items are
    # distributed in a round-robin fashion.
    label_amounts_per_fold = [
        {
            label: len(indices) // k
            for label, indices in indices_per_label.items()
        }
        for _ in range(k)
    ]
    print('CHECKING BASELINE')
    for i in label_amounts_per_fold[1:]:
        assert i == label_amounts_per_fold[0]
    print('BASELINE OKAY. ALL FOLDS ARE EQUAL')
    # Now, distribute the remaining labels.
    # We choose the target folds semi-randomly,
    # making sure that we do not insert more items in
    # any given fold than the others.
    # This random fold assignment avoids the problem
    # that the train/test/validation set appear to
    # be more skewed later on, due to the fact that
    # consecutive bins would get the remaining labels
    # when using a round-robin approach.
    fold_index = 0
    #rng = UniformRoundRobin(range(k))
    for label, indices in indices_per_label.items():
        remaining = len(indices) % k
        print(f'LABEL {label} HAS {remaining} REMAINDERS')
        for _ in range(remaining):
            #fold_index = rng.get_next()
            label_amounts_per_fold[fold_index][label] += 1
            print(f'ADDING 1 SAMPLE TO FOLD {fold_index}')
            fold_index = (fold_index + 1) % k
    # Now, construct the actual folds by sampling from the
    # labels. We do not actually sample, but shuffle the
    # indices first and then popping from the lists.
    for indices in indices_per_label.values():
        random.shuffle(indices)
    folds = [_sample_from_labels(indices_per_label, label_amounts)
             for label_amounts in label_amounts_per_fold]
    # Shuffle all the folds
    for fold in folds:
        random.shuffle(fold)
    # Return result
    return folds


def _sample_from_labels(indices_per_label: dict,
                        label_amounts: dict) -> list:
    fold = []
    for label, amount in label_amounts.items():
        samples = [indices_per_label[label].pop() for _ in range(amount)]
        fold.extend(samples)
    return fold


class UniformRoundRobin:

    def __init__(self, numbers):
        self.__numbers = tuple(numbers)
        self.__choices = []

    def get_next(self):
        if not self.__choices:
            self.__choices = list(range(len(self.__numbers)))
            random.shuffle(self.__choices)
        return self.__numbers[self.__choices.pop()]


def convert_dictionaries(dicts):
    return [
        {int(key): value for key, value in d.items()}
        for d in dicts
    ]


##############################################################################
##############################################################################
# Main Function
##############################################################################


def main():
    parser = argparse.ArgumentParser(__doc__)
    algorithm_help = ('Machine learning algorithm to use. '
                      'Must be one of "svm", '
                      '"dtree", "logreg", "onevsrest", "bayes"')
    parser.add_argument('--algorithm', type=str,
                        help=algorithm_help)
    file_help = 'Path to the file of text data with labels, in JSON format.'
    parser.add_argument('--file', '-f', type=str,
                        help=file_help)
    parser.add_argument('--test-file', '-tf', type=str, default='',
                        help='Optional file containing test data. Assumes balanced data')
    parser.add_argument('-s', '--split-size', type=float,
                        help='Percentage of data to use for training.')
    parser.add_argument('--max-from-class', type=int,
                        help='Maximum amount of features from any given class')
    parser.add_argument('--benchmark', type=int, default=0,
                        help='Amount of benchmark runs')
    parser.add_argument('--cross-project', action='store_true', default=False,
                        help='If given, perform project cross validation.')
    args = parser.parse_args()

    # Load data
    with open(args.file) as file:
        data = json.load(file)
    vectors = numpy.asarray(convert_dictionaries(data['features']))
    labels = numpy.asarray(data['labels'])
    vector_size = data['vector_size']
    projects = data['projects']

    vectors, labels, projects = limit_size(vectors, labels, projects, args.max_from_class)

    if args.test_file:
        with open(args.test_file) as file:
            data = json.load(file)
        test_vectors = numpy.asarray(convert_dictionaries(data['features']))
        test_labels = numpy.asarray(data['labels'])
    else:
        test_vectors = test_labels = None

    if args.cross_project and test_vectors is not None:
        raise ValueError('Cannot have --test-file combined with --cross-project')

    # Select Algorithm
    functions = {
        'svm': run_svm_model,
        'dtree': run_dtree_model,
        'logreg': run_logistic_regression_model,
        'onevsrest': run_onevsrest_model,
        'bayes': run_naive_bayes_model
    }

    # Create Spark context
    spark = get_spark_context(get_env_settings())
    session = pyspark.sql.SparkSession(spark)

    if args.benchmark == 0:
        try:
            function = functions[args.algorithm]
        except KeyError:
            print(f'Unknown algorithm: {args.algorithm}')
        else:
            function(vectors.tolist(),
                     labels.tolist(),
                     args.split_size,
                     vector_size,
                     spark,
                     test_vectors.tolist() if test_vectors is not None else None,
                     test_labels.tolist() if test_labels is not None else None,
                     args.cross_project,
                     projects)
    else:
        results = {}
        for algorithm in functions:
            for idx in range(0, args.benchmark):
                function = functions[algorithm]
                results[algorithm] = function(
                    vectors.tolist(),
                    labels.tolist(),
                    args.split_size,
                    vector_size,
                    spark,
                    test_vectors.tolist() if test_vectors is not None else None,
                    test_labels.tolist() if test_labels is not None else None,
                    args.cross_project,
                    projects)
        with open('results/kfold_benchmark.json', 'w') as file:
            json.dump(results, file)


##############################################################################
##############################################################################
# Custom KFold Implementation
##############################################################################


def kfold(k, labels, training_portion):
    yield from stratified_kfold(k, labels)


def restore_split_to_fold(split):
    return round(10 * split)


def cross_project_iterator(vectors, labels, projects):
    bins = collections.defaultdict(list)
    for index, project in enumerate(projects):
        bins[project].append(index)
    for test_project, testing_indices in bins.items():
        training_projects = set(bins) - {test_project}
        training_indices = []
        for training_project in training_projects:
            training_indices.extend(bins[training_project])
        random.shuffle(testing_indices)
        random.shuffle(training_indices)
        yield test_project, (training_indices, testing_indices)


##############################################################################
##############################################################################
# Data Preparation
##############################################################################


def limit_size(vectors, labels, projects, max_class_size):
    assert len(vectors) == len(labels)
    assert len(vectors) == len(projects)
    classes = collections.defaultdict(list)
    for index, label in enumerate(labels):
        classes[label].append(index)

    indices = []
    for cls in classes.values():
        if len(cls) <= max_class_size:
            indices.extend(cls)
        else:
            indices.extend(
                random.sample(cls, max_class_size)
            )

    return vectors[indices], labels[indices], select_by_indices(projects, indices)


def sparse_to_dense(sparse: typing.Dict[int, float],
                    size: int) -> typing.List[float]:
    dense = [0.0] * size
    for index, value in sparse.items():
        dense[index] = value
    return dense


def numpy_to_sparse(
        dense: numpy.ndarray) -> typing.Tuple[typing.Dict[int, float], int]:
    sparse = {}
    for index, value in enumerate(dense):
        if value != 0:
            sparse[index] = float(value)
    return sparse, len(dense)


##############################################################################
##############################################################################
# Machine Learning Algorithms
##############################################################################


def run_svm_model(vectors, labels, split_size, vector_size, spark, test_vectors, test_labels, cross_project, projects):
    def svm_classifier(train_vectors, train_labels,
                       test_vectors, test_labels):
        # Train an SVM model with
        # SVM type C-SVC and kernel type linear.
        model = libsvm.svmutil.svm_train(train_labels,
                                         train_vectors,
                                         '-s 0 -t 0 -q')
        predictions = libsvm.svmutil.svm_predict(test_labels,
                                                 test_vectors,
                                                 model,
                                                 '-q')
        predicted_labels, _, _ = predictions
        return predicted_labels

    return run_spark_classifier(vectors,
                                labels,
                                split_size,
                                None,
                                'Support Vector Machine',
                                spark,
                                vector_size,
                                classifier_function=svm_classifier,
                                test_vectors=test_vectors,
                                test_labels=test_labels,
                                cross_project=cross_project,
                                projects=projects)


def run_dtree_model(vectors, labels, split_size, vector_size, spark, test_vectors, test_labels, cross_project, projects):
    import warnings
    warnings.warn('Criterion: gain ratio has not been checked')
    classifier = pyspark.ml.classification.DecisionTreeClassifier(
        labelCol="indexedLabel", featuresCol="features",
        maxDepth=20,
        impurity='entropy',     # entropy = change in gain
        minWeightFractionPerNode=0.25,
        minInfoGain=1)
    return run_spark_classifier(vectors,
                                labels,
                                split_size,
                                classifier,
                                'Decision Tree',
                                spark,
                                vector_size,
                                test_vectors=test_vectors,
                                test_labels=test_labels,
                                cross_project=cross_project,
                                projects=projects)


def run_logistic_regression_model(vectors, labels, split_size, vector_size, spark, test_vectors, test_labels, cross_project, projects):
    import warnings
    warnings.warn('Kernel: dot parameter has not been checked')
    classifier = pyspark.ml.classification.LogisticRegression(
        labelCol="indexedLabel", featuresCol="features",
        elasticNetParam=0.8, regParam=0.001, maxIter=10
    )

    def apply_kernel(data: typing.List[typing.Dict[int, float]], size):
        matrix = numpy.asarray(
            [sparse_to_dense(sparse, size) for sparse in data]
        )
        transformed = polynomial_kernel(
            matrix, matrix, degree=1, coef0=0
        )
        result = []
        for index in range(len(data)):
            sparse, _ = numpy_to_sparse(transformed[index,:])
            result.append(sparse)
        return result

    return run_spark_classifier(vectors,
                                labels,
                                split_size,
                                classifier,
                                'Logistic Regression',
                                spark,
                                vector_size,
                                data_transform=apply_kernel,
                                test_vectors=test_vectors,
                                test_labels=test_labels,
                                cross_project=cross_project,
                                projects=projects)


def run_onevsrest_model(vectors, labels, split_size, vector_size, spark, test_vectors, test_labels, cross_project, projects):
    import warnings
    warnings.warn('Not sure whether inner logit model params must be set')
    classifier = pyspark.ml.classification.OneVsRest(
        labelCol="indexedLabel",
        featuresCol="features",
        classifier=pyspark.ml.classification.LogisticRegression()
    )
    return run_spark_classifier(vectors,
                                labels,
                                split_size,
                                classifier,
                                'One-vs-Rest',
                                spark,
                                vector_size,
                                test_vectors=test_vectors,
                                test_labels=test_labels,
                                cross_project=cross_project,
                                projects=projects)


def run_naive_bayes_model(vectors, labels, split_size, vector_size, spark, test_vectors, test_labels, cross_project, projects):
    classifier = pyspark.ml.classification.NaiveBayes(
        labelCol="indexedLabel",
        featuresCol="features",
        smoothing=1
    )
    return run_spark_classifier(vectors,
                                labels,
                                split_size,
                                classifier,
                                'Naive Bayes',
                                spark,
                                vector_size,
                                test_vectors=test_vectors,
                                test_labels=test_labels,
                                cross_project=cross_project,
                                projects=projects)


benchmark_file = open('results/benchmark.csv', 'w+')
benchmark_writer = csv.writer(benchmark_file)
benchmark_header_written = False


def select_by_indices(array, indices):
    return [array[index] for index in indices]


def run_spark_classifier(vectors,
                         labels,
                         split_size,
                         classifier,
                         name,
                         spark,
                         vector_size,
                         *,
                         classifier_function=None,
                         data_transform=None,
                         test_vectors=None,
                         test_labels=None,
                         cross_project: bool = False,
                         projects: list = None) -> list:
    results = []
    # Transform data, if applicable
    if data_transform is not None:
        vectors = data_transform(vectors, vector_size)
        if test_vectors is not None:
            test_vectors = data_transform(test_vectors, vector_size)
    # Load data once to fit indexer
    pairs = list(zip(vectors, labels))
    data = spark.parallelize(pairs).toDF(['features', 'label'])
    indexer = pyspark.ml.feature.StringIndexer(
        inputCol='label', outputCol='indexedLabel').fit(data)
    classes = indexer.labels
    # KFold loop
    if test_vectors is None:
        training_size = restore_split_to_fold(split_size)
        if not cross_project:
            stream = enumerate(kfold(10, labels, training_size))
        else:
            stream = cross_project_iterator(vectors, labels, projects)
        for fold, (training_indices, testing_indices) in stream:
            print(labels[0:10])
            assert set(training_indices) & set(testing_indices) == set()
            # Load data into Spark
            train = sparkify_data(spark,
                                  select_by_indices(vectors, training_indices),
                                  select_by_indices(labels, training_indices),
                                  vector_size)
            test = sparkify_data(spark,
                                 select_by_indices(vectors, testing_indices),
                                 select_by_indices(labels, testing_indices),
                                 vector_size)
            predictions = run_classifier(spark,
                                         classifier, classifier_function,
                                         indexer,
                                         train, test)
            results.append(
                {
                    "results": evaluate_model(name, classes, predictions),
                    "fold": fold
                }
            )
    else:
        training_data = sparkify_data(spark, vectors, labels, vector_size)
        testing_data = sparkify_data(spark, test_vectors, test_labels, vector_size)
        predictions = run_classifier(spark, classifier, classifier_function,
                                     indexer, training_data, testing_data)
        results.append(
            evaluate_model(name, classes, predictions)
        )
    return results


def sparkify_data(spark, vectors, labels, vector_size):
    # Setup udf
    array_to_vector = pyspark.sql.functions.udf(
        lambda vs: pyspark.ml.linalg.Vectors.sparse(vector_size, vs),
        pyspark.ml.linalg.VectorUDT()
    )
    # Load data
    pairs = list(zip(vectors, labels))
    data = spark.parallelize(pairs).toDF(['features', 'label'])
    # Prepare data transformations
    data = data.withColumn(
        'featureVector',
        array_to_vector(pyspark.sql.functions.col('features'))
    )
    return data


def run_classifier(spark,
                   classifier, classifier_function,
                   indexer,
                   train, test) -> pyspark.sql.DataFrame:
    if classifier is not None:
        classifier.setFeaturesCol('featureVector')
        # Setup pipeline
        pipeline = pyspark.ml.Pipeline(stages=[indexer, classifier])
        # Fit model
        model = pipeline.fit(train)
        # Make Predictions
        predictions = model.transform(test)
    else:
        assert classifier_function is not None
        train_pairs = indexer.transform(train).select(
            pyspark.sql.functions.col('features'),
            pyspark.sql.functions.col('indexedLabel')
        ).collect()
        # Transpose
        train_vectors, train_labels = zip(*train_pairs)
        test_pairs = indexer.transform(test).select(
            pyspark.sql.functions.col('features'),
            pyspark.sql.functions.col('indexedLabel')
        ).collect()
        test_vectors, test_labels = zip(*test_pairs)
        predicted_labels = classifier_function(train_vectors,
                                               train_labels,
                                               test_vectors,
                                               test_labels)
        predictions = spark.parallelize(
            list(zip(test_labels, predicted_labels))
        ).toDF(['indexedLabel', 'prediction'])
    return predictions


def evaluate_model(name: str,
                   classes: list,
                   predictions: pyspark.sql.DataFrame) -> dict:
    json_result = {}
    results = []
    benchmark_header = ['model', 'accuracy', 'accuracy (f1)']
    with open('./results/model_results.txt', 'w') as file:
        evaluator = pyspark.ml.evaluation.MulticlassClassificationEvaluator(
            labelCol="indexedLabel",
            predictionCol="prediction")
        accuracy = evaluator.evaluate(predictions,
                                      {evaluator.metricName: 'accuracy'})
        f1 = evaluator.evaluate(predictions, {evaluator.metricName: 'f1'})
        print(f'Results for model type: {name}', file=file)
        results.append(name)
        print(f'    * Accuracy: {accuracy}', file=file)
        json_result['accuracy'] = accuracy
        print(f'    * Accuracy (F1): {f1}', file=file)
        json_result['accuracy (f1)'] = f1
        results.append(accuracy)
        results.append(f1)
        metrics = ('precisionByLabel', 'recallByLabel', 'fMeasureByLabel')
        for index, cls in enumerate(classes):
            for metric in metrics:
                print('>' * 70, index)
                value = evaluator.evaluate(predictions,
                                           {evaluator.metricName: metric,
                                            evaluator.metricLabel: index})
                print(f'    * {cls} {metric}: {value}', file=file)
                json_result[f'{metric}__{cls}'] = value
                results.append(value)
                benchmark_header.append(f'{cls} {metric}')

    global benchmark_header_written
    if not benchmark_header_written:
        benchmark_writer.writerow(benchmark_header)
        benchmark_header_written = True
    benchmark_writer.writerow(results)
    return json_result


##############################################################################
##############################################################################
# Spark Setup
##############################################################################


class Environment(typing.NamedTuple):
    spark_master_url: str


def get_env_settings() -> Environment:
    return Environment(
        spark_master_url=os.environ['SPARK_MASTER_URL']
    )


def get_spark_context(env: Environment) -> pyspark.SparkContext:
    config = pyspark.SparkConf().\
        setAppName('ml').setMaster(env.spark_master_url).set("spark.executor.memory", "4g")
    return pyspark.SparkContext(conf=config)


##############################################################################
##############################################################################
# Program Entry Point
##############################################################################


if __name__ == '__main__':
    main()
