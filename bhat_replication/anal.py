import collections
import json
import statistics


def analyze(files):
    data = []
    for filename in files:
        with open(filename) as file:
            data.append(json.load(file))
    scores_per_classifier = collections.defaultdict(list)
    for n, result_set in enumerate(data, start=1):
        for key, folds in result_set.items():
            scores = []
            for fold in folds:
                #scores[(key, fold['fold'], n)].append(fold['results']['accuracy (f1)'])
                scores.append(fold['results']['accuracy (f1)'])
            #print(f'{key.capitalize()} (ngram = {n}): {statistics.mean(scores)}')
            scores_per_classifier[key].append((n, statistics.mean(scores)))
    for key, scores in scores_per_classifier.items():
        best_n, best_score = max(scores, key=lambda x: x[1])
        print(f'{key.capitalize()} (ngram = {best_n}): {best_score}')
    # for (classifier, project, n), values in scores.items():
    #     best_score, best_n, classifier = max(values, key=lambda x: x[0])
    #     print(f'Project {key}: {best_score} (classifier: {classifier}; n-grams: {best_n})')


print('Detection Cross Project EBSE')
analyze([
    f'ebse_cross_project_detection_n{i}.json' for i in range(1, 6)
])

print('Classification Cross Project EBSE')
analyze([
    f'ebse_cross_project_classification_n{i}.json' for i in range(1, 6)
])

print('Detection Cross Project BHAT')
analyze([
    f'bhat_cross_project_detection_n{i}.json' for i in range(1, 6)
])

print('Classification Cross Project EBSE')
analyze([
    f'bhat_cross_project_classification_n{i}.json' for i in range(1, 6)
])

print('EBSE Detection')
analyze([
    f'ebse_detection_kfold_benchmark_n{i}_s59.json' for i in range(1, 6)
])

print('EBSE Classification')
analyze([
    f'ebse_classification_kfold_benchmark_n{i}_s59.json' for i in range(1, 6)
])

print('EBSE Detection Relabeled')
analyze([
    f'ebse_detection_kfold_benchmark_n{i}_s59_relabeled.json' for i in range(1, 6)
])

print('EBSE Classification Relabeled')
analyze([
    f'ebse_classification_kfold_benchmark_n{i}_s59_relabeled.json' for i in range(1, 6)
])

print('Combined Classification EBSE')
analyze([
    f'ebse_detect_and_classify_n{i}.json' for i in range(1, 2)
])

print('Combined Classification BHAT')
analyze([
    f'bhat_detect_and_classify_n{i}.json' for i in range(1, 2)
])

