##############################################################################
##############################################################################
# Imports
##############################################################################

import argparse
import functools
import operator
import os
import typing
import warnings

import gensim
import csv
import json
import re

import nltk.tokenize
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import scipy.sparse


csv.field_size_limit(100000000)

LEGACY_NGRAMS = True


##############################################################################
##############################################################################
# Text Cleanup
##############################################################################


def remove_formatting(text: str) -> str:
    text = re.sub(r'\[.*?\|.*?\]', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'org.\S+', '', text)
    text = _remove_code_blocks(text)
    text = _remove_no_format_blocks(text)
    text = re.sub(r'\{\{(?P<code>.*?)\}\}', '', text)
    return text


# Removes code blocks from a text
def _remove_code_blocks(text: str) -> str:
    # Step 1: find all text markers
    starts = list(re.finditer(r'\{code:.*?\}', text))
    generic = list(re.finditer(r'\{code\}', text))
    # Step 2: Filter out all equal objects
    pure_starts = []
    for match in starts:
        for item in generic:
            if match.group() == item.group() and match.start() == item.start():
                break
        else:
            pure_starts.append(match)
    # Step 3: Order all match objects
    markers = [(s, True) for s in pure_starts] + [(s, False) for s in generic]
    markers.sort(key=lambda x: x[0].start())
    # Step 4: Remove code blocks, or resolve ambiguity
    removals = []
    while len(markers) >= 2:
        (start, start_is_pure), (end, end_is_pure), *markers = markers
        if end_is_pure:
            # We have two starting tags; We ignore the second one
            markers.insert(0, (start, start_is_pure))
            continue
        removals.append((start.start(), end.end()))
    if markers:
        marker, is_pure = markers.pop()
        # assume this is an unmatched start; remove the entirety of the remaining string
        removals.append((marker.start(), len(text)))
    # Step 5: Remove parts from the string
    for start, stop in reversed(removals):
        text = f'{text[:start]}{text[stop + 1:]}'
    return text


def _remove_no_format_blocks(text: str) -> str:
    matches = list(re.finditer(r'\{noformat\}', text))
    markers = []
    for i, match in enumerate(matches):
        if i % 2 == 0:
            markers.append(match.start())
        else:
            markers.append(match.end())
    # If the last block is not closed, remove all trailing content
    if len(markers) % 2 == 1:
        markers.append(len(text))
    # Create pairs of markers
    blocks = []
    for start, end in zip(markers[::2], markers[1::2]):
        blocks.append((start, end))
    # Remove code from input string
    for start, stop in reversed(blocks):
        text = f'{text[:start]}{text[stop:]}'
    return text


##############################################################################
##############################################################################
# Pre-processing
##############################################################################


# Apply all preprocessing steps from Bhat to a text
def preprocess(text: str) -> typing.List[str]:
    words = tokenize(text)
    words = transform_lowercase(words)
    words = remove_stopwords(words)
    words = stem_words(words)
    return words


# Tokenize a text to a list of tokens
def tokenize(text: str) -> typing.List[str]:
    return list(gensim.utils.tokenize(text))


# Convert character in a list of tokens to lower case
def transform_lowercase(words: typing.List[str]) -> typing.List[str]:
    return [word.lower() for word in words]


# Remove stopwords from a list of tokens
def remove_stopwords(words: typing.List[str]) -> typing.List[str]:
    #stopwords = nltk.corpus.stopwords.words('english')
    with open('repo_stopwords.txt') as file:
        stopwords = [line.strip() for line in file if line.strip()]
    return [word for word in words if word not in stopwords]


# Stem all words in a token list
def stem_words(words: typing.List[str]) -> typing.List[str]:
    return [nltk.stem.PorterStemmer().stem(word) for word in words]


##############################################################################
##############################################################################
# Feature Vector Generation
##############################################################################


# Creates a list of ngrams from a list of tokens
def create_ngrams(words: typing.List[str], n: int) -> [str]:
    return ngrams(words, n=n)


# Calculate the term frequency inverse document frequency of a list of issues
def tfidf(corpus: list[str] | list[list[str]], n: int, *,
          multi_file=False, all_at_once=True):
    if multi_file and all_at_once:
        corpus = functools.reduce(operator.concat,
                                  corpus,
                                  [])
    elif multi_file:
        return tfidf_train(corpus, n)
    return compute_feature_vectors(corpus, TfidfVectorizer, n)


def tfidf_train(items, n: int):
    train, *test = items
    vector_length, train_vectors, vec, ngram_vec = compute_feature_vectors(train,
                                                                           TfidfVectorizer,
                                                                           n,
                                                                           return_vectorizers=True)
    test = [compute_feature_vectors_trained(x, vec, ngram_vec, n) for x in test]
    return vector_length, [train_vectors] + test


# Calculate the term frequency of a list of issues
def term_frequency(corpus: list[str], n: int):
    return compute_feature_vectors(corpus, CountVectorizer, n)


def compute_feature_vectors(corpus: list[str],
                            vectorizer_factory,
                            n: int, *,
                            return_vectorizers=False):
    def __as_list(f):
        g = f.toarray()
        return [g[0, i] for i in range(g.shape[1])]

    if LEGACY_NGRAMS:
        vectorizer: CountVectorizer = vectorizer_factory()
        left_half = vectorizer.fit_transform(corpus).tocsr()

        ngram_vectorizer = None
        if n != 1:
            ngram_vectorizer: CountVectorizer = vectorizer_factory(
                ngram_range=(n, n)
            )
            right_half = ngram_vectorizer.fit_transform(corpus).tocsr()
            features = scipy.sparse.hstack((left_half, right_half)).tocsr()
        else:
            features = left_half

        return (
            features.shape[1],
            [__as_list(features[i, :]) for i in range(features.shape[0])]
        ) + (
            () if not return_vectorizers else (vectorizer, ngram_vectorizer)
        )

    else:
        vectorizer: CountVectorizer = vectorizer_factory(ngram_range=(1, n))
        features = vectorizer.fit_transform(corpus).tocsr()
        return (
            features.shape[1],
            [__as_list(features[i, :]) for i in range(features.shape[0])]
        ) + (
            () if not return_vectorizers else (vectorizer, None)
        )


def compute_feature_vectors_trained(corpus: list[str],
                            vectorizer: CountVectorizer,
                            ngram_vectorizer: CountVectorizer,
                            n: int):
    def __as_list(f):
        g = f.toarray()
        return [g[0, i] for i in range(g.shape[1])]

    if LEGACY_NGRAMS:
        left_half = vectorizer.transform(corpus).tocsr()

        if n != 1:
            right_half = ngram_vectorizer.transform(corpus).tocsr()
            features = scipy.sparse.hstack((left_half, right_half)).tocsr()
        else:
            features = left_half

        return [__as_list(features[i, :]) for i in range(features.shape[0])]
    else:
        features = vectorizer.transform(corpus).tocsr()
        return [__as_list(features[i, :]) for i in range(features.shape[0])]


##############################################################################
##############################################################################
# Utilities
##############################################################################


# Convert a vector to a sparse dict
def vector_to_sparse_dict(vector: list[float]) -> dict:
    sparse_dict = dict()
    for idx in range(len(vector)):
        if vector[idx] != 0:
            sparse_dict[idx] = float(vector[idx])
    return sparse_dict


##############################################################################
##############################################################################
# Helper Functions
##############################################################################


def get_issues_by_task(filename: str):
    detection_text = []
    classifying_text = []
    detection_labels = []
    classifying_labels = []

    projects = []

    with open(filename, encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=',', quotechar='"')

        # skip header
        next(reader)

        # store issues in two separate lists.
        for uid, summary, description, is_design, design_type in reader:
            projects.append(uid.split('-')[0])
            issue_text = ' '.join(preprocess(summary + ' ' + description))
            detection_text.append(issue_text)
            if is_design == 'True':
                # Issue is architectural
                classifying_text.append(issue_text)
                detection_labels.append("true")
                classifying_labels.append(design_type)
            else:
                # Issue is not architectural
                detection_labels.append("false")

    return (
        projects,
        (detection_text, detection_labels),
        (classifying_text, classifying_labels)
    )


def get_myresources():
    detection_text = []
    classifying_text = []
    detection_labels = []
    classifying_labels = []

    raise NotImplementedError

    with open('myresources_DesignDecisions.csv') as file:
        reader = csv.reader(file, delimiter=',')

        for row in reader:
            detection_text.append(row[1])
            if row[0] == 'Yes':
                detection_labels.append("true")
            else:
                detection_labels.append("false")

    with open('myresources_DecisionCategory.csv') as file:
        reader = csv.reader(file, delimiter=',')

        for row in reader:
            classifying_text.append(row[1])
            classifying_labels.append(row[0])

    return (
        (detection_text, detection_labels),
        (classifying_text, classifying_labels)
    )


def save_feature_vectors(filename, vectors, labels, vector_size, projects):
    data = {
        'vector_size': vector_size,
        'features': vectors,
        'labels': labels,
        'projects': projects
    }

    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file)


##############################################################################
##############################################################################
# Main function
##############################################################################


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-files', type=str,
                        default='issues.csv', nargs='+',
                        help='Specify the path of the input csv file')
    parser.add_argument('-n', type=int, default=3,
                        help='Specify n in the n-grams')
    parser.add_argument('--myresources', type=bool, default=False)
    parser.add_argument('--cleanup-formatting', action='store_true', default=False)
    parser.add_argument('--proper-multi-input', action='store_true', default=False,
                        help='If given, train TF/IDF on first input and apply transform on second')
    args = parser.parse_args()

    if len(args.input_files) > 1:
        warnings.warn('No classification data is generated when >= 2 files are given.')

    args.input_files = [args.input_files]
    if len(args.input_files) <= 1:
        if args.myresources:
            detection_task, classification_task = get_myresources()
            raise NotImplementedError
        else:
            projects, detection_task, classification_task = get_issues_by_task(args.input_files[0])

        projects = [p if p != 'HBASE' else 'HADOOP' for p in projects]

        if args.cleanup_formatting:
            detection_task = [remove_formatting(x) for x in detection_task[0]], detection_task[1]
            classification_task = [remove_formatting(x) for x in classification_task[0]], classification_task[1]

        # Process detection part
        detection_text, detection_labels = detection_task
        vector_size, detection_features = tfidf(detection_text,
                                                args.n)

        detection_features_sparse = [
            vector_to_sparse_dict(feature)
            for feature in detection_features
        ]

        save_feature_vectors('detection_data.json',
                             detection_features_sparse,
                             detection_labels,
                             vector_size,
                             projects)

        # Process classification part
        classification_text, classification_labels = classification_task
        vector_size, classification_features = term_frequency(classification_text,
                                                              args.n)

        classification_features_sparse = [
            vector_to_sparse_dict(feature)
            for feature in classification_features
        ]

        save_feature_vectors('classifying_data.json',
                             classification_features_sparse,
                             classification_labels,
                             vector_size,
                             projects)

        combined_text = []
        combined_labels = []
        classification_index = 0
        for label, detection_text in zip(detection_labels, detection_text):
            if label == 'false':
                combined_labels.append('non-architectural')
                combined_text.append(detection_text)
            else:
                combined_labels.append(classification_labels[classification_index])
                combined_text.append(classification_text[classification_index])
                classification_index += 1
        vector_size, combined_features = term_frequency(combined_text, args.n)

        combined_features_sparse = [
            vector_to_sparse_dict(feature)
            for feature in combined_features
        ]

        save_feature_vectors('detect_and_classify__data.json',
                             combined_features_sparse,
                             combined_labels,
                             vector_size,
                             projects)

    else:   # len(args.input_files) > 1
        all_texts = []
        all_labels = []
        all_projects = []

        for filename in args.input_files:
            projects, (text, labels), _ = get_issues_by_task(filename)
            all_texts.append(text)
            all_labels.append(labels)
            all_projects.append(projects)

        # Now, put in all text and generate feature vectors
        # combined_text = functools.reduce(operator.concat,
        #                                  all_texts,
        #                                  [])

        if args.proper_multi_input:
            vector_size, all_features = tfidf(all_texts, args.n,
                                              multi_file=True,
                                              all_at_once=False)
            all_features = [
                [vector_to_sparse_dict(vector) for vector in feature_set]
                for feature_set in all_features
            ]
        else:
            vector_size, combined_features = tfidf(all_texts, args.n,
                                                   multi_file=True,
                                                   all_at_once=True)

            combined_sparse_features = [
                vector_to_sparse_dict(vector)
                for vector in combined_features
            ]

            # Now, we must pull out all the individual sets of feature vectors
            all_features = []
            start_index = 0
            stop_index = 0
            for label_set in all_labels:
                stop_index += len(label_set)
                all_features.append(combined_sparse_features[start_index:stop_index])
                start_index = stop_index

        # Finally, save all files
        for filename, features, labels, project_labels in zip(args.input_files, all_features, all_labels, all_projects):
            assert len(all_features) == len(all_labels)
            filename_start = os.path.splitext(os.path.split(filename)[1])[0]
            save_feature_vectors(f'detection_data_{filename_start}.json',
                                 features,
                                 labels,
                                 vector_size,
                                 project_labels)


if __name__ == '__main__':
    main()
