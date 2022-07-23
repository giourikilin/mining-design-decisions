import collections
import math

from . import ParameterSpec
from .generator import AbstractFeatureGenerator
from ..classifiers import InputEncoding


class TfidfGenerator(AbstractFeatureGenerator):

    @staticmethod
    def input_encoding_type() -> InputEncoding:
        return InputEncoding.Vector

    def generate_vectors(self, tokenized_issues: list[list[str]], metadata, args: ...):
        document_frequency = collections.defaultdict(int)
        for document in tokenized_issues:
            for word in set(document):
                document_frequency[word] += 1
        inverse_document_frequency = collections.defaultdict(float)
        inverse_document_frequency.update({
            term: math.log10(len(tokenized_issues) / count)
            for term, count in document_frequency.items()
        })
        # Layout (in words) of the resulting feature vectors
        layout = sorted(document_frequency)
        feature_vectors = []
        for document in tokenized_issues:
            term_counts = collections.defaultdict(int)
            for word in document:
                term_counts[word] += 1
            term_frequency = collections.defaultdict(float)
            for word, count in term_counts.items():
                term_frequency[word] = count / len(document)
            vector = [
                term_frequency[term] * inverse_document_frequency[term]
                for term in layout
            ]
            feature_vectors.append(vector)
        assert len(set(len(x) for x in feature_vectors)) == 1
        print(len(feature_vectors))
        print(set(len(x) for x in feature_vectors))
        return {
            'features': feature_vectors,
            'feature_shape': len(layout)
        }

    @staticmethod
    def get_parameters() -> dict[str, ParameterSpec]:
        return super().get_parameters()
