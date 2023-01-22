from ..classifiers import InputEncoding
from .word2vec import AbstractFeatureGenerator, ParameterSpec


class Bert(AbstractFeatureGenerator):
    @staticmethod
    def input_encoding_type() -> InputEncoding:
        return InputEncoding.Text

    def generate_vectors(self,
                         tokenized_issues: list[list[str]],
                         metadata,
                         args: ...):
        features = []
        for tokenized_issue in tokenized_issues:
            features.append(tokenized_issue[0])

        return {
            'features': features,
            'feature_shape': None
        }

    @staticmethod
    def get_parameters() -> dict[str, ParameterSpec]:
        return {

        }
