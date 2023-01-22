from .generator import AbstractFeatureGenerator, InputEncoding, ParameterSpec


class Metadata(AbstractFeatureGenerator):

    @staticmethod
    def input_encoding_type() -> InputEncoding:
        return InputEncoding.Vector

    def generate_vectors(self,
                         tokenized_issues: list[list[str]],
                         metadata,
                         args: ...):
        return {'features': metadata,
                'feature_shape': len(metadata[0])}

    @staticmethod
    def get_parameters() -> dict[str, ParameterSpec]:
        return {} | super(Metadata, Metadata).get_parameters()
