import tensorflow as tf

from .model import AbstractModel, HyperParameter, InputEncoding, _fix_hyper_params


class LinearRNNModel(AbstractModel):

    def get_model(self, *,
                  embedding=None,
                  embedding_size: int | None = None,
                  embedding_output_size: int | None = None,
                  **kwargs) -> tf.keras.Model:
        inputs, next_layer = self.get_input_layer(
            embedding=embedding,
            embedding_size=embedding_size,
            embedding_output_size=embedding_output_size,
            trainable_embedding=(kwargs.get('use-trainable-embedding', 'False') == 'True')
        )
        bilayer_size = int(kwargs.get('bidirectional-layer-size', 64))
        current = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(bilayer_size)
        )(next_layer)
        n_layers = int(kwargs.get('number-of-hidden-layers', 1))
        for i in range(1, n_layers + 1):
            layer_size = int(kwargs.get(f'hidden-layer-{i}-size', 64))
            current = tf.keras.layers.Dense(layer_size)(current)
        outputs = self.get_output_layer()(current)
        return tf.keras.Model(inputs=[inputs], outputs=outputs)

    @staticmethod
    def supported_input_encodings() -> list[InputEncoding]:
        return [
            InputEncoding.Vector,
            InputEncoding.Embedding,
        ]

    @staticmethod
    def input_must_support_convolution() -> bool:
        return False

    @classmethod
    @_fix_hyper_params
    def get_hyper_parameters(cls) -> dict[str, HyperParameter]:
        return {
            'bidirectional_layer_size': HyperParameter(
                default=64, minimum=1, maximum=128
            ),
            'number_of_hidden_layers': HyperParameter(default=1,
                                                      minimum=0,
                                                      maximum=3),
            'hidden_layer_1_size': HyperParameter(default=64,
                                                  minimum=8,
                                                  maximum=128),
            'hidden_layer_2_size': HyperParameter(default=32,
                                                  minimum=8,
                                                  maximum=128),
            'hidden_layer_3_size': HyperParameter(default=16,
                                                  minimum=8,
                                                  maximum=128),
        } | super().get_hyper_parameters()
