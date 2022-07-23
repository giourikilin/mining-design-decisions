from abc import ABC

import tensorflow as tf

from .model import AbstractModel, HyperParameter, InputEncoding


class NonlinearConv2Model(AbstractModel):

    def get_model(self, *,
                  embedding=None,
                  embedding_size: int | None = None,
                  embedding_output_size: int | None = None,
                  **kwargs) -> tf.keras.Model:
        inputs, next_layer = self.get_input_layer(
            embedding=embedding,
            embedding_size=embedding_size,
            embedding_output_size=embedding_output_size,
            trainable_embedding=kwargs.get('use-trainable-embedding', False)
        )
        n_convolutions = int(kwargs.get('number-of-convolutions', 3))
        filters = int(kwargs.get('filters', 32))
        height, width = self.input_size
        convolutions = [
            tf.keras.layers.Conv2D(filters,
                                   (
                                       int(kwargs.get(f'convolution-{i}-size', 1)),
                                       width))(next_layer)
            for i in range(1, n_convolutions + 1)
        ]
        pooling_layers = [
            tf.keras.layers.MaxPooling2D(pool_size=(
                height - int(kwargs.get(f'pooling-{i}-size', 1)),
                1)
            )
            for i in range(1, n_convolutions + 1)
        ]
        pooling = [
            pooling_layer(conv_layer)
            for pooling_layer, conv_layer in zip(pooling_layers, convolutions)
        ]
        concatenated = tf.keras.layers.concatenate(pooling)
        current = tf.keras.layers.Flatten()(concatenated)
        n_layers = int(kwargs.get('number-of-hidden-layers', 1))
        for i in range(1, n_layers + 1):
            layer_size = int(kwargs.get(f'hidden-layer-{i}-size', 64))
            current = tf.keras.layers.Dense(layer_size)(current)
        outputs = self.get_output_layer()(current)
        return tf.keras.Model(inputs=[inputs], outputs=outputs)

    @staticmethod
    def supported_input_encodings() -> list[InputEncoding]:
        return [
            InputEncoding.Matrix,
        ]

    @staticmethod
    def input_must_support_convolution() -> bool:
        return True

    @classmethod
    def get_hyper_parameters(cls) -> dict[str, HyperParameter]:
        return {
            'number_of_convolutions': HyperParameter(
                default=3, minimum=1, maximum=5
            ),
            'filters': HyperParameter(
                default=32, minimum=1, maximum=64
            ),
            'convolution_1_size': HyperParameter(
                default=1, minimum=1, maximum=10
            ),
            'convolution_2_size': HyperParameter(
                default=1, minimum=1, maximum=10
            ),
            'convolution_3_size': HyperParameter(
                default=1, minimum=1, maximum=10
            ),
            'convolution_4_size': HyperParameter(
                default=1, minimum=1, maximum=10
            ),
            'convolution_5_size': HyperParameter(
                default=1, minimum=1, maximum=10
            ),
            'pooling_1_size': HyperParameter(
                default=1, minimum=0, maximum=10
            ),
            'pooling_2_size': HyperParameter(
                default=1, minimum=0, maximum=10
            ),
            'pooling_3_size': HyperParameter(
                default=1, minimum=0, maximum=10
            ),
            'pooling_4_size': HyperParameter(
                default=1, minimum=0, maximum=10
            ),
            'pooling_5_size': HyperParameter(
                default=1, minimum=0, maximum=10
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
