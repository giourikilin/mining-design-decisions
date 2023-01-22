import tensorflow as tf

from .model import AbstractModel, HyperParameter, InputEncoding, _fix_hyper_params


class LinearConv1Model(AbstractModel):

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
        layer_size = int(kwargs.get('fully-connected-layer-size', 32))
        filters = int(kwargs.get('filters', 32))
        num_convolutions = int(kwargs.get('number-of-convolutions', 1))
        convolution_sizes = [int(kwargs.get(f'kernel-{i}-size', 8))
                             for i in range(1, num_convolutions + 1)]
        height = self.input_size
        pooling_sizes = [height - int(kwargs.get(f'kernel-{i}-size', 8))
                         for i in range(1, num_convolutions + 1)]
        convolutions = [
            tf.keras.layers.Conv1D(filters=filters,
                                   kernel_size=kernel_size)(next_layer)
            for kernel_size in convolution_sizes
        ]
        pooling_layers = [
            tf.keras.layers.MaxPooling1D(pool_size=p_size)(hidden)
            for hidden, p_size in zip(convolutions, pooling_sizes)
        ]
        #hidden = tf.keras.layers.Conv1D(filters=filters,
        #                                kernel_size=kernel_size,
        #                                activation='relu')(next_layer)
        #hidden = tf.keras.layers.MaxPooling1D(pool_size=pooling_size)(hidden)
        concatenated = tf.keras.layers.concatenate(pooling_layers, axis=1)
        hidden = tf.keras.layers.Flatten()(concatenated)
        if layer_size > 0:
            hidden = tf.keras.layers.Dense(layer_size)(hidden)
        outputs = self.get_output_layer()(hidden)
        return tf.keras.Model(inputs=[inputs], outputs=outputs)

    @staticmethod
    def supported_input_encodings() -> list[InputEncoding]:
        return [
            InputEncoding.Vector,
            InputEncoding.Embedding,
        ]

    @staticmethod
    def input_must_support_convolution() -> bool:
        return True

    @classmethod
    @_fix_hyper_params
    def get_hyper_parameters(cls) -> dict[str, HyperParameter]:
        return {
            'fully_connected_layer_size': HyperParameter(
                default=32, minimum=0, maximum=128
            ),
            'number_of_convolutions': HyperParameter(
                default=1, minimum=1, maximum=32
            ),
            'kernel_1_size': HyperParameter(
                default=8, minimum=1, maximum=64
            ),
            'filters': HyperParameter(
                default=32, minimum=1, maximum=64
            ),
            'pooling_size': HyperParameter(
                default=2, minimum=2, maximum=16
            ),
        } | super().get_hyper_parameters()
