from ..classifiers import InputEncoding
from .word2vec import AbstractWord2Vec


class Word2Vec1D(AbstractWord2Vec):

    @staticmethod
    def input_encoding_type() -> InputEncoding:
        return InputEncoding.Embedding

    @staticmethod
    def finalize_vectors(tokenized_issues, wv, args):
        idx = 0
        word_to_idx = dict()
        embedding_weights = []
        features = []
        original_text = []
        for tokenized_issue in tokenized_issues:
            feature = []
            current_issue_original_text = []
            for token in tokenized_issue:
                if token in word_to_idx:
                    current_issue_original_text.append(token)
                    feature.append([word_to_idx[token]])
                else:
                    if token in wv:
                        current_issue_original_text.append(token)
                        word_to_idx[token] = idx
                        embedding_weights.append(wv[token].tolist())
                        feature.append([idx])
                        idx += 1
            original_text.append(current_issue_original_text)
            feature.extend([[0]] * (int(args['max-len']) - len(feature)))
            features.append(feature)

        return {'features': features,
                'weights': embedding_weights,
                'feature_shape': int(args['max-len']),
                'vocab_size': idx,
                'word_vector_length': int(args['vector-length']),
                'original': original_text
                }
