from typing import List
from nltk.tokenize import wordpunct_tokenize
import logging


class BytePairEncoder:
    """
    Byte Pair Encoding (BPE) is a data compression technique that is used to create a vocabulary of subword units.
    The algorithm iteratively merges the most frequent pair of bytes (characters) to create a new byte (character).
    The process is repeated until the vocabulary size reaches the desired number of tokens.

    # Reference
    - Sennrich, R., Haddow, B., & Birch, A. (2016). Neural machine translation of rare words with subword units.

    Parameters
    ----------
    vocab_size : int, default=1000
        The number of tokens in the vocabulary.
    log_level : int, default=logging.WARNING
        The logging level for the logger.

    Attributes
    ----------
    EOW : str
        The end-of-word token.
    SOW : str
        The start-of-word token.
    UNK : str
        The unknown token.
    PAD : str
        The padding token.

    Examples
    --------
    encoder = BytePairEncoder(vocab_size=500, log_level=logging.DEBUG)
    encoder.fit(test_corpus)

    test = 'öğrenmesini tamamlayan tokenizer bu metni tokenlerine ayırıyor.'
    # test = 'this text is written in a different language'
    # test = 'öğrenmesini tamamlayan tokenizer bu metni tokenlerine ayırıyor.😊'
    # test = 'Thissrasp is ~noxt😊 a token.'
    # test = '学習を完了したトークナイザは、このテキストをトークンに分割します。'

    Text:
        öğrenmesini tamamlayan tokenizer bu metni tokenlerine ayırıyor.
    Tokenize result:
        [
            '__sow', 'öğren', 'm', 'es', 'in', 'i', '__eow', '__sow', 'ta', 'ma', 'm', 'l', 'ay', 'an', '__eow',
            '__sow', 't', 'o', 'ken', 'iz', 'er', '__eow', '__sow', 'b', 'u', '__eow', '__sow', 'm', 'et', 'n', 'i',
            '__eow', '__sow', 't', 'o', 'ken', 'lerin', 'e', '__eow', '__sow', 'ay', 'ır', 'ı', 'y', 'or', '__eow',
            '__sow', '.', '__eow'
        ]
    Transform result:
        [
            [
                0, 102, 10, 82, 49, 4, 1, 0, 119, 126, 10, 7, 65, 48, 1, 0, 13, 18, 250, 125, 47, 1, 0, 19, 14, 1, 0,
                10, 55, 8, 4, 1, 0, 13, 18, 250, 96, 5, 1, 0, 65, 67, 12, 11, 103, 1, 0, 21, 1
            ]
        ]
    Inverse transform:
        ['öğrenmesini tamamlayan tokenizer bu metni tokenlerine ayırıyor .']-
    """
    EOW = '__eow'
    SOW = '__sow'
    UNK = '__unk'
    PAD = '__pad'

    def __init__(self, vocab_size=1000, log_level=logging.WARNING):
        self._logger = logging.getLogger('BytePairEncoderLogger')
        self._logger.setLevel(log_level)

        self.merges = {}
        self.inverse_merges = {}
        self.vocab = []
        self.inverse_vocab = {}

        self.token_mapper = {
            BytePairEncoder.SOW: '',
            BytePairEncoder.EOW: ' '
        }

        self.required_tokens = [BytePairEncoder.SOW, BytePairEncoder.EOW, BytePairEncoder.UNK]

        self.vocab_size = vocab_size
        self._logger.debug('Initialized')

    def __set_log_level(self, log_level):
        self._logger.setLevel(log_level)

    def __tokenize_word(self, sentence: str):
        return wordpunct_tokenize(sentence)

    def __initialize_word_frequencies(self, corpus: List[str]):
        vocab = {}
        for sentence in corpus:
            for word in self.__tokenize_word(sentence):
                vocab[word] = vocab.get(word, 0) + 1
        self._logger.debug('Word frequency map initialized!')
        return vocab

    def __initialize_base_vocab(self, word_freqs):
        char_freqs = {}
        for word, frequency in word_freqs.items():
            for char in word:
                char_freqs[char] = char_freqs.get(char, 0) + frequency
        char_freqs = list(map(lambda x: x[0], sorted(char_freqs.items(), key=lambda x: x[1], reverse=True)))

        base_vocab = self.required_tokens + char_freqs
        self._logger.debug('Base vocabulary initialized!')
        return base_vocab

    def __compute_pair_freqs(self, word_freqs, splits):
        pair_freqs = {}
        for word, freq in word_freqs.items():
            split = splits[word]
            if len(split) == 1:
                continue

            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
        return pair_freqs

    def __get_most_frequent_pair(self, word_freqs, splits):
        pair_freqs = self.__compute_pair_freqs(word_freqs, splits)
        most_frequent_pair = max(pair_freqs, key=pair_freqs.get)
        return most_frequent_pair

    def __learn_vocab(self, word_freqs, vocab):
        merges = {}
        splits = {word: [c for c in word] for word in word_freqs.keys()}
        idx = len(vocab)
        while len(vocab) < self.vocab_size:
            if all(len(tokens) <= 1 for tokens in splits.values()):
                self._logger.warning('All words are tokenized. There is no pair to merge. Breaking...')
                break

            most_frequent_pair = self.__get_most_frequent_pair(word_freqs, splits)
            (a, b) = most_frequent_pair
            for word in word_freqs:
                split = splits[word]
                if len(split) == 1:
                    continue

                i = 0
                while i < len(split) - 1:
                    if split[i] == a and split[i + 1] == b:
                        split = split[:i] + [a + b] + split[i + 2:]
                    else:
                        i += 1
                splits[word] = split

            merges[most_frequent_pair] = idx
            vocab.append(a + b)
            idx += 1
        self._logger.debug('BPE vocabulary and merge map created!')
        return vocab, merges

    def fit(self, corpus: List[str]):
        word_freqs = self.__initialize_word_frequencies(corpus)
        vocab = self.__initialize_base_vocab(word_freqs)

        self.vocab, self.merges = self.__learn_vocab(word_freqs, vocab)
        self.inverse_vocab = {token: idx for idx, token in enumerate(self.vocab)}
        self.inverse_merges = {idx: pair for pair, idx in self.merges.items()}

    def tokenize(self, text):
        text = self.__tokenize_word(text)
        splits = [[l for l in word] for word in text]

        for pair, merge in self.merges.items():
            for idx, split in enumerate(splits):
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [pair[0] + pair[1]] + split[i + 2:]
                    else:
                        i += 1
                splits[idx] = split

        for idx, split in enumerate(splits):
            splits[idx] = [BytePairEncoder.SOW] + splits[idx] + [BytePairEncoder.EOW]

        return sum(splits, [])

    def single_transform(self, text):
        tokens = self.tokenize(text)
        encoded = []
        for token in tokens:
            if token in self.vocab:
                encoded.append(self.inverse_vocab[token])
            else:
                self._logger.debug(f'Character \'{token}\' not found in vocabulary, adding UNK token!')
                encoded.append(self.inverse_vocab[BytePairEncoder.UNK])
        return encoded

    def transform(self, list_of_texts):
        return [self.single_transform(text) for text in list_of_texts]

    def single_inverse_transform(self, tokens):
        decoded = ''
        for idx in tokens:
            token = self.vocab[idx]
            decoded += self.token_mapper.get(token, token)
        return decoded.strip()

    def inverse_transform(self, list_of_tokens):
        return [self.single_inverse_transform(tokens) for tokens in list_of_tokens]
