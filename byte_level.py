from typing import List
from nltk.tokenize import wordpunct_tokenize
import logging


class ByteLevelBytePairEncoding:
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

    # test = 'öğrenmesini tamamlayan tokenizer bu metni tokenlerine ayırıyor.'
    # test = 'this text is written in a different language'
    # test = 'öğrenmesini tamamlayan tokenizer bu metni tokenlerine ayırıyor.😊'
    # test = 'Thissrasp is ~noxt😊 a token.'
    test = '学習を完了したトークナイザは、このテキストをトークンに分割します。'

    Text:
        学習を完了したトークナイザは、このテキストをトークンに分割します。
    Tokenize result:
        [
            257, 229, 173, 166, 231, 191, 146, 227, 130, 146, 229, 174, 140, 228, 186, 134, 227, 129, 151, 227, 129,
            159, 227, 131, 136, 227, 131, 188, 227, 130, 175, 227, 131, 138, 227, 130, 164, 227, 130, 182, 227, 129,
            175, 256, 257, 227, 128, 129, 256, 257, 227, 129, 147, 227, 129, 174, 227, 131, 134, 227, 130, 173, 227,
            130, 185, 227, 131, 136, 227, 130, 146, 227, 131, 136, 227, 131, 188, 227, 130, 175, 227, 131, 179, 227,
            129, 171, 229, 136, 134, 229, 137, 178, 227, 129, 151, 227, 129, 190, 227, 129, 153, 256, 257, 227, 128,
            130, 256
        ]
    Transform result:
        [
            [
                257, 229, 173, 166, 231, 191, 146, 227, 130, 146, 229, 174, 140, 228, 186, 134, 227, 129, 151, 227, 129,
                159, 227, 131, 136, 227, 131, 188, 227, 130, 175, 227, 131, 138, 227, 130, 164, 227, 130, 182, 227, 129,
                175, 256, 257, 227, 128, 129, 256, 257, 227, 129, 147, 227, 129, 174, 227, 131, 134, 227, 130, 173, 227,
                130, 185, 227, 131, 136, 227, 130, 146, 227, 131, 136, 227, 131, 188, 227, 130, 175, 227, 131, 179, 227,
                129, 171, 229, 136, 134, 229, 137, 178, 227, 129, 151, 227, 129, 190, 227, 129, 153, 256, 257, 227, 128,
                130, 256
            ]
        ]
    Inverse transform:
        ['学習を完了したトークナイザは 、 このテキストをトークンに分割します 。']
    """

    def __init__(self, vocab_size=1000, log_level=logging.WARNING):
        self._logger = logging.getLogger('BytePairEncoderLogger')
        self._logger.setLevel(log_level)

        self.merges = {}
        self.inverse_merges = {}
        self.vocab = []
        self.inverse_vocab = {}

        self.EOW = 256
        self.SOW = 257
        self.UNK = 258
        self.PAD = 259

        self.token_mapper = {
            self.EOW: 32  # White space
        }

        self.required_tokens = [self.SOW, self.EOW, self.UNK]

        self.vocab_size = vocab_size
        self._logger.debug('Initialized')

    def __set_log_level(self, log_level):
        self._logger.setLevel(log_level)

    def __tokenize_word(self, sentence: str):
        return [''.join(word).encode('utf-8') for word in wordpunct_tokenize(sentence)]

    def __initialize_word_frequencies(self, corpus: List[str]):
        vocab = {}
        for sentence in corpus:
            for word in self.__tokenize_word(sentence):
                vocab[word] = vocab.get(word, 0) + 1
        self._logger.debug('Word frequency map initialized!')
        return vocab

    def __initialize_base_vocab(self, word_freqs):
        byte_freqs = {}
        for word, frequency in word_freqs.items():
            for char in word:
                byte_freqs[char] = byte_freqs.get(char, 0) + frequency
        byte_freqs = list(map(lambda x: x[0], sorted(byte_freqs.items(), key=lambda x: x[1], reverse=True)))
        base_vocab = list(dict.fromkeys(self.required_tokens[:] + byte_freqs + list(range(256))))

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
        pair_freqs =  self.__compute_pair_freqs(word_freqs, splits)
        most_frequent_pair = max(pair_freqs, key=pair_freqs.get)
        return most_frequent_pair

    def __learn_vocab(self, word_freqs, vocab):
        merges = {}
        splits = {word: [c for c in word] for word in word_freqs.keys()}
        new_token_count = 0
        start_idx = len(vocab)
        while len(vocab) < self.vocab_size:
            if all(len(tokens) <= 1 for tokens in splits.values()):
                self._logger.warning('All words are tokenized. There is no pair to merge. Breaking...')
                break

            most_frequent_pair = self.__get_most_frequent_pair(word_freqs, splits)
            (a, b) = most_frequent_pair
            new_token_count += 1
            for word in word_freqs:
                split = splits[word]
                if len(split) == 1:
                    continue

                i = 0
                while i < len(split) - 1:
                    if split[i] == a and split[i + 1] == b:
                        split = split[:i] + [start_idx + new_token_count] + split[i + 2:]
                    else:
                        i += 1
                splits[word] = split

            merges[most_frequent_pair] = start_idx + new_token_count
            vocab.append(start_idx + new_token_count)
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
                        split = split[:i] + [merge] + split[i + 2:]
                    else:
                        i += 1
                splits[idx] = split

        for idx, split in enumerate(splits):
            splits[idx] = [self.SOW] + splits[idx] + [self.EOW]
        return sum(splits, [])

    def transform(self, text_list):
        # for text in text_list:
        #     yield self.tokenize(text)
        return [self.tokenize(text) for text in text_list]

    def _inverse_transform_single(self, tokens):
        idx = 0
        decoded = []
        while idx < len(tokens) -1:
            token = tokens[idx]
            if token in self.inverse_merges:
                merges = self.inverse_merges[token]
                tokens = tokens[:idx] + [merges[0], merges[1]] + tokens[idx + 1:]
            else:
                idx += 1
                if token in [self.SOW, self.UNK]:
                    continue
                token = self.token_mapper.get(token, token)
                decoded.append(token)
        return bytes(decoded).decode('utf-8')

    def inverse_transform(self, token_lists):
        # for tokens in token_lists:
        #     yield self._inverse_transform_single(tokens)
        return [self._inverse_transform_single(tokens) for tokens in token_lists]
