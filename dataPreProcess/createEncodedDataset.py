import tensorflow as tf
import numpy as np
from dataPreProcess import encodeWords


class EncodedDataset:
    """ encode all content and labels for all three files:
        cnews.train.seg.txt
        cnews.val.seg.txt
        cnews.test.seg.txt
    """
    def __init__(self, filename, vocab_dict, catego_dict, encoded_length):
        """
        :param filename: any file from above
        :param vocab_dict: instance of class VocabDict
        :param catego_dict: instance of class CategoryDict
        :param encoded_length: the length of every encoded sentences
        """
        self._vocab_dict = vocab_dict
        self._catego_dict = catego_dict
        self._encoded_length = encoded_length
        self._input = []
        self._output = []
        self._indicator = 0  # the position of next_batch
        self._parse_file(filename)  # create _input & _output

    def _parse_file(self, filename):
        """
        convert all words to id and form the corresponding label & comments
        :param filename: train.seg.txt / val.seg.txt / test.seg.txt
        :return: none
        """
        tf.logging.info('Loading data from %s', filename)
        with open(filename, 'r') as f:
            lines = f.readlines()
        for line in lines:    # for each line,
            label, content = line.strip('\r\n').split('\t')
            # convert label and content to sequence of ids
            id_label = self._catego_dict.category_to_id(label)
            id_words = self._vocab_dict.sentence_to_id(content)
            id_words = id_words[0: self._encoded_length]           # cut
            padding_length = self._encoded_length - len(id_words)  # pad
            id_words = id_words + [self._vocab_dict.unk for _ in range(padding_length)]

            self._input.append(id_words)
            self._output.append(id_label)
        # convert to numpy array
        self._input = np.asarray(self._input, dtype=np.int32)
        self._output = np.asarray(self._output, dtype=np.int32)

        # shuffle category and content
        self._random_shuffle()

    def _random_shuffle(self):
        """
        random shuffle input and output
        """
        p = np.random.permutation(len(self._input))
        self._input = self._input[p]
        self._output = self._output[p]

    def num_samples(self):
        """ how many samples in the _input """
        return len(self._input)

    def next_batch(self, batch_size):
        """
        get next batch data
        :param batch_size:
        :return: the next batch of input and output
        """
        end_indicator = self._indicator + batch_size
        if end_indicator > len(self._input):
            self._random_shuffle()
            self._indicator = 0
            end_indicator = batch_size
        if end_indicator > len(self._input):
            raise Exception("batch size : %d is too large" % batch_size)

        batch_input = self._input[self._indicator: end_indicator]
        batch_ouput = self._output[self._indicator: end_indicator]
        self._indicator = end_indicator
        # return what we require
        return batch_input, batch_ouput


if __name__ == '__main__':

    # files required
    seg_train_file = '../cnews_data/cnews.train.seg.txt'
    seg_val_file = '../cnews_data/cnews.val.seg.txt'
    seg_test_file = '../cnews_data/cnews.test.seg.txt'

    vocab_file = '../cnews_data/cnews.vocab.txt'
    category_file = '../cnews_data/cnews.category.txt'

    # encoded_length & num_word_threshold
    encoded_length = 50
    num_word_threshold = 10

    # create two instance for VocabDict & CategoryDict
    vocab_instance = encodeWords.VocabDict(vocab_file, num_word_threshold)
    catego_instance = encodeWords.CategoryDict(category_file)

    train_dataset = EncodedDataset(
        seg_train_file, vocab_instance, catego_instance, encoded_length)
    val_dataset = EncodedDataset(
        seg_val_file, vocab_instance, catego_instance, encoded_length)
    test_dataset = EncodedDataset(
        seg_test_file, vocab_instance, catego_instance, encoded_length)

    print(train_dataset.next_batch(2), train_dataset.num_samples())
    print(val_dataset.next_batch(2), val_dataset.num_samples())
    print(test_dataset.next_batch(2), test_dataset.num_samples())
