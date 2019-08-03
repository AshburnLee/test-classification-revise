
class VocabDict:
    def __init__(self, filename, num_word_threshold):
        self._word_to_id = {}  # 读文件，将每个词语的id放在此
        self._unk = -1     # 给_unk赋值为-1
        self._num_word_threshold = num_word_threshold   # 频数小于这个数的忽略
        self._read_dict(filename)    # 读filename构成_word_to_dict

    # 实现，读filename构成_word_to_dict
    def _read_dict(self, filename):
        """ read filename and generate {word: id} dict """
        with open(filename, 'r') as f:
            lines = f.readlines()
        for line in lines:    # 读词与词频
            word, frequency = line.strip('\r\n').split('\t')
            frequency = int(frequency)
            if frequency < self._num_word_threshold:
                continue
            idx = len(self._word_to_id)  # idx随_word_to_id大小递增
            if word == '<UNK>':   # 特殊处理UNK
                self._unk = idx
            self._word_to_id[word] = idx  # 构建处字典：{词： idx}

    def show_word_to_list(self):
        """ show what has been generated """
        print(self._word_to_id)

    def word_to_id(self, word):
        """当词没有idx时，返回unk的idx"""
        return self._word_to_id.get(word, self._unk)

    @property
    def unk(self):
        return self._unk

    def size(self):
        return len(self._word_to_id)

    def sentence_to_id(self, sentence):
        """ 用词的idx编码每个句子 """
        # 切分句子后的每个词，找它的idx，
        word_ids = [self.word_to_id(cur_word) for cur_word in sentence.split()]
        return word_ids  # 返回词idx的列表，[你好 呀 ， 参加 比赛 了 吗]->[9901, 5667, 1, 381, 124, 6, 445]


class CategoryDict:
    def __init__(self, filename):
        self._catgo_to_id = {}
        with open(filename, 'r') as f:
            lines = f.readlines()
        for line in lines:
            category = line.strip('\r\n')
            idx = len(self._catgo_to_id)
            self._catgo_to_id[category] = idx

    def category_to_id(self, category_name):
        if category_name not in self._catgo_to_id:
            raise Exception("%s is not in our category list " % category_name)

        return self._catgo_to_id[category_name]


if __name__ == '__main__':

    # test for VocabDict
    vocab_file = '../cnews_data/cnews.vocab.txt'
    Vocab = VocabDict(vocab_file, 20)
    print('总词数： %d' % Vocab.size())
    test_word = '你好 呀 ， 参加 比赛 了 吗'
    print(Vocab.sentence_to_id(test_word))

    # test for CategoryDict
    category_file = '../cnews_data/cnews.category.txt'
    category_dict = CategoryDict(category_file)
    test_label = '科技'
    print('科技的id： %s' % category_dict.category_to_id(test_label))


