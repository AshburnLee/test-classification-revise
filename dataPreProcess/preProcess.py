
import jieba  # pip install jieba to anaconda !

# input files
train_file = '../cnews_data/cnews.train.txt'
val_file = '../cnews_data/cnews.val.txt'
test_file = '../cnews_data/cnews.test.txt'

# output files
seg_train_file = '../cnews_data/cnews.train.seg.txt'
seg_val_file = '../cnews_data/cnews.val.seg.txt'
seg_test_file = '../cnews_data/cnews.test.seg.txt'

vocab_file = '../cnews_data/cnews.vocab.txt'
category_file = '../cnews_data/cnews.category.txt'


def generate_seg_file(input_file, output_seg_file):
    """对input_file内容分词"""
    with open(input_file, 'r') as f:  # 读文件
        lines = f.readlines()
    with open(output_seg_file, 'w') as f:  # 写的方式打开写文件
        for line in lines:  # 对于每一行
            label, content = line.strip('\r\n').split('\t')  # 分出lenbel和content
            word_iter = jieba.cut(content)  # 对content切分成词
            word_content = ''  # 保存每个次
            for word in word_iter:  # 对切分结果中每个词作如下操作：
                word = word.strip(' ')
                if word != '':
                    word_content += word + ' '
            out_line = '%s\t%s\n' % (label, word_content.strip(' '))  # 输入文件每一行的格式

            f.write(out_line)  # 写入文件


# 对三个数据集作同样操作
generate_seg_file(train_file, seg_train_file)
generate_seg_file(val_file, seg_val_file)
generate_seg_file(test_file, seg_test_file)


def generate_vocab_file(input_seg_file, output_vocab_file):
    """将input_seg_file中做词频统计，输出到out_put_file中"""
    with open(input_seg_file, 'r') as f:
        lines = f.readlines()
    word_dict = {}
    for line in lines:
        label, content = line.strip('\r\n').split('\t')
        for word in content.split():
            word_dict.setdefault(word, 0)
            word_dict[word] += 1
    # [(word, frequency), ..., ()]
    sorted_word_dict = sorted(
        word_dict.items(), key=lambda d:d[1], reverse=True)
    with open(output_vocab_file, 'w') as f:
        f.write('<UNK>\t10000000\n')      # 当在测试集中找不到一个词时，用<UNK> 代替
        for item in sorted_word_dict:
            f.write('%s\t%d\n' % (item[0], item[1]))


# 只对已知的训练集执行此操作
generate_vocab_file(seg_train_file, vocab_file)


def generate_category_dict(input_file, catego_file):
    """统计类别"""
    with open(input_file, 'r') as f:
        lines = f.readlines()
    category_dict = {}
    for line in lines:
        label, content = line.strip('\r\n').split('\t')
        category_dict.setdefault(label, 0)
        category_dict[label] += 1
    category_number = len(category_dict)
    with open(catego_file, 'w') as f:
        for category in category_dict:
            line = '%s\n' % category
            print('%s\t%d' % (category, category_dict[category]))
            f.write(line)


generate_category_dict(train_file, category_file)
