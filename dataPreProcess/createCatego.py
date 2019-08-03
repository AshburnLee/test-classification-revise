# -*- coding: utf-8 -*-


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
    category_file = '../cnews_data/cnews.category.txt'
    category_dict = CategoryDict(category_file)

    test_label = '科技'
    print('科技的id： %s' % category_dict.category_to_id(test_label))
