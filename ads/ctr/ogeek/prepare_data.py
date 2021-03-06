# coding=utf8

import os
import pickle


def get_avg_ctr(filename):
    file = open(filename, 'r', encoding='utf8')
    ctr_dict = {}
    avg_ctr_list = []
    while True:
        line = file.readline()
        if line is None or len(line) == 0:
            break
        line = line.strip()
        left_idx, right_idx = line.index('{'), line.index('}')
        search = line[:left_idx].strip()
        search_prediction = line[left_idx: right_idx + 1]
        rest_info = line[right_idx + 1:].strip().split()
        label = int(rest_info[-1])
        category = rest_info[-2]
        title = rest_info[:-2]
        key = search + search_prediction + str(title) + category
        if key in ctr_dict:
            ctr_dict[key][label] += 1
        else:
            ctr_dict[key] = [0, 0]
            ctr_dict[key][label] += 1
    file.close()
    file = open(filename, 'r', encoding='utf8')
    while True:
        line = file.readline()
        if line is None or len(line) == 0:
            break
        line = line.strip()
        left_idx, right_idx = line.index('{'), line.index('}')
        search = line[:left_idx].strip()
        search_prediction = line[left_idx: right_idx + 1]
        rest_info = line[right_idx + 1:].strip().split()
        category = rest_info[-2]
        title = rest_info[: -2]
        key = search + search_prediction + str(title) + category
        avg_ctr = ctr_dict[key][1] / (ctr_dict[key][0] + ctr_dict[key][1])
        record = {'search': search, 'tile': title, 'category': category, 'ctr': avg_ctr}
        predicts = {}
        # print(line)
        for predict_prop in search_prediction[1: -1].split('",'):
            predict_prop = predict_prop.strip()
            if len(predict_prop) == 0:
                break
            # print('--' + predict_prop)
            predict, prop = predict_prop.split('": ')
            predict = predict[1:]
            prop = prop.strip()
            if prop[-1] == '"':
                prop = float(prop[1: -1])
            else:
                prop = float(prop[1:])
            predicts[predict] = prop
        record['search_predict'] = predicts
        avg_ctr_list.append(record)
    pickle.dump(avg_ctr_list, open('data/train_ctr.pickle', 'wb'))


def test_ctr_data(filename):
    train_ctr = pickle.load(open('data/train_ctr.pickle', 'rb'))
    for train_ctr_item in train_ctr:
        print(train_ctr_item)


def one_record_some_search(data_filename, result_filename):
    """
    :param data_filename: 使用的源数据文件
    :param result_filename: 目标文件，存放一次记录进行了多次搜索的记录
    :return:
    """
    # 将一次记录中进行了多次搜索的记录存放到文件中
    data_file = open(data_filename, 'r', encoding='utf8')
    result_file = open(result_filename, 'w', encoding='utf8')
    while True:
        line = data_file.readline()
        if line is None or len(line) == 0:
            break
        line = line.strip()
        left_idx, right_idx = line.index('{'), line.index('}')
        rest_info = line[right_idx + 1:].strip().split()
        if len(rest_info) > 3:
            result_file.writelines(line)
            result_file.write('\n')

    data_file.close()
    result_file.close()


if __name__ == '__main__':
    get_avg_ctr('data/oppo_round1_train_20180929.txt')

    # test_ctr_data('data/train_ctr.pickle')

    # 将一次记录中进行了多次搜索的记录存放到文件中
    # one_record_some_search('data/oppo_round1_train_20180929.txt', 'data/one_record_some_search.txt')



