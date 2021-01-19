# -*- coding: utf-8 -*-


def get_data(data_path):
    total_sample = []
    Label = []
    with open(data_path) as f:
        for line in f:
            item = line.strip().split('\t')
            mirna = int(item[0])
            disease = int(item[1])
            label = int(item[2])
            Label.append(label)
            index_list = [mirna, disease]
            total_sample.append(index_list)
    total_sample.reverse()
    Label.reverse()
    return total_sample, Label


def get_train_data():
    data_path = '../data/train_data.txt'
    total_sample, label = get_data(data_path)
    return total_sample, label

