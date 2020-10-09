import math
from collections import defaultdict
from typing import List

import numpy as np
import pickle as pkl


class Model:
    def __init__(self, train_datas: List[str], category_end: int):
        self.train_datas = np.array(train_datas)
        self.category_data = defaultdict(list)
        self.category_end = category_end

    def pre_process(self):
        """
        label을 태깅하고 각각의 categorical data를 one-hot-encoding하는 과정
        :return: None
        """
        label = []
        for data in self.train_datas:
            spliced_data = data[:-1].split("\t")
            if spliced_data[1] == "":
                label.append(0)
            else:
                label.append(1)
            for i, feature in enumerate(spliced_data[2:self.category_end]):  # 0 -> row num, 1 -> timestamp, if you train Dataset1, you should use 2:6
                if feature == "":
                    continue
                else:
                    if feature in self.category_data[i]:
                        continue
                    self.category_data[i].append(feature)
        self.train_label = np.array(label)

        one_hot_encodings = []
        for i, data in enumerate(self.train_datas):
            if i % 1000000 == 0:
                print(f"doin good {i}")
            encoding = np.array([], int)
            spliced_data = data[:-1].split("\t")

            for i, feature in enumerate(spliced_data[2:self.category_end]): # 0 -> row num, 1 -> timestamp, if you train Dataset1, you should use 2:6
                encoded_feature = np.zeros(len(self.category_data[i]))
                if feature == "":
                    encoding = np.append(encoding, encoded_feature) # [...] + [only zeros]
                    continue
                encoded_feature[self.category_data[i].index(feature)] = 1
                encoding = np.append(encoding, encoded_feature)
            one_hot_encodings.append(encoding)
        self.one_hot_encoding = np.array(one_hot_encodings)

    def pre_process_test(self, test_datas):
        """
        test data들을 pre processing 하는 과
        :param test_datas: test_data
        :return: None
        """
        test_labels = []
        one_hot_encodings = []
        for data in test_datas:
            spliced_data = data[:-1].split("\t")
            encoding = np.array([], int)

            if spliced_data[1] == "":
                test_labels.append(0)
            else:
                test_labels.append(1)

            for i, feature in enumerate(spliced_data[2:self.category_end]): # 0 -> row num, 1 -> timestamp, if you train Dataset1, you should use 2:6
                encoded_feature = np.zeros(len(self.category_data[i]))
                if feature == "" or feature not in self.category_data[i]:
                    encoding = np.append(encoding, encoded_feature)
                    continue
                encoded_feature[self.category_data[i].index(feature)] = 1
                encoding = np.append(encoding, encoded_feature)
            one_hot_encodings.append(encoding)
        self.test_label = np.array(test_labels)
        self.test_one_hot_encoding = np.array(one_hot_encodings)

    def phi(self, x: np.ndarray):
        """
        phi function of each model
        :param x:
        :return: None
        """
        pass

    def sub_gradient(self, x, y, lambd):
        """
        calculating sub_gradient for adagrad
        :param x:
        :param y:
        :param lambd: lambda
        :return: loss function의 w에 대한 편미분
        """
        pass

    def gradient(self, h, sub_gradient: np.ndarray):
        """
        update h
        :param h:
        :param sub_gradient:
        :return: h
        """
        return h + np.dot(sub_gradient, sub_gradient)

    # Adagrad
    def train(self, epoch: int, lambd, delta, h):
        """
        training algorithm = adagrad
        :param epoch:
        :param lambd:
        :param delta:
        :param h:
        :return:
        """
        for e in range(epoch):
            print(f"epoch {e+1} start")
            for i, data_y in enumerate(zip(self.one_hot_encoding, self.train_label)):
                data, y = data_y
                if i % 1000000 == 0:
                    print(f"I did training no.{i}")
                sub_gradient = self.sub_gradient(x=data, y=y, lambd=lambd)
                h = self.gradient(h, sub_gradient)
                self.weight = self.weight - delta * sub_gradient / np.math.sqrt(h)

    def probability(self, x: np.ndarray):
        return 1 / (1+np.exp(-self.phi(x)))

    def train_evaluate(self):
        def exp(value):
            return 1 / (1 + np.exp(-value))
        return np.array(list(map(exp, np.dot(self.one_hot_encoding, self.weight))))

    def test_evaluate(self):
        def exp(value):
            return 1 / (1 + np.exp(-value))
        return np.array(list(map(exp, self.phi(self.test_one_hot_encoding))))

    def logloss(self, y, x):
        loss = -  y * math.log(1/ (1 + math.exp(-self.phi(x)))) - (1-y) * math.log(1- 1/ (1 + math.exp(-self.phi(x))))
        return loss

    def test_logloss(self):
        """
        to avoid log(0) I add 1e-60 or multiply (1+1e-60)
        :return: loss
        """
        logloss_arr = -np.multiply(self.test_label, np.log(self.test_evaluate())) - np.multiply((1 - self.test_label), np.ma.log((1 - self.test_evaluate())).filled(0))
        return np.average(logloss_arr)

    def train_logloss(self):
        logloss_arr = -np.multiply(self.train_label, np.log(self.train_evaluate())) - np.multiply((1 - self.train_label), np.ma.log((1 - self.train_evaluate())).filled(0))
        return np.average(logloss_arr)

    def save(self, pkl_name):
        with open(pkl_name, 'wb') as f:
            pkl.dump(self, f)