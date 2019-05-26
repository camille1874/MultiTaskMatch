# -*- encoding:utf-8 -*-
import re
from bert_serving.client import BertClient
import numpy as np

def get_encoding(lists):
    bc = BertClient(ip="219.223.189.238")
    return bc.encode(lists)


def clean_str(string):
    string = re.sub(r"[(\-lrb\-)(\-rrb\-)]", "", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower() 


def build_path(prefix, data_type, model_type, num_layers, postpix=""):
    return prefix + data_type + "-" + model_type + "-" + str(num_layers) + postpix


def edit_dis(s1, s2):
    len1 = len(s1) + 1
    len2 = len(s2) + 1
    dis = [[0] * len2 for i in range(len1)]
    for i in range(len1):
        dis[i][0] = i
    for i in range(len2):
        dis[0][i] = i
    for i in range(1, len1):
        for j in range(1, len2):
            if s1[i - 1] == s2[j - 1]:
                dis[i][j] = dis[i - 1][j - 1]
            else:
                dis[i][j] = min(dis[i - 1][j - 1], min(dis[i - 1][j], dis[i][j - 1])) + 1
    return dis[-1][-1] / max(len1, len2)


def cos_dis(s1, s2):
    return s1.dot(s2)/(np.linalg.norm(s1)*np.linalg.norm(s2))


def add_trans_bert(s1s, s2s, threthold=0.8):
    bc = BertClient()
    l1 = bc.encode(s1s)
    l2 = bc.encode(s2s)
    diss = []
    for s1 in l1:
        for s2 in l2:
            diss.append(cos_dis(s1, s2))
    aver_dis = sum(diss) / len(diss)
    return aver_dis >= threthold


def add_trans_edit(s1s, s2s, threshold=0.8):
    diss = []
    for s1 in s1s:
        for s2 in s2s:
            diss.append(edit_dis(s1, s2))
    aver_dis = sum(diss) / len(diss)
    return aver_dis >= threshold
