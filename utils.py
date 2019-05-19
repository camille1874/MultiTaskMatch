# -*- encoding:utf-8 -*-
import re
from bert_serving.client import BertClient

def get_encoding(lists):
    bc = BertClient()
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
    # dis1 = [[0] * len2] * len1 乘法会导致引用
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

#print(edit_dis("which position does sb play in baseball games", "position/play/baseball")) 0.54
