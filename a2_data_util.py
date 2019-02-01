# -*- coding: utf-8 -*-
import codecs
import numpy as np
import os
import pickle
import gensim

PAD_ID = 0
from tflearn.data_utils import pad_sequences
_GO="_GO"
_END="_END"
_PAD="_PAD"
def create_vocabulary(simple=None,word2vec_model_path='../../GoogleNews-vectors-negative300.bin', name_scope=''): 
    cache_path ='../cache_vocabulary_label_pik/'+ name_scope + "_word_voabulary.pik"
    print("cache_path:",cache_path,"file_exists:",os.path.exists(cache_path))
    if os.path.exists(cache_path):#如果缓存文件存在，则直接读取
        with codecs.open(cache_path, 'rb') as data_f:
            vocabulary_word2index, vocabulary_index2word=pickle.load(data_f)
            return vocabulary_word2index, vocabulary_index2word
    else:
        vocabulary_word2index={}
        vocabulary_index2word={}
        if simple is not None:
            word2vec_model_path='../../GoogleNews-vectors-negative300.bin'
        print("create vocabulary. word2vec_model_path:",word2vec_model_path)
        model=gensim.models.KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)
        vocabulary_word2index['PAD_ID']=0
        vocabulary_index2word[0]='PAD_ID'
        special_index=0
        if 'biLstmTextRelation' in name_scope:
            vocabulary_word2index['EOS']=1 # a special token for biLstTextRelation model. which is used between two sentences.
            vocabulary_index2word[1]='EOS'
            special_index=1
        for i,vocab in enumerate(model.vocab):
            #vocabulary_word2index[vocab]=i+1+special_index
            #vocabulary_index2word[i+1+special_index]=vocab
            vocabulary_word2index[vocab]=model.word_vec(word)
           

        #save to file system if vocabulary of words is not exists.
        if not os.path.exists(cache_path): #如果不存在写到缓存文件中
            with codecs.open(cache_path, 'wb') as data_f:
                #pickle.dump((vocabulary_word2index,vocabulary_index2word), data_f)
                pickle.dump((vocabulary_word2index,vocabulary_index2word), data_f)
    #return vocabulary_word2index,vocabulary_index2word
    return vocabulary_word2index


def sort_by_value(d):
    items=d.items()
    backitems=[[v[1],v[0]] for v in items]
    backitems.sort(reverse=True)
    return [ backitems[i][1] for i in range(0,len(backitems))]


def load_data_multilabel_new(vocabulary_word2index,valid_portion=0.05,training_data_path='../wikiqa-train.txt',max_length=50):  # n_words=100000,
    """
    input: a file path
    :return: train, test, valid. where train=(trainX, trainY). where
                trainX: is a list of list.each list representation a sentence.trainY: is a list of label. each label is a number
    """
    print("load_data.started...")
    print("load_data_multilabel_new.training_data_path:",training_data_path)
    raw_data = codecs.open(training_data_path, 'r', 'utf8') #-zhihu4-only-title.txt
    lines = raw_data.readlines()
    # 2.transform X as indices
    # 3.transform  y as scalar
    X = []
    Y = []
    Y_decoder_input=[] #ADD 2017-06-15
    for i, line in enumerate(lines):
        x, y = line.split('__label__') #x='w17314 w5521 w7729 w767 w10147 w111'
        y = int(y.strip())
        x = x.strip()
        if i<1:
            print(i,"x0:",x) #get raw x
        #x_=process_one_sentence_to_get_ui_bi_tri_gram(x)
        x=x.split(" ")[:max_length]
        x = [vocabulary_word2index.get(e,0) for e in x] #if can't find the word, set the index as '0'.(equal to PAD_ID = 0)
        if i<2:
            print(i,"x1:",x) #word to index
            #if multi_label_flag: # 2)prepare multi-label format for classification
            #    ys = y.replace('\n', '').split(" ")  # ys is a list
            #    ys_index=[]
            #    for y in ys:
            #        #y_index = vocabulary_word2index_label[y]
            #        ys_index.append(y)
            #    ys_mulithot_list=transform_multilabel_as_multihot(ys_index)
            #else:                #3)prepare single label format for classification
            #    #ys_mulithot_list=vocabulary_word2index_label[y]
        ys_mulithot_list=y
        if i<=3:
            print("ys_index:")
            #print(ys_index)
            print(i,"y:",y," ;ys_mulithot_list:",ys_mulithot_list) #," ;ys_decoder_input:",ys_decoder_input)
        X.append(x)
        Y.append(ys_mulithot_list)
    # 4.split to train,test and valid data
    number_examples = len(X)
    print("number_examples:",number_examples) #
    train = (X[0:int((1 - valid_portion) * number_examples)], Y[0:int((1 - valid_portion) * number_examples)])
    test = (X[int((1 - valid_portion) * number_examples) + 1:], Y[int((1 - valid_portion) * number_examples) + 1:])
    print("load_data.ended...")
    return train, test, test


 # 将一句话转化为(uigram,bigram,trigram)后的字符串
def process_one_sentence_to_get_ui_bi_tri_gram(sentence,n_gram=3):
    """
    :param sentence: string. example:'w17314 w5521 w7729 w767 w10147 w111'
    :param n_gram:
    :return:string. example:'w17314 w17314w5521 w17314w5521w7729 w5521 w5521w7729 w5521w7729w767 w7729 w7729w767 w7729w767w10147 w767 w767w10147 w767w10147w111 w10147 w10147w111 w111'
    """
    result=[]
    word_list=sentence.split(" ") #[sentence[i] for i in range(len(sentence))]
    unigram='';bigram='';trigram='';fourgram=''
    length_sentence=len(word_list)
    for i,word in enumerate(word_list):
        unigram=word                           #ui-gram
        word_i=unigram
        if n_gram>=2 and i+2<=length_sentence: #bi-gram
            bigram="".join(word_list[i:i+2])
            word_i=word_i+' '+bigram
        if n_gram>=3 and i+3<=length_sentence: #tri-gram
            trigram="".join(word_list[i:i+3])
            word_i = word_i + ' ' + trigram
        if n_gram>=4 and i+4<=length_sentence: #four-gram
            fourgram="".join(word_list[i:i+4])
            word_i = word_i + ' ' + fourgram
        if n_gram>=5 and i+5<=length_sentence: #five-gram
            fivegram="".join(word_list[i:i+5])
            word_i = word_i + ' ' + fivegram
        result.append(word_i)
    result=" ".join(result)
    return result

#将LABEL转化为MULTI-HOT
def transform_multilabel_as_multihot(label_list,label_size=2): #1999label_list=[0,1,4,9,5]
    """
    :param label_list: e.g.[0,1,4]
    :param label_size: e.g.199
    :return:e.g.[1,1,0,1,0,0,........]
    """
    result=np.zeros(label_size)
    #set those location as 1, all else place as 0.
    result[label_list] = 1
    return result

#将LABEL转化为MULTI-HOT
def transform_multilabel_as_multihotO(label_list,label_size=2): #1999label_list=[0,1,4,9,5]
    batch_size=len(label_list)
    result=np.zeros((batch_size,label_size))
    #set those location as 1, all else place as 0.
    result[(range(batch_size),label_list)]=1
    return result

def load_final_test_data(file_path):
    final_test_file_predict_object = codecs.open(file_path, 'r', 'utf8')
    lines=final_test_file_predict_object.readlines()
    question_lists_result=[]
    for i,line in enumerate(lines):
        question_id, question_string = line.split("\t")
        question_string = question_string.strip().replace("\n","")
        question_lists_result.append((question_id, question_string))
    print("length of total question lists:",len(question_lists_result))
    return question_lists_result

def load_data_predict(vocabulary_word2index, questionid_question_lists, uni_to_tri_gram=False):  # n_words=100000,
    final_list=[]
    for i, tuplee in enumerate(questionid_question_lists):
        question_id, question_string_list = tuplee
        if uni_to_tri_gram:
            x_=process_one_sentence_to_get_ui_bi_tri_gram(question_string_list)
            x=x_.split(" ")
        else:
            x=question_string_list.split(" ")
        x = [vocabulary_word2index.get(e, 0) for e in x] #if can't find the word, set the index as '0'.(equal to PAD_ID = 0)
        if i<=2:
            print("question_id:", question_id);print("question_string_list:", question_string_list);print("x_indexed:", x)
        final_list.append((question_id, x))
    number_examples = len(final_list)
    print("number_examples:",number_examples) #
    return  final_list

def load_data_predict_y(y_file):
    y_list = codecs.open(y_file, encoding="utf-8").readlines()
    y_list = [int(x.strip()) for x in y_list]
    return y_list

def test_pad():
    trainX='w18476 w4454 w1674 w6 w25 w474 w1333 w1467 w863 w6 w4430 w11 w813 w4463 w863 w6 w4430 w111'
    trainX=trainX.split(" ")
    trainX = pad_sequences([[trainX]], maxlen=100, value=0.)
    print("trainX:",trainX)

topic_info_file_path='topic_info.txt'
def read_topic_info():
    f = codecs.open(topic_info_file_path, 'r', 'utf8')
    lines=f.readlines()
    dict_questionid_title={}
    for i,line in enumerate(lines):
        topic_id,partent_ids,title_character,title_words,desc_character,decs_words=line.split("\t").strip()
        # print(i,"------------------------------------------------------")
        # print("topic_id:",topic_id)
        # print("title_character:",title_character)
        # print("title_words:",title_words)
        # print("desc_character:",desc_character)
        # print("decs_words:",decs_words)
        dict_questionid_title[topic_id]=title_words+" "+decs_words
    print("len(dict_questionid_title):",len(dict_questionid_title))
    return dict_questionid_title

def stat_training_data_length():
    training_data='train-zhihu4-only-title-all.txt'
    f = codecs.open(training_data, 'r', 'utf8')
    lines=f.readlines()
    length_dict={0:0,5:0,10:0,15:0,20:0,25:0,30:0,35:0,40:0,100:0,150:0,200:0,1500:0}
    length_list=[0,5,10,15,20,25,30,35,40,100,150,200,1500]
    for i,line in enumerate(lines):
        line_list=line.split('__label__')[0].strip().split(" ")
        length=len(line_list)
        #print(i,"length:",length)
        for l in length_list:
            if length<l:
                length=l
                #print("length.assigned:",length)
                break
        #print("length.before dict assign:", length)
        length_dict[length]=length_dict[length]+1
    print("length_dict:",length_dict)

