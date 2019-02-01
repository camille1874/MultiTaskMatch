import tensorflow as tf
import numpy as np
from multi_model_v1 import  Transformer
from data_util import load_data_predict,load_data_predict_y,load_final_test_data,create_vocabulary
from tflearn.data_utils import pad_sequences #to_categorical
import os
import codecs
from preprocess_abcnn import Word2Vec, MSRP, WikiQA
from ABCNN_raw import ABCNN

#configuration
FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("num_classes",2,"number of label")
tf.app.flags.DEFINE_float("learning_rate",0.01,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size for training/evaluating.") #批处理的大小 32-->128 #16
tf.app.flags.DEFINE_integer("decay_steps", 6000, "how many steps before decay learning rate.") #6000批处理的大小 32-->128
tf.app.flags.DEFINE_float("decay_rate", 0.001, "Rate of decay for learning rate.") #0.87一次衰减多少
tf.app.flags.DEFINE_string("ckpt_dir","../checkpoint_transformer_classification/","checkpoint location for the model")
tf.app.flags.DEFINE_integer("sequence_length",80,"max sentence length") #100-->25
tf.app.flags.DEFINE_integer("embed_size",300,"embedding size")
tf.app.flags.DEFINE_boolean("is_training",False,"is traning.true:tranining,false:testing/inference")
#tf.app.flags.DEFINE_string("cache_path","text_cnn_checkpoint/data_cache.pik","checkpoint location for the model")
#train-zhihu4-only-title-all.txt
tf.app.flags.DEFINE_string("word2vec_model_path","../data/GoogleNews-vectors-negative300.bin","word2vec's vocabulary and vectors") 
tf.app.flags.DEFINE_boolean("multi_label_flag",False,"use multi label or single label.") #set this false. becase we are using it is a sequence of token here.
tf.app.flags.DEFINE_float("l2_lambda", 0.0001, "l2 regularization")
tf.app.flags.DEFINE_string("predict_target_file","../checkpoint_transformer_classification/result_transformer_classification.csv","target file path for final prediction")
tf.app.flags.DEFINE_string("predict_source_file",'../data/wikiqa-test-x.txt',"target file path for final prediction") 
tf.app.flags.DEFINE_string("predict_source_file_y",'../data/wikiqa-test-y.txt',"target file path for final prediction") 
tf.app.flags.DEFINE_integer("d_model",300,"hidden size")
tf.app.flags.DEFINE_integer("d_k",75,"hidden size")
tf.app.flags.DEFINE_integer("d_v",75,"hidden size")
tf.app.flags.DEFINE_integer("h",4,"hidden size")
tf.app.flags.DEFINE_integer("num_layer",3,"hidden size") #6

tf.app.flags.DEFINE_integer("max_compare_len",40,"max compare length")
#1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
# 1.load data with vocabulary of words and labels
_GO="_GO"
_END="_END"
_PAD="_PAD"

def main(_):
    # 1.load data with vocabulary of words and labels
    compare_test_data = WikiQA(word2vec=Word2Vec(), max_len=FLAGS.max_compare_len)
    compare_test_data.open_file(mode="test")

    vocabulary_word2index, vocabulary_index2word = create_vocabulary(word2vec_model_path=FLAGS.word2vec_model_path,name_scope="transformer_classification")  # simple='simple'
    vocab_size = len(vocabulary_word2index)
    print("transformer_classification.vocab_size:", vocab_size)
    questionid_question_lists=load_final_test_data(FLAGS.predict_source_file)
    print("list of total questions:",len(questionid_question_lists))
    test= load_data_predict(vocabulary_word2index,questionid_question_lists)
    print("list of total questions2:",len(test))
    testX=[]
    question_id_list=[]
    for tuple in test:
        question_id,question_string_list=tuple
        question_id_list.append(question_id)
        testX.append(question_string_list)
    # 2.Data preprocessing: Sequence padding
    print("start padding....")
    testX2 = pad_sequences(testX, maxlen=FLAGS.sequence_length, value=0.)  # padding to max length
    testY2 = load_data_predict_y(FLAGS.predict_source_file_y)
    print("list of total questions3:", len(testX2))
    print("end padding...")
   # 3.create session.
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        # 4.Instantiate Model
        model=Transformer(FLAGS.num_classes, FLAGS.learning_rate, len(testX2), FLAGS.decay_steps, FLAGS.decay_rate, FLAGS.sequence_length,
                 vocab_size, FLAGS.embed_size,FLAGS.d_model,FLAGS.d_k,FLAGS.d_v,FLAGS.h,FLAGS.num_layer,FLAGS.is_training,compare_test_data.num_features, di=50, s=compare_test_data.max_len, w=4, l2_reg=0.0004, l2_lambda=FLAGS.l2_lambda)
        saver=tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
            print("Restoring Variables from Checkpoint")
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print("Can't find the checkpoint.going to stop")
            return
        # 5.feed data, to get logits
        number_of_training_data=len(testX2);print("number_of_training_data:",number_of_training_data)
        batch_x1, batch_x2, _, batch_features = compare_test_data.next_batch(batch_size=number_of_training_data)
        predict_target_file_f = codecs.open(FLAGS.predict_target_file, 'w', 'utf-8')
        logits=sess.run(model.return_logits,feed_dict={model.input_x:testX2,model.input_y_label:testY2,model.dropout_keep_prob:1, model.x1:batch_x1, model.x2: batch_x2, model.features:batch_features}) #logits:[batch_size,self.num_classes]
       
        answers = {}
        MAP, MRR = 0, 0
        total = len(logits)
        for i in range(total):
            #prob = logits[i][1] - logits[i][0]
            prob = logits[i][1]
            if question_id_list[i] in answers:
                answers[question_id_list[i]].append((testX[i], testY2[i], prob))
            else:                
                answers[question_id_list[i]] = [(testX[i], testY2[i], prob)]
            predict_target_file_f.write(str(logits[i]) + "\n")
        for i in answers.keys():
            p, AP = 0, 0
            MRR_check = False
            answers[i] = sorted(answers[i], key=lambda x: x[-1], reverse=True) 
            for idx, (s, label, prob) in enumerate(answers[i]):
                if label == 1:
                    if not MRR_check:
                        MRR += 1 / (idx + 1)
                        MRR_check = True
                    p += 1
                    AP += p / (idx + 1)

            AP /= p
            MAP += AP
        
        total_q = len(answers.keys()) 
        MAP /= total_q
        MRR /= total_q
        print("MAP", MAP, ",MRR", MRR)

        predict_target_file_f.close()

if __name__ == "__main__":
    tf.app.run()
