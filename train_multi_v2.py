# -*- encoding: utf-8 -*-
import sys
import tensorflow as tf
import numpy as np
from multi_model_v2_2 import TextCNN
from data_util import load_data_multilabel_new,create_vocabulary
from preprocess_compare import Word2Vec, WikiQA
from tflearn.data_utils import to_categorical, pad_sequences
import os,math
import pickle
import gensim

#configuration
FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("num_classes",2,"number of label") 
tf.app.flags.DEFINE_float("learning_rate",0.01,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size for training/evaluating.") #批处理的大小 32-->128-->512
tf.app.flags.DEFINE_integer("decay_steps", 6000, "how many steps before decay learning rate.") #6000批处理的大小 32-->128
tf.app.flags.DEFINE_float("decay_rate", 0.001, "Rate of decay for learning rate.") #0.87一次衰减多少
tf.app.flags.DEFINE_string("ckpt_dir","../checkpoint_wikiqa/","checkpoint location for the model")
#tf.app.flags.DEFINE_integer("sequence_length",80,"max sentence length")
tf.app.flags.DEFINE_integer("sequence_length",100,"max sentence length")
tf.app.flags.DEFINE_integer("embed_size",300,"embedding size")
tf.app.flags.DEFINE_boolean("is_training",True,"is training.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",50,"number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.") #每10轮做一次验证
tf.app.flags.DEFINE_integer("validate_step", 200, "how many step to validate.") #1500做一次检验
tf.app.flags.DEFINE_boolean("use_embedding",True,"whether to use embedding or not.")
#tf.app.flags.DEFINE_string("cache_path","text_cnn_checkpoint/data_cache.pik","checkpoint location for the model")
#train-zhihu4-only-title-all.txt
tf.app.flags.DEFINE_string("training_data_path","../data/wikiqa-train0.txt","path of training data.") 
#tf.app.flags.DEFINE_string("training_data_path","../data/wikiqa-train-processed.txt","path of training data.") 
tf.app.flags.DEFINE_string("word2vec_model_path","../data/GoogleNews-vectors-negative300.bin","word2vec's vocabulary and vectors") 
tf.app.flags.DEFINE_float("l2_lambda", 0.0001, "l2 regularization")

tf.app.flags.DEFINE_integer("max_len_compare",50,"max compare length")
tf.app.flags.DEFINE_integer("d_model",300,"model hidden size")
tf.app.flags.DEFINE_string("filter_sizes", "3", "Comma-separated filter sizes")
#tf.app.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size")
tf.app.flags.DEFINE_integer("num_filters", 1, "Number of filters per filter size")
def main(_):
    #1.load data(X:list of lint,y:int).
    #if os.path.exists(FLAGS.cache_path):  # 如果文件系统中存在，那么加载故事（词汇表索引化的）
    #    with open(FLAGS.cache_path, 'r') as data_f:
    #        trainX, trainY, testX, testY, vocabulary_index2word=pickle.load(data_f)
    #        vocab_size=len(vocabulary_index2word)
    #else:
    if 1==1:
        trainX, trainY = None, None
        vocabulary_word2index,vocabulary_index2word = create_vocabulary(word2vec_model_path=FLAGS.word2vec_model_path,name_scope="transformer_classification") 
        vocab_size = len(vocabulary_word2index)
        print("transformer.vocab_size:",vocab_size)
        train=load_data_multilabel_new(vocabulary_word2index,training_data_path=FLAGS.training_data_path)

        compare_train_data = WikiQA(word2vec=Word2Vec(), max_len=FLAGS.max_len_compare)
        compare_train_data.open_file(mode="train")
        compare_test_data = WikiQA(word2vec=Word2Vec(), max_len=FLAGS.max_len_compare)

        trainX, trainY, = train

        trainX = pad_sequences(trainX, maxlen=FLAGS.sequence_length, value=0.)  
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        model=TextCNN(FLAGS.num_classes, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps, FLAGS.decay_rate, FLAGS.sequence_length,
                 vocab_size, FLAGS.embed_size,FLAGS.d_model,list(map(int, FLAGS.filter_sizes.split(","))),FLAGS.num_filters,FLAGS.is_training, compare_train_data.num_features, di=50, s=compare_train_data.max_len, w=4, l2_reg=0.0004, l2_lambda=FLAGS.l2_lambda)
        print("=" * 50)
        print("List of Variables:")
        for v in tf.trainable_variables():
            print(v.name)
        print("=" * 50)
        saver=tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
            print("Restoring Variables from Checkpoint")
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            if FLAGS.use_embedding: #load pre-trained word embedding
                assign_pretrained_word_embedding(sess, vocabulary_index2word,vocab_size, model,word2vec_model_path=FLAGS.word2vec_model_path)
        curr_epoch=sess.run(model.epoch_step)
        number_of_training_data=len(trainX)
        print("number_of_training_data:",number_of_training_data)

        previous_eval_loss=10000
        best_eval_loss=10000
        batch_size=FLAGS.batch_size
        for epoch in range(curr_epoch,FLAGS.num_epochs):
            loss, acc, counter = 0.0, 0.0, 0
            compare_train_data.reset_index()
            for start, end in zip(range(0, number_of_training_data, batch_size),range(batch_size, number_of_training_data, batch_size)):
                if epoch==0 and counter==0:
                    print("trainX[start:end]:",trainX[start:end])
                batch_x1, batch_x2, _, batch_features = compare_train_data.next_batch(batch_size=end - start)
                feed_dict = {model.input_x: trainX[start:end],model.dropout_keep_prob: 0.9, model.x1: batch_x1, model.x2: batch_x2, model.features:batch_features}
                feed_dict[model.input_y_label]=trainY[start:end]
                curr_loss,curr_acc,_=sess.run([model.loss_val,model.accuracy,model.train_op],feed_dict) #curr_acc--->TextCNN.accuracy
                loss,counter,acc=loss+curr_loss,counter+1,acc+curr_acc
                if counter %50==0:
                    print("transformer.classification==>Epoch %d\tBatch %d\tTrain Loss:%.3f\tTrain Accuracy:%.3f" %(epoch,counter,loss/float(counter),acc/float(counter))) #tTrain Accuracy:%.3f---》acc/float(counter)
            print("going to increment epoch counter....")
            sess.run(model.epoch_increment)
        save_path = FLAGS.ckpt_dir + "model.ckpt"
        saver.save(sess, save_path, global_step=epoch)

def assign_pretrained_word_embedding(sess,vocabulary_index2word,vocab_size,model,word2vec_model_path=None):
    print("using pre-trained word emebedding.started.word2vec_model_path:",word2vec_model_path)
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)
    word2vec_dict = {}
    for word, vector in zip(word2vec_model.vocab, word2vec_model.vectors):
        word2vec_dict[word] = vector
    word_embedding_2dlist = [[]] * vocab_size  # create an empty word_embedding list.
    word_embedding_2dlist[0] = np.zeros(FLAGS.embed_size)  # assign empty for first word:'PAD'
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # bound for random variables.
    count_exist = 0;
    count_not_exist = 0
    for i in range(1, vocab_size):  # loop each word
        word = vocabulary_index2word[i]  # get a word
        embedding = None
        try:
            embedding = word2vec_dict[word]  # try to get vector:it is an array.
        except Exception:
            embedding = None
        if embedding is not None:  # the 'word' exist a embedding
            word_embedding_2dlist[i] = embedding;
            count_exist = count_exist + 1  # assign array to this word.
        else:  # no embedding for this word
            word_embedding_2dlist[i] = np.random.uniform(-bound, bound, FLAGS.embed_size);
            count_not_exist = count_not_exist + 1  # init a random value for the word.
    word_embedding_final = np.array(word_embedding_2dlist)  # covert to 2d array.
    #word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)  # convert to tensor
    #t_assign_embedding = tf.assign(model.Embedding,word_embedding)  
    sess.run(model.embedding_init, feed_dict={model.embedding_placeholder: word_embedding_final});
    print("word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word emebedding.ended...")


if __name__ == "__main__":
    tf.app.run()
