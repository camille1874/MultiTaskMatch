# -*- encoding: utf-8 -*-
import sys
import tensorflow as tf
import numpy as np
from multi_model_v1_2 import  Transformer
from data_util import load_data_multilabel_new,create_vocabulary
from preprocess_compare import Word2Vec, WikiQA, MSRP
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
tf.app.flags.DEFINE_string("ckpt_dir","../checkpoint_transformer_classification/","checkpoint location for the model")
tf.app.flags.DEFINE_integer("sequence_length",80,"max sentence length")
tf.app.flags.DEFINE_integer("embed_size",300,"embedding size")
tf.app.flags.DEFINE_boolean("is_training",True,"is training.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",50,"number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.") #每10轮做一次验证
tf.app.flags.DEFINE_integer("validate_step", 200, "how many step to validate.") #1500做一次检验
tf.app.flags.DEFINE_boolean("use_embedding",True,"whether to use embedding or not.")
#tf.app.flags.DEFINE_string("cache_path","text_cnn_checkpoint/data_cache.pik","checkpoint location for the model")
#train-zhihu4-only-title-all.txt
tf.app.flags.DEFINE_string("training_data_path","../data/wikiqa-train.txt","path of training data.") #test-zhihu4-only-title-all.txt.one record like this:'w183364 w11 w3484 w3125 w155457 w111 __label__-2086863971949478092'
tf.app.flags.DEFINE_string("word2vec_model_path","../data/GoogleNews-vectors-negative300.bin","word2vec's vocabulary and vectors") 
tf.app.flags.DEFINE_float("l2_lambda", 0.0001, "l2 regularization")

tf.app.flags.DEFINE_integer("max_len_compare",40,"max compare length")
tf.app.flags.DEFINE_integer("d_model",300,"hidden size")
tf.app.flags.DEFINE_integer("d_k",75,"hidden size")
tf.app.flags.DEFINE_integer("d_v",75,"hidden size")
tf.app.flags.DEFINE_integer("h",4,"hidden size")
tf.app.flags.DEFINE_integer("num_layer",3,"hidden size") 

def main(_):
    #1.load data(X:list of lint,y:int).
    #if os.path.exists(FLAGS.cache_path):  # 如果文件系统中存在，那么加载故事（词汇表索引化的）
    #    with open(FLAGS.cache_path, 'r') as data_f:
    #        trainX, trainY, testX, testY, vocabulary_index2word=pickle.load(data_f)
    #        vocab_size=len(vocabulary_index2word)
    #else:
    if 1==1:
        trainX, trainY, testX, testY = None, None, None, None
        vocabulary_word2index,vocabulary_index2word = create_vocabulary(word2vec_model_path=FLAGS.word2vec_model_path,name_scope="transformer_classification") 
        vocab_size = len(vocabulary_word2index)
        print("transformer.vocab_size:",vocab_size)
        train,test,_=load_data_multilabel_new(vocabulary_word2index,training_data_path=FLAGS.training_data_path)

        compare_train_data = WikiQA(word2vec=Word2Vec(), max_len=FLAGS.max_len_compare)
        compare_train_data.open_file(mode="train")
        compare_test_data = WikiQA(word2vec=Word2Vec(), max_len=FLAGS.max_len_compare)
        compare_test_data.open_file(mode="valid")

        trainX, trainY, = train
        testX, testY = test

        trainX = pad_sequences(trainX, maxlen=FLAGS.sequence_length, value=0.)  
        testX = pad_sequences(testX, maxlen=FLAGS.sequence_length, value=0.)
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        model=Transformer(FLAGS.num_classes, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps, FLAGS.decay_rate, FLAGS.sequence_length,
                 vocab_size, FLAGS.embed_size,FLAGS.d_model,FLAGS.d_k,FLAGS.d_v,FLAGS.h,FLAGS.num_layer,FLAGS.is_training, compare_train_data.num_features, di=50, s=compare_train_data.max_len, w=4, l2_reg=0.0004, l2_lambda=FLAGS.l2_lambda)
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
                ##VALIDATION VALIDATION VALIDATION PART######################################################################################################
                if FLAGS.batch_size!=0 and (start%(FLAGS.validate_step*FLAGS.batch_size)==0):
                    eval_loss, eval_acc = do_eval(sess, model, testX, testY, compare_test_data, batch_size)
                    print("transformer.classification.validation.part. previous_eval_loss:", previous_eval_loss,";current_eval_loss:",eval_loss)
                    if eval_loss > previous_eval_loss: #if loss is not decreasing
                        # reduce the learning rate by a factor of 0.5
                        print("transformer.classification.==>validation.part.going to reduce the learning rate.")
                        learning_rate1 = sess.run(model.learning_rate)
                        lrr=sess.run([model.learning_rate_decay_half_op])
                        learning_rate2 = sess.run(model.learning_rate)
                        print("transformer.classification==>validation.part.learning_rate1:", learning_rate1, " ;learning_rate2:",learning_rate2)
                    #print("HierAtten==>Epoch %d Validation Loss:%.3f\tValidation Accuracy: %.3f" % (epoch, eval_loss, eval_acc))
                    else:# loss is decreasing
                        if eval_loss<best_eval_loss:
                            print("transformer.classification==>going to save the model.eval_loss:",eval_loss,";best_eval_loss:",best_eval_loss)
                            # save model to checkpoint
                            save_path = FLAGS.ckpt_dir + "model.ckpt"
                            saver.save(sess, save_path, global_step=epoch)
                            best_eval_loss=eval_loss
                    previous_eval_loss = eval_loss
                    compare_test_data.reset_index()
                ##VALIDATION VALIDATION VALIDATION PART######################################################################################################

            #epoch increment
            print("going to increment epoch counter....")
            sess.run(model.epoch_increment)


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

# 在验证集上做验证，报告损失、精确度
def do_eval(sess,model,evalX,evalY,compare_test_data,batch_size,eval_decoder_input=None):
    #ii=0
    number_examples=len(evalX)
    print(number_examples)
    eval_loss,eval_acc,eval_counter=0.0,0.0,0
    for start,end in zip(range(0,number_examples,batch_size),range(batch_size,number_examples,batch_size)):
        batch_x1, batch_x2, _, batch_features = compare_test_data.next_batch(batch_size=end - start)
        feed_dict = {model.input_x: evalX[start:end], model.dropout_keep_prob: 1.0, model.x1: batch_x1, model.x2: batch_x2, model.features:batch_features}
        feed_dict[model.input_y_label] = evalY[start:end]
        curr_eval_loss, logits,curr_eval_acc,pred= sess.run([model.loss_val,model.logits,model.accuracy,model.predictions],feed_dict)#curr_eval_acc--->textCNN.accuracy
        eval_loss,eval_acc,eval_counter=eval_loss+curr_eval_loss,eval_acc+curr_eval_acc,eval_counter+1
        #if ii<20:
            #print("1.evalX[start:end]:",evalX[start:end])
            #print("2.evalY[start:end]:", evalY[start:end])
            #print("3.pred:",pred)
            #ii=ii+1
    return eval_loss/float(eval_counter),eval_acc/float(eval_counter)

if __name__ == "__main__":
    tf.app.run()
