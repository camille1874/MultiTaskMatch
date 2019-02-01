import tensorflow as tf
import numpy as np
import sys

from preprocess_abcnn import Word2Vec, WikiQA, MSRP
from ABCNN_raw import ABCNN
from utils import build_path
#from sklearn import linear_model, svm
from sklearn import linear_model
from sklearn.externals import joblib


def train(lr, w, l2_reg, epoch, batch_size, model_type, num_layers, data_type, word2vec, num_classes=2):
    if data_type == "WikiQA":
        train_data = WikiQA(word2vec=word2vec)
    elif data_type == "MSRP":
        train_data = MSRP(word2vec=word2vec)

    train_data.open_file(mode="train")

    print("=" * 50)
    print("training data size:", train_data.data_size)
    print("training max len:", train_data.max_len)
    print("=" * 50)

    model = ABCNN(s=train_data.max_len, w=w, l2_reg=l2_reg, model_type=model_type,
                  num_features=train_data.num_features, num_classes=num_classes, num_layers=num_layers)

    optimizer = tf.train.AdagradOptimizer(lr, name="optimizer").minimize(model.cost)

    # Due to GTX 970 memory issues
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)

    init = tf.global_variables_initializer()

    # keep no more than 100 models
    saver = tf.train.Saver(max_to_keep=100)
    
    session_config = tf.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth=True
    with tf.Session(config=session_config) as sess:
    #with tf.Session() as sess:
        train_summary_writer = tf.summary.FileWriter("C:/tf_logs/train", sess.graph)

        sess.run(init)

        print("=" * 50)
        for e in range(1, epoch + 1):
            print("[Epoch " + str(e) + "]")

            train_data.reset_index()
            i = 0

            clf_features = []

            while train_data.is_available():
                i += 1

                batch_x1, batch_x2, batch_y, batch_features = train_data.next_batch(batch_size=batch_size)

                merged, _, c, features = sess.run([model.merged, optimizer, model.cost, model.output_features],
                                                  feed_dict={model.x1: batch_x1,
                                                             model.x2: batch_x2,
                                                             model.y: batch_y,
                                                             model.features: batch_features})

                clf_features.append(features)

                if i % 100 == 0:
                    print("[batch " + str(i) + "] cost:", c)
                train_summary_writer.add_summary(merged, i)

            save_path = saver.save(sess, build_path("./models/", data_type, model_type, num_layers), global_step=e)
            print("model saved as", save_path)

            clf_features = np.concatenate(clf_features)


        print("training finished!")
        print("=" * 50)


if __name__ == "__main__":

    # Paramters
    # --lr: learning rate
    # --ws: window_size
    # --l2_reg: l2_reg modifier
    # --epoch: epoch
    # --batch_size: batch size
    # --model_type: model type
    # --num_layers: number of convolution layers
    # --data_type: WebQSP or SQ data

    # default parameters
    params = {
        "lr": 0.08,
        "ws": 4,
        "l2_reg": 0.0004,
        "epoch": 50,
        "batch_size": 64,
        "model_type": "ABCNN2",
        "num_layers": 1,
        "data_type": "WikiQA",
        "word2vec": Word2Vec()
    }

    print("=" * 50)
    print("Parameters:")
    for k in sorted(params.keys()):
        print(k, ":", params[k])

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            k = arg.split("=")[0][2:]
            v = arg.split("=")[1]
            params[k] = v

    train(lr=float(params["lr"]), w=int(params["ws"]), l2_reg=float(params["l2_reg"]), epoch=int(params["epoch"]),
          batch_size=int(params["batch_size"]), model_type=params["model_type"], num_layers=int(params["num_layers"]),
          data_type=params["data_type"], word2vec=params["word2vec"])
