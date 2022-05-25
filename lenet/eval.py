


import tensorflow as tf
import inferance
import preprocess
import train
import time
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder
EVAL_INTERVAL_SECS = 10


def evaluate(X_test, Y_test):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32 ,[None, inferance.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, inferance.OUTPUT_NODE], name='y-input')
        validate_feed = {x: X_test, y_: Y_test}

        y = inferance.inference(x, None, None)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(EVAL_INTERVAL_SECS)

def main(argv=None):
    X_test = np.load("test_data.npy")
    Y_test = np.load("test_label.npy")
    X_test, Y_test = preprocess.processTraindata(X_test, Y_test)
    X_test, Y_test = preprocess.get_balanced(X_test, Y_test)

    X_test = np.reshape(X_test, [-1, 5, 5, 1])
    Y_test = np.reshape(Y_test, [-1, 1])
    enc = OneHotEncoder()
    enc.fit(Y_test)
    Y_test = enc.transform(Y_test).toarray()


    # X_train = np.load("train_patch.npy")
    # Y_train = np.load("train_label.npy")
    #
    # X_train = np.reshape(X_train, [-1, 5, 5, 1])
    # Y_train = np.reshape(Y_train, [-1, 1])
    # enc = OneHotEncoder()
    # enc.fit(Y_train)
    # Y_train = enc.transform(Y_train).toarray()


    evaluate(X_test, Y_test)

if __name__ == '__main__':
    main()







