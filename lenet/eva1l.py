import numpy as np
from util import *
import tensorflow as tf
import matplotlib.pyplot as plt
import os
def rectest(test_x,test_y):
    tf.reset_default_graph() 
    with tf.Session() as sess:
        model_path = os.getcwd()+'/model/'
        saver = tf.train.import_meta_graph(model_path + 'model.meta')
        saver.restore(sess,tf.train.latest_checkpoint(model_path))

        graph = tf.get_default_graph()
        x = graph.get_operation_by_name("x-input").outputs[0] 
        pred = graph.get_operation_by_name("prediction").outputs[0]
        test = graph.get_operation_by_name("out_put").outputs[0]
        pred ,test= sess.run([pred,test], feed_dict={x: test_x})
        sess.close()
    test = np.array(test)
    pred = test[:,1]
  
    print(pred)
    index = np.where(pred>0.80)
    y_threshold = np.zeros(pred.shape)
    y_threshold[index] = 1
    y_threshold = y_threshold.reshape(101,102)

    '''显示0.5概率'''
    index = np.where(pred>0.50)
    y_half = np.zeros(pred.shape)
    y_half[index] = 1
    y_half = y_half.reshape(101,102)
   
    '''显示概率'''
    y_prob = pred.reshape(101,102)
    label = np.reshape(test_y,(101,102))
    return y_threshold, y_half, y_prob,label

def main():
    data_path = os.getcwd()+'/data'
    test_data = np.load(data_path+'/test_data.npy')
    test_label = np.load(data_path+'/test_label.npy')
    test_x ,test_y = getPatch(test_data[0],test_label[0])
    test_x = np.reshape(test_x,(-1,5,5,1))
    res_threshold, res_half, res_prob, label = rectest(test_x,test_y)
    plt.imsave('res_threshold.jpg',res_threshold)
    plt.imsave('res_half.jpg',res_half)
    plt.imsave('res_prob.jpg',res_prob)
    plt.imsave('label.jpg',label)

if __name__ == '__main__':
    main()












