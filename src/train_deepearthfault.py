import tensorflow as tf  
import inferance  
import os  
import numpy as np  
import  matplotlib.pyplot as plt  
    
# %% md  
    
#### 1. 定义神经网络相关的参数  
    
# %%  
    
def train(data, label, totel_num, valid_data, valid_label):  
    BATCH_SIZE = 32  
    LEARNING_RATE_BASE = 0.01  
    LEARNING_RATE_DECAY = 0.99  
    REGULARIZATION_RATE = 0.0001  
    TRAINING_STEPS = 1000  
    MOVING_AVERAGE_DECAY = 0.99  
    x = tf.placeholder(tf.float32, [  
        None,  
        inferance.IMAGE_SIZE,  
        inferance.IMAGE_SIZE,  
        inferance.NUM_CHANNELS],  
                        name='x-input')  
    y_ = tf.placeholder(tf.float32, [None, inferance.OUTPUT_NODE], name='y-input')  
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)  
    y = inferance.inference(x, False, regularizer)  
    global_step = tf.Variable(0, trainable=False)  
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)  
    variables_averages_op = variable_averages.apply(tf.trainable_variables())  
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))  
    cross_entropy_mean = tf.reduce_mean(cross_entropy)  
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))  
    out_put = tf.nn.softmax(y, name='out_put')  
    res_pre = tf.cast(tf.argmax(tf.nn.softmax(y), 1), tf.float32, name='prediction')  
    
    correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  
    learning_rate = tf.train.exponential_decay(  
        LEARNING_RATE_BASE,  
        global_step,  
        totel_num // BATCH_SIZE, LEARNING_RATE_DECAY,  
        staircase=True)  
    
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)  
    with tf.control_dependencies([train_step, variables_averages_op]):  
        train_op = tf.no_op(name='train')  
    saver = tf.train.Saver()  
    loss_arr = []  
    acc_arr = []  
    with tf.Session() as sess:  
        # 损失  
        # 正确率  
        tf.global_variables_initializer().run()  
        start = 0  
        end = BATCH_SIZE  
        for i in range(TRAINING_STEPS):  
            xs = data[start:end]  
            ys = tf.one_hot(label[start:end], 2).eval()  
    
            reshaped_xs = np.reshape(xs, (  
                BATCH_SIZE,  
                inferance.IMAGE_SIZE,  
                inferance.IMAGE_SIZE,  
                inferance.NUM_CHANNELS))  
            _, loss_value, step, acc = sess.run([train_op, loss, global_step, accuracy],  
                                                feed_dict={x: reshaped_xs, y_: ys})  
            loss_arr.append(loss_value)  
            acc_arr.append(acc)  
            if i % 100 == 0:  
                print("After %d training step(s), loss on training batch is %g" % (step, loss_value))  
            start = end  
            end = start + BATCH_SIZE  
        batch_x_validation = np.reshape(valid_data[0:1000], (-1, 5, 5, 1))  
        batch_y_validation = tf.one_hot(valid_label[0:1000], 2).eval()  
        acc = sess.run([accuracy], feed_dict={x: batch_x_validation, y_: batch_y_validation})  
        print(("accuracy = "), acc)  
        path = os.getcwd() + '/model'  
        if (not os.path.exists(path)):  
            os.mkdir(path)  
        saver.save(sess, path + '/model')  
        sess.close()  
    return loss_arr, acc_arr  
    
    
# 画出训练过程中的损失变化  
def paint_loss(fig, training_steps, loss_arr):  
#     print("loss", loss_arr)  
    ax = fig.add_subplot(1, 2, 1)  # 整个绘图区分成一行两列，当前图是第一个。  
    ax.set_title("loss")  
    ax.set_xlabel("training_step")  
    ax.set_ylabel("loss_arr")  
    ax.plot(range(0, training_steps), loss_arr)  
    
# 画出训练过程中的acc变化  
def paint_acc(fig, training_steps, acc_arr):  
#     print("loss", loss_arr)  
    ax = fig.add_subplot(1, 2, 2)  # 整个绘图区分成一行两列，当前图是第二个。  
    ax.set_title("accuracy")  
    ax.set_xlabel("training_step")  
    ax.set_ylabel("acc_arr")  
    ax.plot(range(0, training_steps), acc_arr)  
def main():  
    path = os.getcwd()  
    
    data = np.load(path + '/data/train_patch.npy')  
    label = np.load(path + '/data/train_label.npy')  
    valid_data = np.load(path + '/data/valid_patch.npy')  
    valid_label = np.load(path + '/data/valid_label.npy')  
    totel_num = data.shape[0]  
    loss_arr, acc_arr = train(data, label, totel_num, valid_data, valid_label)  
    fig = plt.figure()  
    
    
    fig.set_size_inches(10, 4)  # 整个绘图区域的宽度10和高度4  
    paint_loss(fig, 1000, loss_arr)  
    paint_acc(fig, 1000, acc_arr)  
    plt.show()  
    
if __name__ == '__main__':  
    main()
