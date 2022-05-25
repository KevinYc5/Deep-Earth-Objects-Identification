import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import scipy.io as sio
import os
from sklearn.utils import shuffle

from util import getPatch


def importfile(path, name):
    '''
    功能：导入mat格式的数据

    :param path:文件所在的路径
    :param name:导入文件的文件名（不含格式后缀）
    :return:数据数组
    '''
    dictname = name.replace('.mat','')
    rawdata = sio.loadmat(path + '/' +  name)[dictname]

    return rawdata.transpose([0,2,1])


def get_balanced(data, label):
    '''
    功能：平衡样本中的数据类别

    :param data: 地震数据数据
    :param label: 断层标签数据
    :return: 平衡后的样本数据和标签数据(data, label)
    '''
    pos_index = np.where(label >= 0.5)
    neg_index = np.where(label < 0.5)
    rate = neg_index[0].shape[0] // pos_index[0].shape[0]
    data = np.concatenate((data, data[pos_index].repeat(rate, axis=0)), axis=0)
    label = np.concatenate((label, label[pos_index].repeat(rate, axis=0)), axis=0)
    data, label = shuffle(data, label)
    return data, label


def traintestSplit(data,label,test_rate = 0.3):
    '''
    功能：数据划分成为训练数据，测试数据并保存
    :param data: 地震数据
    :param label: 断层标签数据
    :param test_rate: 测试数据所占全部数据的比例

    '''

    currentpath = os.getcwd()
    path = currentpath + '/data'

    inline_size = data.shape[0]

    test_data = data[:(int)(test_rate*inline_size),:,:]
    test_label = label[:(int)(test_rate*inline_size),:,:]
    np.save(path+'/test_data.npy',test_data)
    np.save(path+'/test_label.npy',test_label)
    return  data[(int)(test_rate*inline_size):,:,:],label[(int)(test_rate*inline_size):,:,:]
def processTraindata(data,label):
    '''
    功能:将shape为(inline,xline,t)的地震数据和标签数据分别转换为shape为(inline×️xline×t,5,5)，(inline×️xline×t,1)的样本数据和标签数据

    :param train_data:traintestSplit返回的地震数据矩阵(array)
    :param train_label: traintestSplit返回的标签数据矩阵(array)
    :return: 训练数据(train_data,train_label)

    '''
    inline_range = np.arange(1,data.shape[0])
    train_patch ,train_label = getPatch(data[0],label[0])
    for inline in inline_range:
        temp_patch ,temp_label = getPatch(data[inline],label[inline])
        train_patch = np.append(train_patch,temp_patch,axis=0)
        train_label = np.append(train_label,temp_label,axis=0)
    train_patch,train_label = get_balanced(train_patch,train_label)
    return train_patch,train_label
def trainvalidationSplit(data,label,validation_rate=0.3):
    '''
    功能：将类别均衡后的训练数据拆分为训练数据和验证数据。

    :param data: 地震数据
    :param label: 断层标签数据
    :param validation_rate: 验证数据占验证数据和训练数据总和的比例

    '''
    currentpath = os.getcwd()
    path = currentpath + '/data'
    train_patch,valid_patch,train_label,valid_label = train_test_split(data,label,test_size=validation_rate,shuffle=True)
    np.save(path+'/train_patch.npy',train_patch)
    np.save(path+'/train_label.npy',train_label)
    np.save(path+'/valid_patch.npy',valid_patch)
    np.save(path+'/valid_label.npy',valid_label)



def show(slice_num):
    '''
    功能：可视化切片以及从该切片划分的patch
    :param slice_num:
    :return:
    '''
    currentpath = os.getcwd()
    path = currentpath + '/data'
    data = importfile(path, 'seisclean')
    label = importfile(path, 'seiscleanlable')
    patch, patch_label = getPatch(data[slice_num], label[slice_num])
    fig = plt.figure(figsize=(100, 25))
    columns = 25
    rows = 5
    for i in range(1, columns * rows + 1):
        # img = patch[4 * i]
        img_label = patch[4 * i]
        ax = fig.add_subplot(rows, columns, i)
        plt.axis('off')  # 去掉每个子图的坐标轴
        # plt.imshow(img, cmap=plt.cm.gray)
        plt.imshow(img_label)
        plt.subplots_adjust(wspace=0.1, hspace=0)  # 修改子图之间的间隔
    plt.show()

def main():
    # currentpath = os.getcwd()
    # path = currentpath + '/data'
    # data = importfile(path, 'seisclean')
    # label = importfile(path, 'seiscleanlable')
    # train_data, train_label = traintestSplit(data, label)
    # train_data, train_label = processTraindata(train_data, train_label)
    # train_data, train_label = get_balanced(train_data, train_label)
    # trainvalidationSplit(train_data, train_label, validation_rate=0.3)
    show(1)

if __name__ == '__main__':
    main()