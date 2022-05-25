
import numpy as np
def getData(pos,seisdata,size):
    Half_size = size//2
    data = np.zeros([size,size])
    xline = pos[0]
    t = pos[1]
    width = np.arange(size)
    length = np.arange(size)
    for i in width:
        for j in length:
            data[i,j] = seisdata[xline+i-Half_size,t+j-Half_size]
    return data

def getPatch(slice,label,size = 5,padding = True):
    '''

    :param slice: 输入的切片数据
    :param label: 切片数据的标签
    :param size: patch的size 在本实验中大小为5
    :param padding: 是否采用padding的方式，本实验中采用。
    :return: patch，label
    '''
    Half_size = size//2
    slice_pad = slice
    if padding:
        slice_pad = np.pad(slice,((Half_size,Half_size),(Half_size,Half_size)),mode="constant",constant_values=0)
    xline_size = slice_pad.shape[0]
    t_size = slice_pad.shape[1]
    xline_range = np.arange(Half_size,xline_size-Half_size)
    t_range = np.arange(Half_size,t_size-Half_size)
    patch_label = []
    patch = []
    for xline in xline_range:
        for t in t_range:
            patch.append(getData((xline,t),slice_pad,size))
            patch_label.append(label[xline-Half_size,t-Half_size])
    return np.asarray(patch),np.asarray(patch_label)
