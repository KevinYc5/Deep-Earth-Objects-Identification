import cv2
import numpy as np
import os
#process_img.py文件用于二值化结果图
#产生二值化图像,此代码单独运行观察最后二值化的结果，再调整阈值大小以得到一个较好的结果，不需要重新跑模型
imgdir = r'data/membrane/test_saltbody'  # 预测结果图片文件夹
outdir = r'data/membrane/result'  # 输出根据阈值二值化图片的文件夹

def Threshold(imgpath):
    img = cv2.imread(imgpath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img255 = np.zeros_like(gray, dtype='uint8')
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            if gray[i, j] > 196:  # 注意设置阈值
                img255[i, j] = 255
    return img255


filelist = os.listdir(imgdir)
for item in filelist:
    if item.endswith('_predict.png'):  # 这里网络输出的文件名,格式为'0_predict.png'
        imgpath = imgdir + os.sep + item
        # print(imgpath)
        dst = Threshold(imgpath)
        outfilepath = os.path.join(outdir, os.path.basename(item))
        cv2.imwrite(outfilepath, dst)