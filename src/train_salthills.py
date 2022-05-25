from unet_inference import *  
from unet_data import *  # 导入这两个文件中的所有函数  
from draw_loss import LossHistory  
import numpy as np  
import os  
    
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
# 代码先执行unet_train.py训练模型，再执行unet_eval.py文件预测结果  
# unet_train.py文件用于训练模型#训练图片的质量也会影响模型的质量  
data_gen_args = dict(rotation_range=0.2,  
                        width_shift_range=0.05,  
                        height_shift_range=0.05,  
                        shear_range=0.05,  
                        zoom_range=0.05,  
                        horizontal_flip=True,  
                        fill_mode='nearest')  # 数据增强时的变换方式的字典  
# 得到一个生成器，以batch=2的速率生成增强后的数据  
myGene = trainGenerator(2, 'data/membrane/train', 'image_saltbody', 'label_saltbody',  
                        data_gen_args, save_to_dir=None)  
history = LossHistory()  
model = unet()  
# 回调函数，第一个是保存模型路径，第二个是检测的值，检测Loss是使它最小， 第三个是只保存在验证集上性能最好的模型  
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True)  
# steps_per_epoch指的是每个epoch有多少个batch, size, 也就是训练集总样本数除以batch_size的值  
# 下面一行是利用生成器进行batch_size, 样本和标签通过myGene传入  
# 注意调参。steps_per_epoch, epochs的值对模型结果影响较大。  
model.fit_generator(myGene, steps_per_epoch=300, epochs=1, callbacks=[model_checkpoint, history])  
# 绘制acc- loss曲线  
# history.loss_plot('batch')  
history.loss_plot()
