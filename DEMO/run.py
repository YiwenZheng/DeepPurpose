#!/usr/bin/env python3

import sys
sys.path.append("..")
from DeepPurpose import utils,dataset,DTI
import argparse

#添加argparse
parser = argparse.ArgumentParser()  #创建一个解析对象
#向该对象中添加命令行参数和选项
parser.add_argument("--dataset", type = str, required = True)  #数据集
parser.add_argument("--drug_encoding", type = str, required = True)  #药物编码器
parser.add_argument("--target_encoding", type = str, required = True)  #靶蛋白编码器
args = parser.parse_args()

#加载数据
if args.dataset == "Davis":
    X_drug, X_target, y = dataset.load_process_DAVIS('/y/home/zyw/tmp/DeepPurpose/data/', binary = False)
elif args.dataset == "Kiba":
    X_drug, X_target, y = dataset.load_process_KIBA('/y/home/zyw/tmp/DeepPurpose/data/', binary = False)
else:
    print("Don't exist such a dataset. Please try again.")
    sys.exit()  #提前结束进程

#设置编码器
drug_encoding = args.drug_encoding
target_encoding = args.target_encoding

#分割训练集、验证集和测试集
train, val, test = utils.data_process(X_drug, X_target, y, 
                                drug_encoding, target_encoding, 
                                split_method = 'random',
                                frac = [0.7,0.1,0.2])

#模型配置生成
config = utils.generate_config(drug_encoding = drug_encoding, 
                         target_encoding = target_encoding, 
                         cls_hidden_dims = [1024,1024,512], 
                         train_epoch = 100, 
                         test_every_X_epoch = 10, 
                         LR = 0.001, 
                         batch_size = 128,
                         hidden_dim_drug = 128,
                         mpnn_hidden_size = 128,
                         mpnn_depth = 3, 
                         cnn_target_filters = [32,64,96],
                         cnn_target_kernels = [4,8,12],
                         result_folder = "/y/home/zyw/tmp/DeepPurpose/result")

#模型初始化
model = DTI.model_initialize(**config)

#训练模型
model.train(train, val, test)

#保存模型
model.save_model('/y/home/zyw/tmp/DeepPurpose/save_model/model_CNN_Transformer_Davis')