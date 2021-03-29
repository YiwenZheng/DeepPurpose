#!/usr/bin/env python3

from utils import generate_config
from argparse import ArgumentParser
from train_CompoundPred import train_model


def get_parser():
    parser = ArgumentParser()
    # base config
    parser.add_argument("-i", "--input_file", type = str, required = True,\
        help = "Path to data file")
    parser.add_argument("-o", "--output_dir", type = str, required = True,\
        help = "Directory where model will be saved")
    parser.add_argument("-r", "--result_folder", type = str, default = "./result/",\
        required = False, help = "Folder of result") 
    parser.add_argument("-d_e", "--drug_encoding", type = str, default = "Morgan",\
        required = False, help = "Drug encoding")
    parser.add_argument("--input_dim_drug", type = int, default = 1024,\
        required = False,help = "Dimensionality of input of drug") 
    parser.add_argument("--hidden_dim_drug", type = int, default = 256,\
        required = False, help = "Dimensionality of hidden layers of drug")  
    parser.add_argument("--cls_hidden_dims", type = list, default = [1024, 1024, 512],\
        required = False, help = "Dimensionality of hidden layers of decoder classifier") 
    parser.add_argument("--batch_size", type = int, default = 256, required = False,\
        help = "Batch size") 
    parser.add_argument("--train_epoch", type = int, default = 10, required = False,\
        help = "Number of epochs during training") 
    parser.add_argument("--test_every_X_epoch", type = int, default = 20,\
        required = False, help = "Number of epochs during testing") 
    parser.add_argument("--LR", type = int, default = 1e-4, required = False,\
        help = "Learing rate") 
    parser.add_argument("--decay", type = int, default = 0, required = False,\
        help = "Decay") 
    parser.add_argument("--num_workers", type = int, default = 0, required = False,\
        help = "Number of workers") 

    # config of drug
    parser.add_argument("--mlp_hidden_dims_drug", type = list, default = [1024, 256, 64], required = False,\
        help = "Dimensionality of hidden layers of drug in MLP") 

    return parser


#模型配置生成
def get_model_config(config):
    model_config = generate_config(drug_encoding = config.drug_encoding, 
            result_folder = config.result_folder,
            input_dim_drug = config.input_dim_drug, 
            hidden_dim_drug = config.hidden_dim_drug, 
            cls_hidden_dims = config.cls_hidden_dims,
            batch_size = config.batch_size,
            train_epoch = config.train_epoch,
            test_every_X_epoch = config.test_every_X_epoch,
            LR = config.LR,
            decay = config.decay,
            num_workers = config.num_workers,
            mlp_hidden_dims_drug = config.mlp_hidden_dims_drug)
    return model_config

if __name__ == "__main__":
    parser = get_parser()
    config = parser.parse_known_args()[0]
    train_model(config)