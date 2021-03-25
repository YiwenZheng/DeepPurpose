#!/usr/bin/env python3

import utils,dataset,DTI
from argparse import ArgumentParser

def get_parser():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_file", type = str, required = True, help = "Path to data file")
    parser.add_argument("-o", "--output_dir", type = str, required = True,\
        help = "Directory where model will be saved")
    # generate_config
    parser.add_argument("-d_e", "--drug_encoding", type = str, default = None, required = True,\
        help = "Drug encoding")
    parser.add_argument("-t_e", "--target_encoding", type = str, default = None, required = True,\
        help = "Target encoding")
    parser.add_argument("--result_folder", type = str, default = "./result/", required = False,\
        help = "Folder of result") 
    parser.add_argument("--input_dim_drug", type = int, default = 1024, required = False,\
        help = "Dimensionality of input of drug") 
    parser.add_argument("--input_dim_protein", type = int, default = 8420, required = False,\
        help = "Dimensionality of input of protein") 
    parser.add_argument("--hidden_dim_drug", type = int, default = 256, required = False,\
        help = "Dimensionality of hidden layers of drug") 
    parser.add_argument("--hidden_dim_protein", type = int, default = 256, required = False,\
        help = "Dimensionality of hidden layers of protein") 
    parser.add_argument("--cls_hidden_dims", type = list, default = [1024, 1024, 512], required = False,\
        help = "Dimensionality of hidden layers of drug in CLS") 
    parser.add_argument("--mlp_hidden_dims_drug", type = list, default = [1024, 256, 64], required = False,\
        help = "Dimensionality of hidden layers of drug in MLP") 
    parser.add_argument("--mlp_hidden_dims_target", type = list, default = [1024, 256, 64], required = False,\
        help = "Dimensionality of hidden layers of target in MLP") 
    parser.add_argument("--batch_size", type = int, default = 256, required = False, help = "Batch size") 
    parser.add_argument("--train_epoch", type = int, default = 10, required = False,\
        help = "Number of epochs during training") 
    parser.add_argument("--test_every_X_epoch", type = int, default = 20, required = False,\
        help = "Number of epochs during testing") 
    parser.add_argument("--LR", type = int, default = 1e-4, required = False, help = "Learing rate") 
    parser.add_argument("--decay", type = int, default = 0, required = False, help = "Decay") 
    parser.add_argument("--transformer_emb_size_drug", type = int, default = 128, required = False,\
        help = "Size of embedding of drug in Transformer") 
    parser.add_argument("--transformer_intermediate_size_drug", type = int, default = 512, required = False,\
        help = "Size of intermediate layers of drug in Transformer") 
    parser.add_argument("--transformer_num_attention_heads_drug", type = int, default = 8, required = False,\
        help = "Number of heads of attention layers of drug in Transformer") 
    parser.add_argument("--transformer_n_layer_drug", type = int, default = 8, required = False,\
        help = "Number of layers of drug in Transfromer") 
    parser.add_argument("--transformer_emb_size_target", type = int, default = 64, required = False,\
        help = "Size of embedding of target in Transformer") 
    parser.add_argument("--transformer_intermediate_size_target", type = int, default = 256, required = False,\
        help = "Size of intermediate layers of target in Transformer") 
    parser.add_argument("--transformer_num_attention_heads_target", type = int, default = 4, required = False,\
        help = "Number of heads of attention layers of target in Transformer") 
    parser.add_argument("--transformer_n_layer_target", type = int, default = 2, required = False,\
        help = "Number of layers of target in Transfromer") 
    parser.add_argument("--transformer_dropout_rate", type = int, default = 0.1, required = False,\
        help = "Dropout rate of in Transformer") 
    parser.add_argument("--transformer_attention_probs_dropout", type = int, default = 0.1, required = False,\
        help = "Dropout probability of attention layers in Transformer") 
    parser.add_argument("--transformer_hidden_dropout_rate", type = int, default = 0.1, \
        required = False, help = "Dropout rate of hidden layers in Transformer") 
    parser.add_argument("--mpnn_hidden_size", type = int, default = 50, required = False, \
        help = "Dimensionality of hidden layers in MPNN") 
    parser.add_argument("--mpnn_depth", type = int, default = 3, required = False, help = "Depth of MPNN") 
    parser.add_argument("--cnn_drug_filters", type = list, default = [32,64,96], required = False,\
        help = "Filters of drug in CNN") 
    parser.add_argument("--cnn_drug_kernels", type = list, default = [4,6,8], required = False,\
        help = "Kernels of drug in CNN") 
    parser.add_argument("--cnn_target_filters", type = list, default = [32,64,96], required = False,\
        help = "Filters of target in CNN") 
    parser.add_argument("--cnn_target_kernels", type = list, default = [4,8,12], required = False,\
        help = "Kernels of target in CNN") 
    parser.add_argument("--rnn_Use_GRU_LSTM_drug", type = str, default = "GRU", required = False,\
        help = "Drug use GRU or LSTM in RNN") 
    parser.add_argument("--rnn_drug_hid_dim", type = int, default = 64, required = False,\
        help = "Dimensionality of hidden layers of drug in RNN")
    parser.add_argument("--rnn_drug_n_layers", type = int, default = 2, required = False,\
        help = "Number of layers of drug in RNN") 
    parser.add_argument("--rnn_drug_bidirectional", type = bool, default = True, required = False,\
        help = "Whether bidirectional of drug in RNN")
    parser.add_argument("--rnn_Use_GRU_LSTM_target", type = str, default = "GRU", required = False,\
        help = "Target use GRU or LSTM in RNN") 
    parser.add_argument("--rnn_target_hid_dim", type = int, default = 64, required = False,\
        help = "Dimensionality of hidden layers of target in RNN") 
    parser.add_argument("--rnn_target_n_layers", type = int, default = 2, required = False,\
        help = "Number of layers of target in RNN") 
    parser.add_argument("--rnn_target_bidirectional", type = bool, default = True, required = False,\
        help = "Whether bidirectional of target in RNN") 
    parser.add_argument("--num_workers", type = int, default = 0, required = False, help = "Number of workers") 
   
    return parser


def train_model(config):
    #加载数据
    X_drug, X_target, y = dataset.load_process(config.input_file)

    #分割训练集、验证集和测试集
    train, val, test = utils.data_process(X_drug, X_target, y, 
                                    config.drug_encoding, config.target_encoding, 
                                    split_method = 'random', frac = [0.7,0.1,0.2])

    #模型配置生成
    model_config = utils.generate_config(drug_encoding = config.drug_encoding, 
                    target_encoding = config.target_encoding,
                    result_folder = config.result_folder,
                    input_dim_drug = config.input_dim_drug, 
                    input_dim_protein = config.input_dim_protein,
                    hidden_dim_drug = config.hidden_dim_drug, 
                    hidden_dim_protein = config.hidden_dim_protein,
                    cls_hidden_dims = config.cls_hidden_dims,
                    mlp_hidden_dims_drug = config.mlp_hidden_dims_drug,
                    mlp_hidden_dims_target = config.mlp_hidden_dims_target,
                    batch_size = config.batch_size,
                    train_epoch = config.train_epoch,
                    test_every_X_epoch = config.test_every_X_epoch,
                    LR = config.LR,
                    decay = config.decay,
                    transformer_emb_size_drug = config.transformer_emb_size_drug,
                    transformer_intermediate_size_drug = config.transformer_intermediate_size_drug,
                    transformer_num_attention_heads_drug = config.transformer_num_attention_heads_drug,
                    transformer_n_layer_drug = config.transformer_n_layer_drug,
                    transformer_emb_size_target = config.transformer_emb_size_target,
                    transformer_intermediate_size_target = config.transformer_intermediate_size_target,
                    transformer_num_attention_heads_target = config.transformer_num_attention_heads_target,
                    transformer_n_layer_target = config.transformer_n_layer_target,
                    transformer_dropout_rate = config.transformer_dropout_rate,
                    transformer_attention_probs_dropout = config.transformer_attention_probs_dropout,
                    transformer_hidden_dropout_rate = config.transformer_hidden_dropout_rate,
                    mpnn_hidden_size = config.mpnn_hidden_size,
                    mpnn_depth = config.mpnn_depth,
                    cnn_drug_filters = config.cnn_drug_filters,
                    cnn_drug_kernels = config.cnn_drug_kernels,
                    cnn_target_filters = config.cnn_target_filters,
                    cnn_target_kernels = config.cnn_target_kernels,
                    rnn_Use_GRU_LSTM_drug = config.rnn_Use_GRU_LSTM_drug,
                    rnn_drug_hid_dim = config.rnn_drug_hid_dim,
                    rnn_drug_n_layers = config.rnn_drug_n_layers,
                    rnn_drug_bidirectional = config.rnn_drug_bidirectional,
                    rnn_Use_GRU_LSTM_target = config.rnn_Use_GRU_LSTM_target,
                    rnn_target_hid_dim = config.rnn_target_hid_dim,
                    rnn_target_n_layers = config.rnn_target_n_layers,
                    rnn_target_bidirectional = config.rnn_target_bidirectional,
                    num_workers = config.num_workers)

    #模型初始化
    model = DTI.model_initialize(**model_config)

    #训练模型
    model.train(train, val, test)

    #保存模型
    model.save_model(config.output_dir)


if __name__ == "__main__":
    parser = get_parser()
    config = parser.parse_known_args()[0]
    train_model(config)