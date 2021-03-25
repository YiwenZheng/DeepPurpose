#!/usr/bin/env python3

import utils, dataset, CompoundPred
from argparse import ArgumentParser
import pandas as pd

def read_data(input_file):
    df = pd.read_csv(input_file)
	y = df.iloc[:,1:2]
	X_drug = df.iloc[:,0:1]
    return X_drug, y


def train_model(config):
    #加载数据
    X_drug, y = read_data(config.input_file)

    #分割训练集、验证集和测试集
    train, val, test = utils.data_process(X_drug, y, config.drug_encoding,
                                split_method = 'random', frac = [0.7,0.1,0.2])

    #模型配置生成
    model_config = utils.generate_config(drug_encoding = config.drug_encoding, 
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
    model = CompoundPred.model_initialize(**model_config)

    #训练模型
    model.train(train, val, test)

    #保存模型
    model.save_model(config.output_dir)
