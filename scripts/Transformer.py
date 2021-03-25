#!/usr/bin/env python3

from torch import nn
from model_helper import Embeddings, Encoder_MultipleLayers
from utils import drug2emb_encoder
from argparse import ArgumentParser
from encoders import Transformer
from train_CompoundPred import train_model



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_file", type = str, required = True,\
        help = "Path to data file")
    parser.add_argument("-o", "--output_dir", type = str, required = True,\
        help = "Directory where model will be saved")
    parser.add_argument("-d_e", "--drug_encoding", type = str, default = None,\
        required = True, help = "Drug encoding")

    # config of drug
    parser.add_argument("--input_dim_drug", type = int, default = 1024,\
        required = False, help = "Dimensionality of input of drug") 
    parser.add_argument("--transformer_emb_size_drug", type = int, default = 128,\
        required = False, help = "Size of embedding of drug in Transformer") 
    parser.add_argument("--transformer_n_layer_drug", type = int, default = 8,\
        required = False, help = "Number of layers of drug in Transfromer") 
    parser.add_argument("--transformer_intermediate_size_drug", type = int, default = 512,\
        required = False, help = "Size of intermediate layers of drug in Transformer") 
    parser.add_argument("--transformer_num_attention_heads_drug", type = int, default = 8,\
        required = False, help = "Number of heads of attention layers of drug in Transformer")
    parser.add_argument("--transformer_attention_probs_dropout", type = int, default = 0.1,\
        required = False, help = "Dropout probability of attention layers in Transformer") 
    parser.add_argument("--transformer_hidden_dropout_rate", type = int, default = 0.1,\
        required = False, help = "Dropout rate of hidden layers in Transformer")
    
    # config of target
    parser.add_argument("--input_dim_protein", type = int, default = 8420, required = False,\
        help = "Dimensionality of input of protein")
    parser.add_argument("--transformer_emb_size_target", type = int, default = 64, required = False,\
        help = "Size of embedding of target in Transformer") 
    parser.add_argument("--transformer_intermediate_size_target", type = int, default = 256,\
        required = False, help = "Size of intermediate layers of target in Transformer") 
    parser.add_argument("--transformer_num_attention_heads_target", type = int, default = 4,\
        required = False, help = "Number of heads of attention layers of target in Transformer") 
    parser.add_argument("--transformer_n_layer_target", type = int, default = 2, required = False,\
        help = "Number of layers of target in Transfromer") 

    config = parser.parse_known_args()[0]
    train_model(config)