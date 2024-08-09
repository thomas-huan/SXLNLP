# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "../data/schema.json",
    "train_data_path": "../data/triplet_train.json", # 训练集
    "valid_data_path": "../data/valid.json", # 验证集
    "vocab_path": "../chars.txt", #中文字符集
    "max_length": 20, #最大长度
    "hidden_size": 128, #隐藏维度
    "epoch": 10,# 训练轮数
    "batch_size": 32, #每个batch的大小
    "epoch_data_size": 200,     #每轮训练中采样数量
    "positive_sample_rate":0.5,  #正样本比例
    "optimizer": "adam", #优化器
    "learning_rate": 1e-3,#学习率0.001
}