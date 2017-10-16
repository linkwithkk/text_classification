# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 12:52:51 2017

Caller module for prediction
@author: linkw
"""

#prepare input csv data in the format of 'label','text' without header

#training
##for CNN1d
from CNN1dTextClassification import *
net=CNN1dTextClassification()
#net.train_with_embeddings(inp_path="D:\\Projects\\Python\\CNN\\r52.csv",out_path="D:\\tst",encoding_type='utf-8',emb_dim=100,seq_length=200,stratify='y',test_split=0.2,train_emb=True,filter_sz=100,hid_dim=100,bch_siz=50,epoch=6)
net.train_with_embeddings(inp_path="D:\\Projects\\Python\\CNN\\r52.csv",encoding_type='ISO-8859-1',emb_dim=100,seq_length=200,stratify='y',test_split=0.2,train_emb=True,filter_sz=100,hid_dim=100,bch_siz=50,epoch=6)

#prediction
##for CNN1d
from CNN1dTextClassification import predict
print(predict('D:\\Projects\\Python\\CNN\\models\\CNN1d\\r52.csv','Traders made windfall gains due to surging market',top=3))
