# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 12:52:51 2017

Caller module for prediction
@author: linkw
"""

#prepare input csv data in the format of 'label','text' without header

#for CNN1d
from CNN1dTextClassification import *

##For training including word embedding training
#net=CNN1dTextClassification("D:\\Projects\\Python\\CNN\\r52.csv",output='D:\\Tst',seq_length=200,stratify='y',epoch=3,emb_dim=50,emb_epoch=70,split=0.3,bch_siz=12,encoding='ISO-8859-1')
net=CNN1dTextClassification("D:\\Projects\\Python\\CNN\\r52.csv",output='D:\\Tst',seq_length=200,stratify='n',epoch=6,emb_dim=100,emb_epoch=35,split=0.2,bch_siz=50,encoding='utf-8')
net.train_with_embeddings()

##for training by loading a pre trained fast text word embedding file ftmodel.bin in output folder
#net=CNN1dTextClassification("D:\\Projects\\Python\\CNN\\r52.csv",output='D:\\Tst',seq_length=200,stratify='y',epoch=5,emb_dim=100,emb_epoch=40,split=0.2,bch_siz=50,encoding='utf-8')
#X,Y,num_classes=net.read_file()
#X=net.pre_processing(X)
#from gensim.models.wrappers import FastText
#import os
#emb_model=FastText.load_fasttext_format(os.path.join(net.dir,'ftmodel.bin'))
#x_train,x_test,y_train,y_test,embedding_matrix=net.prepare(X,Y,emb_model)
#net.train(x_train,x_test,y_train,y_test,embedding_matrix,num_classes)

#for prediction
top_n_classes=net.predict('D:\\Tst','oil money',top=3)
print(top_n_classes)

