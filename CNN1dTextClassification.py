# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 03:05:19 2017
use 1d cnn for text classification using weight initialized by Facebook's fasttext. 
Usually i would freeze the weights in embedding layer but letting the model
modify the weights the model is giving consistent accuracy improvements in test and validation across epochs. 
To freeze embedding layer weights change line 47 trainable parameter from True to False and comment the dropout layer

Reuters 21578 - R52 dataset starts overfitting after approximately 5-6 epochs with frozen weights.
prepare input csv data in the format of 'label','text' without the header

@author: linkw
"""

class CNN1dTextClassification(object):

    def __init__(self,path,output='',emb_epoch=40,emb_lr=0.01,emb_dim=100,seq_length=60,split=0.2,language='english',windows=(3,4,5,6),dropouts=(0.2,0.3,0.5),filter_sz=100,hid_dim=100,bch_siz=50,epoch=8,stratify='n',encoding='utf-8'):
        self.emb_epoch=emb_epoch
        self.emb_lr=emb_lr
        self.emb_dim=emb_dim
        self.seq_length=seq_length
        self.language=language
        self.path=path
        import os
        if output is not '':
            self.dir=output
        else:    
            self.dir=os.path.join(os.path.dirname(path),'models','CNN1d',os.path.basename(path))
        os.makedirs(self.dir,exist_ok=True)
        self.test_split=split
        self.windows=windows
        self.dropouts=dropouts
        self.filter_sz=filter_sz
        self.hid_dim=hid_dim
        self.bch_siz=bch_siz
        self.epoch=epoch
        self.stratify=stratify
        self.encoding=encoding
        
    
    #training method which trains the nueral network
    def train(self,x_train,x_test,y_train,y_test,embedding_matrix,num_classes):
        #setup and train the nueral net
        from tensorflow.contrib.keras.api.keras.models import Model
        from tensorflow.contrib.keras.api.keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Concatenate, Conv1D, MaxPool1D
        inp=Input(shape=(self.seq_length,))
        out=Embedding(input_dim=len(embedding_matrix[:,1]),output_dim=self.emb_dim, input_length=self.seq_length,weights=[embedding_matrix],trainable=True)(inp)
        out=Dropout(self.dropouts[0])(out)
        convs=[]
        for w in self.windows:
            conv=Conv1D(filters=self.filter_sz,kernel_size=w, padding='valid',activation='relu',strides=1)(out)
            conv=MaxPool1D(pool_size=2)(conv)
            conv=Flatten()(conv)
            convs.append(conv)
        out=Concatenate()(convs)
        out=Dense(self.hid_dim,activation='relu')(out)
        out=Dropout(self.dropouts[1])(out)
        out=Activation('relu')(out)
        out=Dense(num_classes, activation='softmax')(out)
        model=Model(inp, out)
        model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
        model.fit(x_train, y_train, batch_size=self.bch_siz,
                  epochs=self.epoch, verbose=2, validation_data=(x_test, y_test))
        import os
        model.save(os.path.join(self.dir,'CNN1d.h5'))
        
    #prepare all requirements of the neural net training
    def prepare(self,X,Y,emb_model):
        import os
        import pickle
        vocab=set(w for x in X for w in x.split())        

        #prepare data for use in NN
        #Convert text to sequences and create word index for use in creating embedding matrix
        from tensorflow.contrib.keras.api.keras.preprocessing.text import Tokenizer
        tokenizer=Tokenizer()
        tokenizer.fit_on_texts(X)
        with open(os.path.join(self.dir,'token_enc.pkl'), 'wb') as f:
            pickle.dump([tokenizer, self.seq_length,self.language], f,protocol=pickle.HIGHEST_PROTOCOL)
        X_seq = tokenizer.texts_to_sequences(X)
        word_idx=tokenizer.word_index
        from tensorflow.contrib.keras.api.keras.preprocessing import sequence
        X_seq=sequence.pad_sequences(X_seq,maxlen=self.seq_length)
        
        #encode labels in 1h vector
        from sklearn.preprocessing import LabelBinarizer
        label_encoder=LabelBinarizer()
        Y_coded=label_encoder.fit_transform(Y)
        with open(os.path.join(self.dir,'label_enc.pkl'), 'wb') as f:
            pickle.dump(label_encoder, f,protocol=pickle.HIGHEST_PROTOCOL)

        #create test and train split
        from sklearn.model_selection import train_test_split
        if self.stratify=='y':
            x_train,x_test,y_train,y_test=train_test_split(X_seq,Y_coded,test_size=self.test_split,random_state=141289,stratify=Y_coded)
        else:
            x_train,x_test,y_train,y_test=train_test_split(X_seq,Y_coded,test_size=self.test_split,random_state=141289)
           
        #learn embedding matrix from the passed model
        import numpy as np
        embedding_mat= np.zeros((len(word_idx) + 1, self.emb_dim))
        for w, i in word_idx.items():
            try:
                embedding_vector=emb_model[w]
                embedding_mat[i]=embedding_vector
            except KeyError:
                    pass #print ("no "+ word+" pos" + str(i))        
        return x_train,x_test,y_train,y_test,embedding_mat
    
    def read_file(self):
        #read the file into dataframe
        import pandas as pd
        dataframe=pd.read_csv(self.path,header=None,encoding = self.encoding)
        Y=dataframe.loc[:,0]
        X=dataframe.loc[:,1]
        num_classes=len(Y.unique())
        return X,Y,num_classes

    #do nlp processing and save cleaned output to be used by fasttext
    def pre_processing(self,txt):
        #remove unwanted characters
        from nltk.corpus import stopwords
        from nltk.stem.snowball import SnowballStemmer
        stopwords=stopwords.words(self.language)
        stemmer=SnowballStemmer(self.language)
        import re
        r=re.compile(r'[\W]', re.U)
        txt=txt.apply(lambda x : ' '.join(stemmer.stem(w.lower()) for w in re.sub('[\\s]+',' ',r.sub(' ',x)).split() if w not in stopwords))
        import os
        txt.to_csv(os.path.join(self.dir,'text.txt'),index=False)
        return txt
        
    #use fasttext to learn word embeddings
    def learn_embeddings(self):
        import fasttext
        import os
        fasttext.skipgram(os.path.join(self.dir,'text.txt'), os.path.join(self.dir,'ftmodel'),epoch=self.emb_epoch,lr=self.emb_lr,dim=self.emb_dim)
        from gensim.models.wrappers import FastText
        return FastText.load_fasttext_format(os.path.join(self.dir,'ftmodel.bin'))
     
    #predict functionality       
    def predict(self,path_out,txt,top=1):
        import pickle
        import os
        with open(os.path.join(path_out,'token_enc.pkl'), 'rb') as f:
            tokenizer,seq_len,language=pickle.load(f)
        
        #do preprocessing bit
        from nltk.corpus import stopwords
        from nltk.stem.snowball import SnowballStemmer
        stopwords=stopwords.words(language)
        stemmer=SnowballStemmer(language)
        import re
        r = re.compile(r'[\W]', re.U)
        txt=r.sub(' ',txt)
        txt=re.sub('[\\s]+',' ',txt)  
        txt=[' '.join(stemmer.stem(w.lower()) for w in txt.split() if w not in stopwords)]
        
        #convert text to sequence 
        txt_seq=tokenizer.texts_to_sequences(txt)
        from tensorflow.contrib.keras.api.keras.preprocessing import sequence
        txt_seq=sequence.pad_sequences(txt_seq,maxlen=seq_len)
        
        #load NN model and predict
        from tensorflow.contrib.keras.api.keras.models import load_model
        model=load_model(os.path.join(path_out,'CNN1d.h5'))
        output=model.predict(txt_seq)
        
        #create binary sequences for top x predictions
        sorted_idx=(-output).argsort()
        import numpy as np
        label=np.zeros((top,len(output[0])))
        for i in range(0,top):
            label[i][sorted_idx[0][i]]=1
        
        #convert to txt labels
        with open(os.path.join(path_out,'label_enc.pkl'), 'rb') as f:
            label_decoder=pickle.load(f)
        return label_decoder.inverse_transform(label)
    
    def train_with_embeddings(self):
        import time
        start=time.time()
        print('Reading data',flush=True)
        X,Y,num_classes=self.read_file()
        print('Cleaning data',flush=True)
        X=self.pre_processing(X)
        print('Learning embeddings',flush=True)
        emb_model=self.learn_embeddings()
        print('Preparing data for use in NN',flush=True)
        x_train,x_test,y_train,y_test,embedding_matrix=self.prepare(X,Y,emb_model)
        print('Starting training',flush=True)
        self.train(x_train,x_test,y_train,y_test,embedding_matrix,num_classes)
        print('Training finished. Total time = %s seconds' % (time.time() - start))