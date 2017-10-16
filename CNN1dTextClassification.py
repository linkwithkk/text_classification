# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 03:05:19 2017
use 1d cnn for text classification using weight initialized by Facebook's fasttext. 
Usually i would freeze the weights in embedding layer but letting the model modify the weights is giving consistent accuracy improvements in test and validation across epochs. 

Reuters 21578 - R52 dataset starts overfitting after approximately 5-6 epochs with frozen weights.
prepare input csv data in the format of 'label','text' without the header

@author: linkw
"""

class CNN1dTextClassification(object):

    def __init__(self):
        pass

    #read the file into dataframe
    def read_file(self,inp_path,encoding_type='utf-8'):
        import pandas as pd
        dataframe=pd.read_csv(inp_path,header=None,encoding = encoding_type)
        Y=dataframe.loc[:,0]
        X=dataframe.loc[:,1]
        num_classes=len(Y.unique())
        return X,Y,num_classes
    
    #do nlp processing and save cleaned output to be used by fasttext
    def pre_processing(self,txt,language='english'):
        #remove unwanted characters
        from nltk.corpus import stopwords
        from nltk.stem.snowball import SnowballStemmer
        stopwords=stopwords.words(language)
        stemmer=SnowballStemmer(language)
        import re
        r=re.compile(r'[\W]', re.U)
        txt=txt.apply(lambda x : ' '.join(stemmer.stem(w.lower()) for w in re.sub('[\\s]+',' ',r.sub(' ',x)).split() if w not in stopwords))
        return txt
    
    #use fasttext to learn word embeddings
    def learn_embeddings(self,inp_path,out_path,emb_epoch=40,emb_lr=0.01,emb_dim=100,encoding_type='utf-8'):
        import fasttext
        fasttext.skipgram(inp_path,out_path,epoch=emb_epoch,lr=emb_lr,dim=emb_dim)
        from gensim.models.wrappers import FastText
        return FastText.load_fasttext_format(out_path+'.bin',encoding=encoding_type)
        
    #prepare all requirements of the neural net training
    def prepare(self,X,Y,emb_model,seq_length=200,stratify='n',test_split=0.2,emb_dim=100):
        #prepare data for use in NN
        #Convert text to sequences and create word index for use in creating embedding matrix
        from tensorflow.contrib.keras.api.keras.preprocessing.text import Tokenizer
        tokenizer=Tokenizer()
        tokenizer.fit_on_texts(X)
        X_seq = tokenizer.texts_to_sequences(X)
        word_idx=tokenizer.word_index
        from tensorflow.contrib.keras.api.keras.preprocessing import sequence
        X_seq=sequence.pad_sequences(X_seq,maxlen=seq_length)
        
        #encode labels in 1h vector
        from sklearn.preprocessing import LabelBinarizer
        label_encoder=LabelBinarizer()
        Y_coded=label_encoder.fit_transform(Y)
        
        #create test and train split
        from sklearn.model_selection import train_test_split
        if stratify=='y':
            x_train,x_test,y_train,y_test=train_test_split(X_seq,Y_coded,test_size=test_split,random_state=141289,stratify=Y_coded)
        else:
            x_train,x_test,y_train,y_test=train_test_split(X_seq,Y_coded,test_size=test_split,random_state=141289)
           
        #learn embedding matrix from the passed model
        import numpy as np
        embedding_mat= np.zeros((len(word_idx) + 1, emb_dim))
        for w, i in word_idx.items():
            try:
                embedding_vector=emb_model[w]
                embedding_mat[i]=embedding_vector
            except KeyError:
                    pass #print ("no "+ word+" pos" + str(i))        
        return x_train,x_test,y_train,y_test,embedding_mat,tokenizer,label_encoder
        
    #training method which trains the nueral network
    def train(self,x_train,x_test,y_train,y_test,embedding_matrix,num_classes,seq_length=200,emb_dim=100,train_emb=True,windows=(3,4,5,6),dropouts=(0.2,0.4),filter_sz=100,hid_dim=100,bch_siz=50,epoch=8):
        #setup and train the nueral net
        from tensorflow.contrib.keras.api.keras.models import Model
        from tensorflow.contrib.keras.api.keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Concatenate, Conv1D, MaxPool1D
        inp=Input(shape=(seq_length,))
        out=Embedding(input_dim=len(embedding_matrix[:,1]),output_dim=emb_dim, input_length=seq_length,weights=[embedding_matrix],trainable=train_emb)(inp)
        out=Dropout(dropouts[0])(out)
        convs=[]
        for w in windows:
            conv=Conv1D(filters=filter_sz,kernel_size=w, padding='valid',activation='relu',strides=1)(out)
            conv=MaxPool1D(pool_size=2)(conv)
            conv=Flatten()(conv)
            convs.append(conv)
        out=Concatenate()(convs)
        out=Dense(hid_dim,activation='relu')(out)
        out=Dropout(dropouts[1])(out)
        out=Activation('relu')(out)
        out=Dense(num_classes, activation='softmax')(out)
        model=Model(inp, out)
        model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
        model.fit(x_train, y_train, batch_size=bch_siz,
                  epochs=epoch, verbose=2, validation_data=(x_test, y_test))
        return model

    def train_with_embeddings(self,inp_path,encoding_type='utf-8',out_path='',emb_epoch=40,emb_lr=0.01,emb_dim=100,seq_length=200,stratify='n',test_split=0.2,language='english',train_emb=True,windows=(3,4,5,6),dropouts=(0.2,0.4),filter_sz=100,hid_dim=100,bch_siz=50,epoch=8):
        import time
        import pickle
        import os
        #note start time
        start=time.time()
        
        #check if input file exists
        print('Reading data',flush=True)
        if os.path.isfile(inp_path):
            try:
                #try to create output directory
                if out_path is not '':
                    out_path=out_path
                else:    
                    out_path=os.path.join(os.path.dirname(inp_path),'models','CNN1d',os.path.basename(inp_path))    
                os.makedirs(out_path,exist_ok=True)
                
            except Exception:
                print(Exception)
            if 1:
                #read file and get labels and descriptions
                X,Y,num_classes=self.read_file(inp_path,encoding_type)
                
                #clean descriptions for use
                print('Cleaning data',flush=True)
                X=self.pre_processing(X,language)
                
                #create cleaned text file
                refined_text=os.path.join(out_path,'text.txt')
                X.to_csv(refined_text,index=False)
                
                #create embedding model output
                emb_model_output=os.path.join(out_path,'ftmodel')
                print('Learning embeddings',flush=True)
                emb_model=self.learn_embeddings(refined_text,emb_model_output,emb_epoch,emb_lr,emb_dim,encoding_type)
                
                #create and save train/test split, embedding matrix etc
                print('Preparing data for use in NN',flush=True)
                x_train,x_test,y_train,y_test,embedding_matrix,tokenizer,label_encoder=self.prepare(X,Y,emb_model,seq_length,stratify,test_split,emb_dim)
                with open(os.path.join(out_path,'token_enc.pkl'), 'wb') as f:
                    pickle.dump([tokenizer, seq_length,language], f,protocol=pickle.HIGHEST_PROTOCOL)
                with open(os.path.join(out_path,'label_enc.pkl'), 'wb') as f:
                    pickle.dump(label_encoder, f,protocol=pickle.HIGHEST_PROTOCOL)
                
                #train the neural net
                print('Starting training',flush=True)
                model=self.train(x_train,x_test,y_train,y_test,embedding_matrix,num_classes,seq_length,emb_dim,train_emb,windows,dropouts,filter_sz,hid_dim,bch_siz,epoch)
                model.save(os.path.join(out_path,'CNN1d.h5'))
                
                #report total time
                print('Training finished. Total time = %s seconds' % (time.time() - start))
            else:
                pass
        else:
            print("Invalid input path")
        
#predict functionality       
def predict(out_path,txt,top=1):
    import pickle
    import os
    if os.path.isfile(os.path.join(out_path,'token_enc.pkl')):      
        with open(os.path.join(out_path,'token_enc.pkl'), 'rb') as f:
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
        model=load_model(os.path.join(out_path,'CNN1d.h5'))
        output=model.predict(txt_seq)
        
        #create binary sequences for top x predictions
        sorted_idx=(-output).argsort()
        import numpy as np
        label=np.zeros((top,len(output[0])))
        for i in range(0,top):
            label[i][sorted_idx[0][i]]=1
        
        #convert to txt labels
        with open(os.path.join(out_path,'label_enc.pkl'), 'rb') as f:
            label_decoder=pickle.load(f)
        return label_decoder.inverse_transform(label)
    else:
        return "Invalid output path!"        