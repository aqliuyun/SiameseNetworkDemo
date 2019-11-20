import numpy as np
import keras
import json
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dropout, Lambda, Bidirectional
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])         
train_path = os.path.join(cur, 'datas/train.txt')  
vocabulary_path = os.path.join(cur, 'models/vocabulary.txt')  
word2vec_path = os.path.join(cur, 'models/wiki.zh.wv.txt')
model_path = os.path.join(cur, 'models/siamese_network_model.h5')        
SEQ_LEN = 30
EMBEDDING_DIM = 300
EPOCHS = 20
BATCH_SIZE = 512  
    
def load_traindata(trainpath):
    sample_x = []
    sample_y = []
    sample_x_left = []
    sample_x_right = []
    for line in open(trainpath,encoding='UTF-8'):
        line = line.rstrip().split('\t')
        if not line:
            continue
        sent_left = line[0]
        sent_right = line[1]
        label = line[2]        
        sample_x_left.append([char for char in sent_left if char])
        sample_x_right.append([char for char in sent_right if char])
        sample_y.append(label)
    print(len(sample_x_left), len(sample_x_right))
    sample_x = [sample_x_left, sample_x_right]

    datas = [sample_x, sample_y]
    return datas

def reprocessing_data(traindatas):        
    vocabulary = load_vocabulary()
    sample_x = traindatas[0]
    sample_y = traindatas[1]
    sample_x_left = sample_x[0]
    sample_x_right = sample_x[1]    
    left_x_train = [[vocabulary[char] for char in data if char in vocabulary] for data in sample_x_left]
    right_x_train = [[vocabulary[char] for char in data if char in vocabulary] for data in sample_x_right]
    y_train = [int(i) for i in sample_y]
    left_x_train = pad_sequences(left_x_train, SEQ_LEN)
    right_x_train = pad_sequences(right_x_train, SEQ_LEN)
    y_train = np.expand_dims(y_train, 2)
    return left_x_train, right_x_train, y_train

def load_vocabulary():
    embeddings_dict = {}
    index = 0
    with open(word2vec_path, 'r',encoding='UTF-8') as f:
        for line in f:
            values = line.strip().split(' ')
            if len(values) < EMBEDDING_DIM:
                continue
            word = values[0]
            index+=1
            embeddings_dict[word] = index
    print('Found %s word vectors.' % len(embeddings_dict))
    return embeddings_dict

def save_vocabulary():
    vocabulary = []
    with open(word2vec_path, 'r',encoding='UTF-8') as f:
        for line in f:
            values = line.strip().split(' ')
            if len(values) < EMBEDDING_DIM:
                continue
            word = values[0]
            vocabulary.append(word)
    with open(vocabulary_path, 'w+',encoding='UTF-8') as f:
        f.write('\n'.join(vocabulary))

def build_embedding_matrix():    
    vocab_size = 0
    with open(word2vec_path, 'r',encoding='UTF-8') as f:
        for line in f:
            values = line.strip().split(' ')
            if len(values) < EMBEDDING_DIM:
                embedding_matrix = np.zeros((int(values[0]) + 1, EMBEDDING_DIM))
                continue
            vocab_size += 1
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_matrix[vocab_size] = coefs
    return embedding_matrix,vocab_size

def exponent_neg_manhattan_distance(sent_left, sent_right):
    return K.exp(-K.sum(K.abs(sent_left - sent_right), axis=1, keepdims=True))

def euclidean_distance(sent_left, sent_right):
    sum_square = K.sum(K.square(sent_left - sent_right), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def bilstm_network(input_shape):
    input = Input(shape=input_shape)
    lstm1 = Bidirectional(LSTM(128, return_sequences=True))(input)
    lstm1 = Dropout(0.5)(lstm1)
    lstm2 = Bidirectional(LSTM(32))(lstm1)
    lstm2 = Dropout(0.5)(lstm2)
    return Model(input, lstm2)

def siamese_network():
    embedding_matrix,vocab_size = build_embedding_matrix()
    embedding_layer = Embedding(vocab_size + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=SEQ_LEN,
                                trainable=False,
                                mask_zero=True)

    left_input = Input(shape=(SEQ_LEN,), dtype='float32')
    right_input = Input(shape=(SEQ_LEN,), dtype='float32')

    encoded_left = embedding_layer(left_input)
    encoded_right = embedding_layer(right_input)

    shared_lstm = bilstm_network(input_shape=(SEQ_LEN, EMBEDDING_DIM))
    left_output = shared_lstm(encoded_left)
    right_output = shared_lstm(encoded_right)

    distance = Lambda(lambda x: exponent_neg_manhattan_distance(x[0], x[1]),
                        output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

    model = Model([left_input, right_input], distance)    

    return model

def train_model(datas):      
    left_x_train, right_x_train, y_train = reprocessing_data(datas)        
    earlyStopping = keras.callbacks.EarlyStopping(monitor='accuracy', patience=8, verbose=0, mode='auto')
    saveCheckpoint = keras.callbacks.ModelCheckpoint(model_path, monitor='accuracy', verbose=1, save_best_only=True)
    model = siamese_network()
    model.compile(loss='binary_crossentropy',
                    optimizer='nadam',
                    metrics=['accuracy'])
    model.summary()
    #model.load_weights(model_path)
    history = model.fit(x=[left_x_train, right_x_train],
                            y=y_train,
                            validation_split=0.2,
                            batch_size=BATCH_SIZE,
                            epochs=EPOCHS,
							callbacks=[earlyStopping,saveCheckpoint])
    #model.save(model_path)
    return model

def predict(model_path,left,right):
    vocabulary = load_vocabulary()
    sample_x_left = [[char for char in data] for data in left]
    sample_x_right = [[char for char in data] for data in right]
    left_x_train = [[vocabulary[char] for char in data] for data in sample_x_left]
    right_x_train = [[vocabulary[char] for char in data] for data in sample_x_right]        
    left_x_train = pad_sequences(left_x_train, SEQ_LEN)
    right_x_train = pad_sequences(right_x_train, SEQ_LEN)        
    model = siamese_network()
    model.load_weights(model_path)
    result = model.predict([left_x_train, right_x_train])        
    print(result)

#save_vocabulary()
train_datas = load_traindata(train_path)
train_model(train_datas)

while True:
    print('start predict')
    text1 = input()
    text2 = input()
    predict(model_path,text1,text2)

