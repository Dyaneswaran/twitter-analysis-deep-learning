#!/usr/bin/env python
# coding: utf-8

# ### Model 1

# In[1]:


import pandas as pd

dataset = pd.read_csv("../training_data.csv",index_col = ['id'], usecols = ['id','tweet','subtask_a'])
dataset.columns = ['tweet', 'sentiment']
dataset.dropna(inplace = True)
dataset.reset_index(drop = True, inplace = True)
print(dataset.head(n = 5))
print(len(dataset.columns))
print(dataset.size)


# In[2]:


print(len(dataset.sentiment) == len(dataset.tweet))
print(dataset.sentiment.unique())

# Very Very disturbing error in the dataset. Thank god, I found it.
dataset = dataset.drop(dataset[(dataset.sentiment != 'OFF') & (dataset.sentiment != 'NOT')].index)

print(dataset.sentiment.unique())
print(dataset.sentiment.value_counts())


# In[4]:


import preprocess as pp


# In[5]:


clean_tweet = pp.clean_HTML(dataset.tweet)
clean_tweet = pp.clean_emoticons(clean_tweet)
clean_tweet = pp.clean_emojis(clean_tweet)
clean_tweet = pp.clean_tokens(clean_tweet)
clean_tweet = pp.clean_mentions(clean_tweet)
clean_tweet = pp.clean_hashtags(clean_tweet)
clean_tweet = pp.expand_contractions(clean_tweet)
clean_tweet = pp.lemmatize(clean_tweet)
clean_tweet = pp.remove_stop_words(clean_tweet)
clean_tweet = pp.clean_censored_words(clean_tweet)
clean_tweet = pp.remove_punctuators(clean_tweet)


# In[6]:


dataset['clean_tweet'] = clean_tweet


# In[7]:


import numpy as np

def encode_labels(labels):
    encoded_labels = []
    for label in labels:
        if label == 'OFF':
            encoded_labels.append(1)
        else:
            encoded_labels.append(0)   
    return np.asarray(encoded_labels)


# In[8]:


dataset['label'] = encode_labels(dataset.sentiment)


# In[13]:


X_train = dataset.clean_tweet
y_train = dataset.label

print(len(X[y == 0]), len(X[y == 1]))


# In[11]:


from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import TaggedDocument
import multiprocessing
from sklearn import utils


# In[12]:


def labelize_tweets_ug(tweets,label):
    result = []
    prefix = label
    for i, t in zip(tweets.index, tweets):
        result.append(TaggedDocument(t.split(), [prefix + '_%s' % i]))
    return result


# In[15]:


all_x = X_train
all_x_w2v = labelize_tweets_ug(all_x, 'all')


# In[16]:


cores = multiprocessing.cpu_count()
model_ug_cbow = Word2Vec(sg=0, size=100, negative=5, window=2, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
model_ug_cbow.build_vocab([x.words for x in tqdm(all_x_w2v)])


# In[17]:


for epoch in range(30):
    model_ug_cbow.train(utils.shuffle([x.words for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)
    model_ug_cbow.alpha -= 0.002
    model_ug_cbow.min_alpha = model_ug_cbow.alpha


# In[18]:


model_ug_sg = Word2Vec(sg=1, size=100, negative=5, window=2, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
model_ug_sg.build_vocab([x.words for x in tqdm(all_x_w2v)])


# In[19]:


get_ipython().run_cell_magic('time', '', 'for epoch in range(30):\n    model_ug_sg.train(utils.shuffle([x.words for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)\n    model_ug_sg.alpha -= 0.002\n    model_ug_sg.min_alpha = model_ug_sg.alpha')


# In[20]:


model_ug_cbow.save('w2v_model_ug_cbow.word2vec')
model_ug_sg.save('w2v_model_ug_sg.word2vec')


# In[21]:


from gensim.models import KeyedVectors
model_ug_cbow = KeyedVectors.load('w2v_model_ug_cbow.word2vec')
model_ug_sg = KeyedVectors.load('w2v_model_ug_sg.word2vec')


# In[22]:


embeddings_index = {}
for w in model_ug_cbow.wv.vocab.keys():
    embeddings_index[w] = np.append(model_ug_cbow.wv[w],model_ug_sg.wv[w])
print('Found %s word vectors.' % len(embeddings_index))


# In[24]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(X_train)
sequences = tokenizer.texts_to_sequences(X_train)


# In[26]:


max_len = max([len(x.split()) for x in X_train])
x_train_seq = pad_sequences(sequences, maxlen=max_len)


# In[69]:


num_words = 20000
embedding_dim = 200
embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in tokenizer.word_index.items():
    if i >= num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# In[32]:


from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Input, Dense, concatenate, Activation
from keras.models import Model


# In[33]:


tweet_input = Input(shape=(max_len,), dtype='int32')

tweet_encoder = Embedding(num_words, embedding_dim, weights=[embedding_matrix], input_length=max_len, trainable=True)(tweet_input)
bigram_branch = Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1)(tweet_encoder)
bigram_branch = GlobalMaxPooling1D()(bigram_branch)
trigram_branch = Conv1D(filters=100, kernel_size=3, padding='valid', activation='relu', strides=1)(tweet_encoder)
trigram_branch = GlobalMaxPooling1D()(trigram_branch)
fourgram_branch = Conv1D(filters=100, kernel_size=4, padding='valid', activation='relu', strides=1)(tweet_encoder)
fourgram_branch = GlobalMaxPooling1D()(fourgram_branch)
merged = concatenate([bigram_branch, trigram_branch, fourgram_branch], axis=1)

merged = Dense(256, activation='relu')(merged)
merged = Dropout(0.2)(merged)
merged = Dense(1)(merged)

output = Activation('sigmoid')(merged)

model = Model(inputs=[tweet_input], outputs=[output])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# In[34]:


from keras.callbacks import ModelCheckpoint

seed = 42
np.random.seed(seed)

filepath="CNN_best_weights.{epoch:02d}-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

model.fit(x_train_seq, y_train, batch_size=32, epochs=10, validation_split=0.2, callbacks = [checkpoint])


# In[35]:


from keras.models import load_model
loaded_CNN_model = load_model('CNN_best_weights.02-0.7609.hdf5')


# In[36]:


test_dataset = pd.read_csv("../test_data.csv",index_col = ['id'], usecols = ['id','tweet','subtask_a'])


# In[38]:


X_test = test_dataset.tweet
y_test = encode_labels(test_dataset.subtask_a)

sequences_test = tokenizer.texts_to_sequences(X_test)
X_test_seq = pad_sequences(sequences_test, maxlen=max_len)


# In[39]:


loaded_CNN_model.evaluate(x=X_test_seq, y=y_test)


# In[42]:


yhat_cnn = loaded_CNN_model.predict(X_test_seq)
print(yhat_cnn[:10] > 0.5)
print((y_test[:10]),(yhat_cnn[:10]))


# In[40]:


import itertools
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# In[43]:


from sklearn.metrics import confusion_matrix


y_pred = (yhat_cnn > 0.5)
cn_matrix = confusion_matrix(y_test, y_pred)
print(cn_matrix)

plot_confusion_matrix(cn_matrix, ['Not Offensive', 'Offensive'])


# In[44]:


from sklearn.metrics import f1_score

print(f1_score(y_true=y_test, y_pred=y_pred, average='macro'))


# ### Model 2

# In[45]:


import os

# I am trying Google's pre-trained word embeddings. Let's see how it performs. I hope it does well.
model_ggl_w2v = gensim.models.KeyedVectors.load_word2vec_format("../Datasets/GoogleNews-vectors-negative300.bin", binary=True)  


# In[47]:


ggl_embedding_index= {}

for w in model_ggl_w2v.vocab.keys():
    ggl_embedding_index[w] = model_ggl_w2v[w]

print("There are {number} words with that many vectors..".format(number=len(ggl_embedding_index)))


# In[48]:


ggl_num_words = 30000
ggl_embedding_dimension = 300
ggl_embedding_matrix = np.zeros((ggl_num_words, ggl_embedding_dimension))
for word, i in tokenizer.word_index.items():
    if i >= ggl_num_words:
        continue
    embedding_vector = ggl_embedding_index.get(word)
    if embedding_vector is not None:
        ggl_embedding_matrix[i] = embedding_vector


# In[49]:


tweet_input = Input(shape=(max_len,), dtype='int32')

tweet_encoder = Embedding(ggl_num_words, ggl_embedding_dimension, weights=[ggl_embedding_matrix], input_length=max_len, trainable=True)(tweet_input)
bigram_branch = Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1)(tweet_encoder)
bigram_branch = GlobalMaxPooling1D()(bigram_branch)
trigram_branch = Conv1D(filters=100, kernel_size=3, padding='valid', activation='relu', strides=1)(tweet_encoder)
trigram_branch = GlobalMaxPooling1D()(trigram_branch)
fourgram_branch = Conv1D(filters=100, kernel_size=4, padding='valid', activation='relu', strides=1)(tweet_encoder)
fourgram_branch = GlobalMaxPooling1D()(fourgram_branch)
merged = concatenate([bigram_branch, trigram_branch, fourgram_branch], axis=1)

merged = Dense(256, activation='relu')(merged)
merged = Dropout(0.2)(merged)
merged = Dense(1)(merged)
output = Activation('sigmoid')(merged)
model = Model(inputs=[tweet_input], outputs=[output])
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
model.summary()


# In[50]:


filepath="CNN_ggl_best_weights.{epoch:02d}-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')

model.fit(x_train_seq, y_train, batch_size=32, epochs=10, validation_split = 0.2, callbacks = [checkpoint])


# In[51]:


loaded_CNN_model = load_model('CNN_ggl_best_weights.01-0.7888.hdf5')


# In[52]:


loaded_CNN_model.evaluate(x=X_test_seq, y=y_test)


# In[54]:


yhat_cnn = loaded_CNN_model.predict(X_test_seq)

y_pred = (yhat_cnn > 0.5)
cn_matrix = confusion_matrix(y_test, y_pred)
print(cn_matrix)

plot_confusion_matrix(cn_matrix, ['Not Offensive', 'Offensive'])
print(f1_score(y_true=y_test, y_pred=y_pred, average='macro'))


# In[62]:


from keras.layers import LSTM, GRU, MaxPooling1D

tweet_input = Input(shape=(max_len,), dtype='int32')

tweet_encoder = Embedding(ggl_num_words, ggl_embedding_dimension, weights=[ggl_embedding_matrix], input_length=max_len, trainable=True)(tweet_input)
bigram_branch = Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1)(tweet_encoder)
bigram_branch = MaxPooling1D()(bigram_branch)
trigram_branch = Conv1D(filters=100, kernel_size=3, padding='valid', activation='relu', strides=1)(tweet_encoder)
trigram_branch = MaxPooling1D()(trigram_branch)
fourgram_branch = Conv1D(filters=100, kernel_size=4, padding='valid', activation='relu', strides=1)(tweet_encoder)
fourgram_branch = MaxPooling1D()(fourgram_branch)
merged = concatenate([bigram_branch, trigram_branch, fourgram_branch], axis=1)
lstm = LSTM(100)(merged)
lstm = Dropout(0.2)(lstm)
lstm = Dense(1)(lstm)
output = Activation('sigmoid')(lstm)
model = Model(inputs=[tweet_input], outputs=[output])
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
model.summary()


# In[63]:


filepath="CNN_LSTM_ggl_best_weights.{epoch:02d}-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')

model.fit(x_train_seq, y_train, batch_size=32, epochs=10, validation_split = 0.2, callbacks = [checkpoint])


# In[64]:


loaded_CNN_model = load_model('CNN_LSTM_ggl_best_weights.01-0.7839.hdf5')
loaded_CNN_model.evaluate(x=X_test_seq, y=y_test)


# In[65]:


yhat_cnn = loaded_CNN_model.predict(X_test_seq)

y_pred = (yhat_cnn > 0.5)
cn_matrix = confusion_matrix(y_test, y_pred)
print(cn_matrix)

plot_confusion_matrix(cn_matrix, ['Not Offensive', 'Offensive'])


print(f1_score(y_true=y_test, y_pred=y_pred, average='macro'))


# In[71]:


from keras.layers import LSTM, GRU, MaxPooling1D

tweet_input = Input(shape=(max_len,), dtype='int32')

tweet_encoder = Embedding(num_words, embedding_dim, weights=[embedding_matrix], input_length=max_len, trainable=True)(tweet_input)
bigram_branch = Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1)(tweet_encoder)
bigram_branch = MaxPooling1D()(bigram_branch)
trigram_branch = Conv1D(filters=100, kernel_size=3, padding='valid', activation='relu', strides=1)(tweet_encoder)
trigram_branch = MaxPooling1D()(trigram_branch)
fourgram_branch = Conv1D(filters=100, kernel_size=4, padding='valid', activation='relu', strides=1)(tweet_encoder)
fourgram_branch = MaxPooling1D()(fourgram_branch)
merged = concatenate([bigram_branch, trigram_branch, fourgram_branch], axis=1)
lstm = LSTM(100)(merged)
lstm = Dropout(0.2)(lstm)
lstm = Dense(1)(lstm)
output = Activation('sigmoid')(lstm)
model = Model(inputs=[tweet_input], outputs=[output])
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
model.summary()

filepath="CNN_LSTM_W2V_best_weights.{epoch:02d}-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')

model.fit(x_train_seq, y_train, batch_size=32, epochs=10, validation_split = 0.2, callbacks = [checkpoint])


# In[74]:


loaded_CNN_model = load_model('CNN_LSTM_W2V_best_weights.02-0.7594.hdf5')
loaded_CNN_model.evaluate(x=X_test_seq, y=y_test)


# In[73]:


yhat_cnn = loaded_CNN_model.predict(X_test_seq)

y_pred = (yhat_cnn > 0.5)
cn_matrix = confusion_matrix(y_test, y_pred)
print(cn_matrix)

plot_confusion_matrix(cn_matrix, ['Not Offensive', 'Offensive'])
print(f1_score(y_true=y_test, y_pred=y_pred, average='macro'))

