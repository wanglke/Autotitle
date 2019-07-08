import os

import pandas as pd
import numpy as np
import tensorflow as tf
from collections import Counter

import Summarizer
import summarizer_data_utils
import summarizer_model_utils

maxlen = 500
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# 数据处理  每个分类的前90%数据用来训练
data_path = "data/"
file_txt = []
for _, _, files in os.walk(data_path):
    for file in files:
        if os.path.splitext(file)[1] == '.txt':
            file_txt.append(file)

summaries_unprocessed_clean = []
texts_unprocessed = []
for file in file_txt:
    linecount = 0
    for index, line in enumerate(open(data_path+file, 'r', encoding="utf-8")):
        linecount += 1
    with open(data_path+file, 'r', encoding='utf-8') as lines:
        index, train_count = 0, linecount * 0.9
        for line in lines:
            if index > train_count:
                break
            stli = line.strip().split('\t')
            if len(stli) < 2:
                continue
            summaries_unprocessed_clean.append(stli[0])
            texts_unprocessed.append(stli[1][:maxlen])
            index += 1
    print('{0} has load..it has {1} news '.format(file, index))


# 清洗文字，有些新闻存在新闻机构后缀或不规则的符号
processed_texts, processed_summaries, words_counted = summarizer_data_utils.preprocess_texts_and_summaries(
    texts_unprocessed,
    summaries_unprocessed_clean,
    keep_most=False)

# 删除不规则数据，某一行不存在标题或内容
processed_texts_clean = []
processed_summaries_clean = []
for t, s in zip(processed_texts, processed_summaries):
    if t != [] and s != []:
        processed_texts_clean.append(t)
        processed_summaries_clean.append(s)

# 创建w2id 和id2w字典
specials = ["<EOS>", "<SOS>", "<PAD>", "<UNK>"]
word2ind, ind2word,  missing_words = summarizer_data_utils.create_word_inds_dicts(words_counted,
                                                                                  specials=specials,
                                                                                  min_occurences=2)


import gensim

model = gensim.models.KeyedVectors.load_word2vec_format('temdata/sgns.sogou.bigram')
vec=[]
for word in [key for key in word2ind.keys()]:
    if word in model.vocab:
        vec.append(model[word])
    else:
        vec.append(np.zeros(model.vector_size))
emb = tf.convert_to_tensor(np.asarray(vec), dtype =tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    embedding = sess.run(emb)


# import tensorflow_hub as hub
# embed = hub.Module("temdata/sgns.sogou.bigram")
# emb = embed([key for key in word2ind.keys()])
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     sess.run(tf.tables_initializer())
#     embedding = sess.run(emb)


print(embedding.shape)
np.save('temdata/sogou_news_embedding.npy', embedding)

converted_texts, unknown_words_in_texts = summarizer_data_utils.convert_to_inds(processed_texts_clean,
                                                                                word2ind)

converted_summaries, unknown_words_in_summaries = summarizer_data_utils.convert_to_inds(processed_summaries_clean,
                                                                                        word2ind,
                                                                                        eos=True,
                                                                                        sos=True)




print(summarizer_data_utils.convert_inds_to_text(converted_texts[0], ind2word))
print(summarizer_data_utils.convert_inds_to_text(converted_summaries[0], ind2word))







# model hyperparameters
num_layers_encoder = 4
num_layers_decoder = 4
rnn_size_encoder = 300
rnn_size_decoder = 300

batch_size = 32
epochs = 100
clip = 5
keep_probability = 0.8
learning_rate = 0.0005
max_lr = 0.005
learning_rate_decay_steps = 100
learning_rate_decay = 0.90


pretrained_embeddings_path = 'temdata/sogou_news_embedding.npy'
#summary_dir = os.path.join('./tensorboard/headlines')

use_cyclic_lr = True
inference_targets=True


# build graph and train the model
summarizer_model_utils.reset_graph()
summarizer = Summarizer.Summarizer(word2ind,
                                   ind2word,
                                   save_path='./models/headlines/my_model',
                                   mode='TRAIN',
                                   num_layers_encoder=num_layers_encoder,
                                   num_layers_decoder=num_layers_decoder,
                                   rnn_size_encoder=rnn_size_encoder,
                                   rnn_size_decoder=rnn_size_decoder,
                                   batch_size=batch_size,
                                   clip=clip,
                                   keep_probability=keep_probability,
                                   learning_rate=learning_rate,
                                   max_lr=max_lr,
                                   learning_rate_decay_steps=learning_rate_decay_steps,
                                   learning_rate_decay=learning_rate_decay,
                                   epochs=epochs,
                                   pretrained_embeddings_path=pretrained_embeddings_path,
                                   use_cyclic_lr=use_cyclic_lr,)
#                                    summary_dir = summary_dir)


summarizer.build_graph()
summarizer.train(converted_texts,
                 converted_summaries)




















# 预测
summarizer_model_utils.reset_graph()
summarizer = Summarizer.Summarizer(word2ind,
                                   ind2word,
                                   './models/headlines/my_model',
                                   'INFER',
                                   num_layers_encoder=num_layers_encoder,
                                   num_layers_decoder=num_layers_decoder,
                                   batch_size=len(converted_texts[:50]),
                                   clip=clip,
                                   keep_probability=1.0,
                                   learning_rate=0.0,
                                   beam_width=5,
                                   rnn_size_encoder=rnn_size_encoder,
                                   rnn_size_decoder=rnn_size_decoder,
                                   inference_targets=False,
                                   pretrained_embeddings_path=pretrained_embeddings_path)

summarizer.build_graph()
preds = summarizer.infer(converted_texts[:50],
                         restore_path='./models/headlines/my_model',
                         targets=converted_summaries[:50])


summarizer_model_utils.sample_results(preds,
                                      ind2word,
                                      word2ind,
                                      converted_summaries[:50],
                                      converted_texts[:50])



