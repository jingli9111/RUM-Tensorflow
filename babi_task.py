from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse, os
import tensorflow as tf
import sys, tarfile, re

from tensorflow.contrib.rnn import BasicLSTMCell, BasicRNNCell, GRUCell

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops

from RUM import RUMCell, ARUMCell 
from baselineModels.GORU import GORUCell
from baselineModels.EUNN import EUNNCell

sigmoid = math_ops.sigmoid 
tanh = math_ops.tanh
matm = math_ops.matmul
mul = math_ops.multiply 
relu = nn_ops.relu


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.

    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format

    If only_supporting is true,
    only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file, retrieve the stories,
    and then convert the sentences into a single story.

    If max_length is supplied,
    any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]

    return data


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    xs = []
    qs = []
    ys = []

    x_len = []
    q_len = []

    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        q = [word_idx[w] for w in query]
        # let's not forget that index 0 is reserved
        # y = np.zeros(len(word_idx) + 1)
        # y[word_idx[answer]] = 1
        len_x = len(x)
        x_len.append(len_x)
        if len_x < story_maxlen:
        	x = [0] * (story_maxlen - len_x) + x

        # for i in range(len_x, story_maxlen):
        #     x = [0] + x
        len_q = len(q)
        q_len.append(len_q)
        for i in range(len_q, query_maxlen):
            q.append(0)       

        xs.append(x)
        qs.append(q)
        ys.append(word_idx[answer])

    return xs, qs, ys, x_len, q_len
    # return pad_sequences(xs, maxlen=story_maxlen), pad_sequences(xqs, maxlen=query_maxlen), np.array(ys)

# RNN = recurrent.LSTM
# SENT_HIDDEN_SIZE = 100
# QUERY_HIDDEN_SIZE = 100
# BATCH_SIZE = 32
# EPOCHS = 40
# print('RNN / Embed / Sent / Query = {}, {}, {}, {}'.format(RNN,
#                                             EMBED_HIDDEN_SIZE,
#                                             SENT_HIDDEN_SIZE,
#                                             QUERY_HIDDEN_SIZE))


def main(model, qid, n_iter, n_batch, n_hidden, n_embed, capacity, comp, FFT, learning_rate, norm, grid_name):

    learning_rate = float(learning_rate)

    path = './data/tasks_1-20_v1-2.tar.gz'
    tar = tarfile.open(path)

    name_str = [
                'single-supporting-fact',
                'two-supporting-facts',  
                'three-supporting-facts',
                'two-arg-relations',
                'three-arg-relations',
                'yes-no-questions',
                'counting',
                'lists-sets',
                'simple-negation',
                'indefinite-knowledge',
                'basic-coreference',
                'conjunction',
                'compound-coreference',
                'time-reasoning',
                'basic-deduction',
                'basic-induction',
                'positional-reasoning',
                'size-reasoning',
                'path-finding',
                'agents-motivations',
                ]

    challenge = 'tasks_1-20_v1-2/en-10k/qa' + str(qid) + '_' + name_str[qid-1] + '_{}.txt'


    train = get_stories(tar.extractfile(challenge.format('train')))
    test = get_stories(tar.extractfile(challenge.format('test')))

    vocab = set()
    for story, q, answer in train + test:
        vocab |= set(story + q + [answer])
    vocab = sorted(vocab)

    # Reserve 0 for masking via pad_sequences
    vocab_size = len(vocab) + 1
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    story_maxlen = max(map(len, (x for x, _, _ in train + test)))
    query_maxlen = max(map(len, (x for _, x, _ in train + test)))

    train_x, train_q, train_y, train_x_len, train_q_len = vectorize_stories(train, word_idx, story_maxlen, query_maxlen)
    test_x, test_q, test_y, test_x_len, test_q_len = vectorize_stories(test, word_idx, story_maxlen, query_maxlen)

    n_data = len(train_x)
    n_val = int(0.1 * n_data)
 
    val_x = train_x[-n_val:]
    val_q = train_q[-n_val:]
    val_y = train_y[-n_val:]
    val_x_len = train_x_len[-n_val:]
    val_q_len = train_q_len[-n_val:]
    train_x = train_x[:-n_val]
    train_q = train_q[:-n_val]
    train_y = train_y[:-n_val]
    train_q_len = train_q_len[:-n_val]
    train_x_len = train_x_len[:-n_val] 


    n_train = len(train_x)

    print('vocab = {}'.format(vocab))
    print('x.shape = {}'.format(np.array(train_x).shape))
    print('xq.shape = {}'.format(np.array(train_q).shape))
    print('y.shape = {}'.format(np.array(train_y).shape))
    print('story_maxlen, query_maxlen = {}, {}'.format(story_maxlen, query_maxlen))

    print('Build model...')

    # sentence = layers.Input(shape=(story_maxlen,), dtype='int32')
    sentence = tf.placeholder("int32", [None, story_maxlen])

    n_output = n_hidden
    n_input = n_embed
    n_classes = vocab_size

    embed_init_val = np.sqrt(6.)/np.sqrt(vocab_size)
    embed = tf.get_variable('Embedding', [vocab_size, n_embed] ,initializer = tf.random_normal_initializer(-embed_init_val, embed_init_val), dtype=tf.float32)
    encoded_sentence = tf.nn.embedding_lookup(embed, sentence)
    question = tf.placeholder("int32", [None, query_maxlen])
    encoded_question = tf.nn.embedding_lookup(embed, question)
    merged = tf.concat([encoded_sentence, encoded_question], axis=1)
    print(encoded_sentence, encoded_question, merged)

    input_data = merged

    if model == "LSTM":
        cell = BasicLSTMCell(n_hidden, state_is_tuple=True, forget_bias=1)
        hidden_out, _ = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
    elif model == "GRU":
        cell = GRUCell(n_hidden)
        hidden_out, _ = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
    elif model == "RUM":
        cell = RUMCell(n_hidden, T_norm = norm)
        hidden_out, _ = tf.nn.dynamic_rnn(cell, input_data, dtype = tf.float32)
    elif model == "ARUM":
        cell = ARUMCell(n_hidden, T_norm = norm)
        hidden_out, _ = tf.nn.dynamic_rnn(cell, input_data, dtype = tf.float32)
    elif model == "ARUM2":
        cell = ARUM2Cell(n_hidden, T_norm = norm)
        hidden_out, _ = tf.nn.dynamic_rnn(cell, input_data, dtype = tf.float32)
    elif model == "RNN":
        cell = BasicRNNCell(n_hidden)
        hidden_out, _ = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
    elif model == "EUNN":
        cell = EUNNCell(n_hidden, capacity, FFT, comp)
        hidden_out, _ = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
    elif model == "GORU":
        cell = GORUCell(n_hidden, capacity, FFT)
        hidden_out, _ = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)


    merged, _ = tf.nn.dynamic_rnn(cell, merged, dtype=tf.float32)



    # --- Hidden Layer to Output ----------------------
    V_init_val = np.sqrt(6.)/np.sqrt(n_output + n_input)

    V_weights = tf.get_variable("V_weights", shape = [n_hidden, n_classes], dtype=tf.float32, initializer=tf.random_uniform_initializer(-V_init_val, V_init_val))
    V_bias = tf.get_variable("V_bias", shape=[n_classes], dtype=tf.float32, initializer=tf.constant_initializer(0.01))

    merged_list = tf.unstack(merged, axis=1)[-1]
    temp_out = tf.matmul(merged_list, V_weights)
    final_out = tf.nn.bias_add(temp_out, V_bias) 


    answer_holder = tf.placeholder("int64", [None])


    # --- evaluate process ----------------------
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=final_out, labels=answer_holder))
    correct_pred = tf.equal(tf.argmax(final_out, 1), answer_holder)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


    # --- Initialization ----------------------
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()


   # --- save result ----------------------
    folder = "./output/babi/" + str(qid) + '/' + model  # + "_lambda=" + str(learning_rate) + "_beta=" + str(decay)
    filename = folder + "_h=" + str(n_hidden)
    filename = filename + "_lr=" + str(learning_rate)
    filename = filename + "_norm=" + str(norm)
    filename = filename + ".txt"
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    if not os.path.exists(os.path.dirname(folder + "/modelCheckpoint/")):
        try:
            print(folder + "/modelCheckpoint/")
            os.makedirs(os.path.dirname(folder + "/modelCheckpoint/"))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    f = open(filename, 'w')
    f.write("########\n\n")
    f.write("## \tModel: %s with N=%d"%(model, n_hidden))
    f.write("########\n\n")


    # --- Training Loop ----------------------
    saver = tf.train.Saver()

    step = 0
    with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=False)) as sess:

        sess.run(init)

        steps = []
        losses = []
        accs = []

        while step < n_iter:
            a = int(step % (n_train / n_batch))
            batch_x = train_x[a * n_batch : (a+1) * n_batch]
            batch_q = train_q[a * n_batch : (a+1) * n_batch]
            batch_y = train_y[a * n_batch : (a+1) * n_batch]


            train_dict = {sentence: batch_x, question: batch_q, answer_holder: batch_y}
            sess.run(optimizer, feed_dict=train_dict)
            acc = sess.run(accuracy, feed_dict=train_dict)
            loss = sess.run(cost, feed_dict=train_dict)

            print("Iter " + str(step) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))


            steps.append(step)
            losses.append(loss)
            accs.append(acc)
            step += 1



            if step % 200 == 1:
            
                saver.save(sess, folder + "/modelCheckpoint/step=" + str(step))

                val_dict = {sentence: val_x, question: val_q, answer_holder: val_y}
                val_acc = sess.run(accuracy, feed_dict=val_dict)
                val_loss = sess.run(cost, feed_dict=val_dict)

                print("Validation Loss= " + \
                      "{:.6f}".format(val_loss) + ", Validation Accuracy= " + \
                      "{:.5f}".format(val_acc))
                f.write("%d\t%f\t%f\n"%(step, val_loss, val_acc))

        print("Optimization Finished!")


        
        # --- test ----------------------
        test_dict = {sentence: test_x, question: test_q, answer_holder: test_y}
        test_acc = sess.run(accuracy, feed_dict=test_dict)
        test_loss = sess.run(cost, feed_dict=test_dict)
        f.write("Test result: Loss= " + "{:.6f}".format(test_loss) + \
                    ", Accuracy= " + "{:.5f}".format(test_acc))
        print("Test result: Loss= " + "{:.6f}".format(test_loss) + \
                    ", Accuracy= " + "{:.5f}".format(test_acc))



if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="bAbI Task")
    parser.add_argument("model", default='LSTM', help='Model name: LSTM, EUNN, GRU, GORU')
    parser.add_argument('qid', type=int, default=20, help='Test set')
    parser.add_argument('--n_iter', '-I', type=int, default=4000, help='training iteration number')
    parser.add_argument('--n_batch', '-B', type=int, default=32, help='batch size')
    parser.add_argument('--n_hidden', '-H', type=int, default=256, help='hidden layer size')
    parser.add_argument('--n_embed', '-E', type=int, default=64, help='embedding size')
    parser.add_argument('--capacity', '-L', type=int, default=2, help='Tunable style capacity, only for EUNN, default value is 2')
    parser.add_argument('--comp', '-C', type=str, default="False", help='Complex domain or Real domain. Default is False: real domain')
    parser.add_argument('--FFT', '-F', type=str, default="True", help='FFT style, default is False')
    parser.add_argument('--learning_rate', '-R', default=0.001, type=str)
    # parser.add_argument('--decay', '-D', default=0.9, type=str)
    parser.add_argument('--norm', '-norm', default=None, type=float)    
    parser.add_argument('--grid_name', '-GN', default = None, type = str, help = 'specify folder to save to')   

    args = parser.parse_args()
    dicts = vars(args)

    for i in dicts:
        if (dicts[i]=="False"):
            dicts[i] = False
        elif dicts[i]=="True":
            dicts[i] = True
        
    kwargs = {  
                'model': dicts['model'],
                'qid': dicts['qid'],
                'n_iter': dicts['n_iter'],
                'n_batch': dicts['n_batch'],
                'n_hidden': dicts['n_hidden'],
                'n_embed': dicts['n_embed'],
                'capacity': dicts['capacity'],
                'comp': dicts['comp'],
                'FFT': dicts['FFT'],
                'learning_rate': dicts['learning_rate'],
                # 'decay': dicts['decay'],
                'norm': dicts['norm'],
                'grid_name': dicts['grid_name']
            }

    main(**kwargs)
