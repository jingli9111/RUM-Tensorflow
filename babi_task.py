'''Trains two recurrent neural networks based upon a story and a question.
The resulting merged vector is then queried to answer a range of bAbI tasks.

The results are comparable to those for an LSTM model provided in Weston et al.:
"Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks"
http://arxiv.org/abs/1502.05698

Task Number      | FB LSTM Baseline | Keras QA
---        | ---        | ---
QA1 - Single Supporting Fact | 50            | 100.0
QA2 - Two Supporting Facts   | 20            | 50.0
QA3 - Three Supporting Facts | 20            | 20.5
QA4 - Two Arg. Relations     | 61            | 62.9
QA5 - Three Arg. Relations   | 70            | 61.9
QA6 - yes/No Questions     | 48            | 50.7
QA7 - Counting         | 49            | 78.9
QA8 - Lists/Sets             | 45            | 77.2
QA9 - Simple Negation      | 64            | 64.0
QA10 - Indefinite Knowledge  | 44            | 47.7
QA11 - Basic Coreference     | 72            | 74.9
QA12 - Conjunction       | 74            | 76.4
QA13 - Compound Coreference  | 94            | 94.4
QA14 - Time Reasoning      | 27            | 34.8
QA15 - Basic Deduction     | 21            | 32.4
QA16 - Basic Induction     | 23            | 50.6
QA17 - Positional Reasoning  | 51            | 49.1
QA18 - Size Reasoning      | 52            | 90.8
QA19 - Path Finding    | 8                | 9.0
QA20 - Agent's Motivations   | 91            | 90.7

For the resources related to the bAbI project, refer to:
https://research.facebook.com/researchers/1543934539189348

Notes:

- With default word, sentence, and query vector sizes, the GRU model achieves:
  - 100% test accuracy on QA1 in 20 epochs (2 seconds per epoch on CPU)
  - 50% test accuracy on QA2 in 20 epochs (16 seconds per epoch on CPU)
In comparison, the Facebook paper achieves 50% and 20% for the LSTM baseline.

- The task does not traditionally parse the question separately. This likely
improves accuracy and is a good example of merging two RNNs.

- The word vector embeddings are not shared between the story and question RNNs.

- See how the accuracy changes given 10,000 training samples (en-10k) instead
of only 1000. 1000 was used in order to be comparable to the original paper.

- Experiment with GRU, LSTM, and JZS1-3 as they give subtly different results.

- The length and noise (i.e. 'useless' story components) impact the ability for
LSTMs / GRUs to provide the correct answer. Given only the supporting facts,
these RNNs can achieve 100% accuracy on many tasks. Memory networks and neural
networks that use attentional processes can efficiently search through this
noise to find the relevant statements, improving performance substantially.
This becomes especially obvious on QA2 and QA3, both far longer than QA1.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse, os, re, tarfile
import tensorflow as tf
from functools import reduce

from tensorflow.contrib.rnn import BasicLSTMCell, BasicRNNCell, GRUCell
from EUNN import EUNNCell
from GORU import GORUCell
from RUM import RUMCell


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


def main(model, qid, n_iter, n_batch, n_hidden, n_embed, capacity, comp, FFT, norm, grid_name):



    path = './data/tasks_1-20_v1-2.tar.gz'
    tar = tarfile.open(path)


    # Default QA1 with 1000 samples
    # challenge = 'tasks_1-20_v1-2/en/qa1_single-supporting-fact_{}.txt'
    # QA1 with 10,000 samples
    # challenge = 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt'
    # QA2 with 1000 samples
    # challenge = 'tasks_1-20_v1-2/en/qa2_two-supporting-facts_{}.txt'
    # QA2 with 10,000 samples
    # challenge = 'tasks_1-20_v1-2/en-10k/qa' + str(qid) + '\*_{}.txt'

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

    # EMBED_HIDDEN_SIZE = 50
    n_output = n_hidden
    n_input = n_embed
    n_classes = vocab_size

    # encoded_sentence = layers.Embedding(vocab_size, EMBED_HIDDEN_SIZE)(sentence)
    embed_init_val = np.sqrt(6.)/np.sqrt(vocab_size)
    embed = tf.get_variable('Embedding', [vocab_size, n_embed] ,initializer = tf.random_normal_initializer(-embed_init_val, embed_init_val), dtype=tf.float32)
    
    encoded_sentence = tf.nn.embedding_lookup(embed, sentence)

    # encoded_sentence = layers.Dropout(0.3)(encoded_sentence)
    # encoded_sentence = tf.nn.dropout(encoded_sentence, drop)

    # question = layers.Input(shape=(query_maxlen,), dtype='int32')
    question = tf.placeholder("int32", [None, query_maxlen])
    # encoded_question = layers.Embedding(vocab_size, EMBED_HIDDEN_SIZE)(question)
    encoded_question = tf.nn.embedding_lookup(embed, question)
    # encoded_question = layers.Dropout(0.3)(encoded_question)
    # encoded_question = tf.nn.dropout(encoded_question, drop)
    # merged = layers.add([encoded_sentence, encoded_question])
    merged = tf.concat([encoded_sentence, encoded_question], axis=1)
    print(encoded_sentence, encoded_question, merged)

    with tf.variable_scope('m'):
    # merged = RNN(EMBED_HIDDEN_SIZE)(merged)
        if model == "LSTM":
            cell = BasicLSTMCell(n_hidden, state_is_tuple=True, forget_bias=1)
        elif model == "GRU":
            cell = GRUCell(n_hidden)
        elif model == "RUM":
            cell = RUMCell(n_hidden, T_norm = norm)
        elif model == "RNN":
            cell = BasicRNNCell(n_hidden)
        elif model == "EUNN":
            cell = EUNNCell(n_hidden, capacity, FFT, comp)
        elif model == "GORU":
            cell = GORUCell(n_hidden, capacity, FFT)

        merged, _ = tf.nn.dynamic_rnn(cell, merged, dtype=tf.float32)





    # merged = layers.Dropout(0.3)(merged)
    # merged = tf.nn.dropout(merged, drop)

    # --- Hidden Layer to Output ----------------------
    V_init_val = np.sqrt(6.)/np.sqrt(n_output + n_input)

    V_weights = tf.get_variable("V_weights", shape = [n_hidden, n_classes], dtype=tf.float32, initializer=tf.random_uniform_initializer(-V_init_val, V_init_val))
    V_bias = tf.get_variable("V_bias", shape=[n_classes], dtype=tf.float32, initializer=tf.constant_initializer(0.01))

    merged_list = tf.unstack(merged, axis=1)[-1]
    temp_out = tf.matmul(merged_list, V_weights)
    final_out = tf.nn.bias_add(temp_out, V_bias) 


    answer_holder = tf.placeholder("int64", [None])

    # print(final_out)
    # print(answer_holder)

    # --- evaluate process ----------------------
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=final_out, labels=answer_holder))
    correct_pred = tf.equal(tf.argmax(final_out, 1), answer_holder)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


    # --- Initialization ----------------------
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
    init = tf.global_variables_initializer()

    for i in tf.global_variables():
        print(i.name)
   # --- save result ----------------------
    filename = "./output/babi/" + str(qid) + "/" + model  # + "_lambda=" + str(learning_rate) + "_beta=" + str(decay)
        
    filename = filename + ".txt"
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    f = open(filename, 'w')
    f.write("########\n\n")
    f.write("## \tModel: %s with N=%d"%(model, n_hidden))
    f.write("########\n\n")


    # --- Training Loop ----------------------

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



            if step % 100 == 99:
            

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
    parser.add_argument('--n_iter', '-I', type=int, default=10000, help='training iteration number')
    parser.add_argument('--n_batch', '-B', type=int, default=32, help='batch size')
    parser.add_argument('--n_hidden', '-H', type=int, default=128, help='hidden layer size')
    parser.add_argument('--n_embed', '-E', type=int, default=64, help='embedding size')
    parser.add_argument('--capacity', '-L', type=int, default=2, help='Tunable style capacity, only for EUNN, default value is 2')
    parser.add_argument('--comp', '-C', type=str, default="False", help='Complex domain or Real domain. Default is False: real domain')
    parser.add_argument('--FFT', '-F', type=str, default="True", help='FFT style, default is False')
    # parser.add_argument('--learning_rate', '-R', default=0.001, type=str)
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
                # 'learning_rate': dicts['learning_rate'],
                # 'decay': dicts['decay'],
                'norm': dicts['norm'],
                'grid_name': dicts['grid_name']
            }

    main(**kwargs)
