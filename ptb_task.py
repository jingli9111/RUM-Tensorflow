import time
import sys
import os
import random
from datetime import datetime

import tensorflow as tf

import auxiliary as aux
import reader
import ptb_configs as configs
from baselineModels import LNLSTM
from baselineModels import FSRNN
import RUM, GORU

from tensorflow.contrib.rnn import MultiRNNCell

random.seed(datetime.now())

flags = tf.flags

flags.DEFINE_string(
    "model", "ptb_rum",
    "A type of model. Check configs file to know which models are available.")
flags.DEFINE_string(
    "mode", "train",
    "A type of mode. Train or test.")
flags.DEFINE_string(
    "restore", "False",
    "Restore? True or False?")
flags.DEFINE_string("data_path", 'data/',
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", 'models/ptb/RUM_test/',
                    "Model output directory.")
FLAGS = flags.FLAGS

class PTBInput(object):
    """The input data."""

    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = reader.ptb_producer(
            data, batch_size, num_steps, name=name)


class PTBModel(object):
    """The PTB model."""

    def __init__(self, is_training, config, input_):
        self._input = input_

        batch_size = input_.batch_size
        num_steps = input_.num_steps
        emb_size = config.embed_size
        vocab_size = config.vocab_size
        F_size = config.cell_size
        if config.cell != "rum": 
            S_size = config.hyper_size

        emb_init = aux.orthogonal_initializer(1.0)
        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                "embedding", [vocab_size, emb_size], initializer=emb_init, dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

        if config.cell != "rum": 
            F_cells = [LNLSTM.LN_LSTMCell(F_size, use_zoneout=True, is_training=is_training,
                                          zoneout_keep_h=config.zoneout_h, zoneout_keep_c=config.zoneout_c)
                       for _ in range(config.fast_layers)]
        if config.cell == "fs-lstm": 
                S_cell  = LNLSTM.LN_LSTMCell(S_size, use_zoneout=True, is_training=is_training,
                                             zoneout_keep_h=config.zoneout_h, zoneout_keep_c=config.zoneout_c)
        elif config.cell == "fs-rum": 
            S_cell  = RUM.RUMCell(
                            hidden_size = S_size,
                            T_norm = config.T_norm,
                            use_zoneout = config.use_zoneout,
                            use_layer_norm = config.use_layer_norm,
                            is_training = is_training)        
        elif config.cell == "fs-goru": 
            with tf.variable_scope("goru"):
                S_cell  = GORU.GORUCell(hidden_size = S_size)    
        if config.cell != "rum": 
            FS_cell = FSRNN.FSRNNCell(F_cells, S_cell, config.keep_prob, is_training)
            self._initial_state = FS_cell.zero_state(batch_size, tf.float32)
            state = self._initial_state
            print FS_cell
        else: 
            def rum_cell(): 
                return RUM.RUMCell(
                                hidden_size = config.cell_size,
                                T_norm = config.T_norm,
                                use_zoneout = config.use_zoneout,
                                use_layer_norm = config.use_layer_norm,
                                is_training = is_training)
            mcell = MultiRNNCell([rum_cell() for _ in range(config.num_layers)], state_is_tuple = True)
            self._initial_state = mcell.zero_state(batch_size, tf.float32)
            state = self._initial_state
        print('generating graph')

        ## Dynamic RNN ##        
        # with tf.variable_scope("RNN"):
        # if config.cell != 'rum':
        #     outputs, _ = tf.nn.dynamic_rnn(F_cells[0], inputs, dtype=tf.float32)
        # else:
        #     outputs, _ = tf.nn.dynamic_rnn(mcell, inputs, dtype=tf.float32)


        ## For Loop RNN ##
        outputs = []
        for time_step in range(num_steps):
            if time_step > 0: tf.get_variable_scope().reuse_variables()
            if config.cell != "rum": 
                out, state = FS_cell(inputs[:, time_step, :], state)
            else: 
                out, state = mcell(inputs[:, time_step, :], state)
            outputs.append(out)
        outputs = tf.concat(axis=1, values=outputs)
        print('graph generated')
        outputs = tf.reshape(outputs, [-1, F_size])

        # Output layer and cross entropy loss

        out_init = aux.orthogonal_initializer(1.0)
        softmax_w = tf.get_variable(
            "softmax_w", [F_size, vocab_size], initializer=out_init, dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
        logits = tf.matmul(outputs, softmax_w) + softmax_b
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(input_.targets, [-1])],
            [tf.ones([batch_size * num_steps], dtype=tf.float32)])
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state

        if not is_training: return

        # Create the parameter update ops if training

        self._lr = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(cost, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N),
            config.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op


def run_epoch(session, model, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
        feed_dict = {}
        feed_dict[model.initial_state] = state

        vals = session.run(fetches, feed_dict)

        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += model.input.num_steps

        if verbose and step % (model.input.epoch_size // 10) == 10:
            print("%.3f BPC: %.3f speed: %.0f characters per second" %
                  (step * 1.0 / model.input.epoch_size, costs / (iters * 0.69314718056),
                   iters * model.input.batch_size / (time.time() - start_time)))

        sys.stdout.flush()

    return costs / (iters * 0.69314718056)

def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB data directory")

    config       = configs.get_config(FLAGS.model)
    eval_config  = configs.get_config(FLAGS.model)
    valid_config = configs.get_config(FLAGS.model)
    print(config.batch_size)
    eval_config.batch_size = 20
    valid_config.batch_size = 20
   
    raw_data = reader.ptb_raw_data(FLAGS.data_path + config.dataset + '/')
    train_data, valid_data, test_data, _ = raw_data

    if not os.path.exists(os.path.dirname(FLAGS.save_path)):
        try:
            os.makedirs(os.path.dirname(FLAGS.save_path))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)

        with tf.name_scope("Train"):
            train_input = PTBInput(config=config, data=train_data, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = PTBModel(is_training=True, config=config, input_=train_input)

        with tf.name_scope("Valid"):
            valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = PTBModel(is_training=False, config=config, input_=valid_input)

        with tf.name_scope("Test"):
            test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mtest = PTBModel(is_training=False, config=eval_config,
                                 input_=test_input)

        saver = tf.train.Saver(tf.trainable_variables())

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=session, coord=coord)
            if FLAGS.restore == "True":
                saver.restore(session, FLAGS.save_path + 'model.ckpt')
            if FLAGS.mode == "train":
                previous_val = 9999
                if FLAGS.restore == "True":
                    f = open(FLAGS.save_path + 'train-and-valid.txt', 'r')
                    x = f.readlines()[2]
                    x = x.rstrip()
                    x = x.split(" ")
                    previous_val = float(x[1])
                    print("previous validation is %f\n" % (previous_val))
                    f.close()
                for i in range(config.max_max_epoch):
                    lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                    m.assign_lr(session, config.learning_rate * lr_decay)

                    print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                    train_perplexity = run_epoch(session, m, eval_op=m.train_op, verbose=True)
                    print("Epoch: %d Train BPC: %.4f" % (i + 1, train_perplexity))
                    valid_perplexity = run_epoch(session, mvalid)
                    print("Epoch: %d Valid BPC: %.4f" % (i + 1, valid_perplexity))
                    sys.stdout.flush()

                    if i == 180: config.learning_rate *= 0.1

                    if valid_perplexity < previous_val:
                        print("Storing weights")
                        saver.save(session, FLAGS.save_path + 'model.ckpt')
                        f = open(FLAGS.save_path + 'train-and-valid.txt', 'w')
                        f.write("Epoch %d\nTrain %f\nValid %f\n" % (i, train_perplexity, valid_perplexity))
                        f.close()
                        previous_val = valid_perplexity
                        counter_val = 0
                    elif config.dataset == 'enwik8':
                        counter_val += 1
                        if counter_val == 2:
                            config.learning_rate *= 0.1
                            counter_val = 0

            print("Loading best weights")
            saver.restore(session, FLAGS.save_path + 'model.ckpt')
            test_perplexity = run_epoch(session, mtest)
            print("Test Perplexity: %.4f" % test_perplexity)
            f = open(FLAGS.save_path + 'test_2.txt', 'w')
            f.write("Test %f\n" % (test_perplexity))
            f.close()
            sys.stdout.flush()
            coord.request_stop()
            coord.join(threads)

if __name__ == "__main__":
    tf.app.run()