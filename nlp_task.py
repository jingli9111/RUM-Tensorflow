from __future__ import absolute_import
from __future__ import division
from __future__	import print_function
import numpy as np
import argparse, os
import tensorflow as tf
import time
from tensorflow.python.ops import init_ops
from tempfile import TemporaryFile
from tensorflow.contrib.rnn import LSTMCell, BasicLSTMCell, BasicRNNCell, GRUCell, LSTMStateTuple, MultiRNNCell
import auxiliary as aux
from RUM import RUMCell
from FSRNN import FSRNNCell
from LNLSTM import LN_LSTMCell
from ptb_iterator import *
import re

def file_data(stage, 
	          n_batch, 
	          n_data, 
	          T, 
	          n_epochs, 
	          vocab_to_idx):
	if stage == 'train':
		file_name = 'data/enwik8/train'
	elif stage == 'valid':
		file_name = 'data/enwik8/valid'	
	elif stage == 'test':
		file_name = 'data/enwik8/test'
	with open(file_name, 'r' ) as f:
		raw_data = f.read()
		print("Data length: ", len(raw_data))
	raw_data = raw_data.replace('\n', '_')
	raw_data = raw_data.replace(' ', '')

	if vocab_to_idx == None:
		vocab = set(raw_data)
		vocab_size = len(vocab)
		print("Vocab size: ", vocab_size)
		my_dict = {}
		idx_to_vocab= {}
		vocab_to_idx = {}
		for index, item in enumerate(vocab):
			idx_to_vocab[index] = item
			vocab_to_idx[item] = index
	data = [vocab_to_idx[c] for c in raw_data][:n_data]
	print("Total data length: " , len(data)) 
	def gen_epochs(n, numsteps, n_batch):
		for i in range(n):
			yield ptb_iterator(data, n_batch, numsteps)
	print("Sequence length: ", T)
	myepochs = gen_epochs(n_epochs, T, n_batch)
	return myepochs, vocab_to_idx

def main(
	model, 
	T, 
	n_epochs, 
	n_batch, 
	n_hidden, 
	learning_rate, 
	decay, 
	nb_v, 
	norm,
	n_layers,
	clip_threshold,
	keep_prob,
	lr_decay,
	max_n_epoch,
	grid_name, 
	is_gates, 
	n_hyper_hidden,
	layer_norm,
	slow_size, 
	fast_size
	):
	max_len_data = 1000000000
	epoch_train, vocab_to_idx = file_data('train', n_batch, max_len_data, T, n_epochs, None)
	n_input = len(vocab_to_idx)
	epoch_val, _ = file_data('valid', nb_v, max_len_data, T, 10000, vocab_to_idx)
	epoch_test, _ = file_data('test', nb_v, max_len_data, T, 10000, vocab_to_idx)
	n_output = n_input

	x = tf.placeholder("int64", [None, T])
	y = tf.placeholder("int64", [None, T])
	new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
	lr = tf.get_variable("learning_rate", shape = [], dtype = tf.float32, trainable = False)
	update = tf.assign(lr, new_lr)

	if model == "LSTM":
		i_s = tuple([LSTMStateTuple(tf.placeholder("float", [None, n_hidden]), tf.placeholder("float", [None, n_hidden]))
			   for _ in range(n_layers)])
	elif model == "HyperDRUM": 
		i_s = tuple([LSTMStateTuple(tf.placeholder("float", [None, n_hyper_hidden]), tf.placeholder("float", [None, n_hidden + n_hyper_hidden]))
			   for _ in range(n_layers)])
	elif model == "FSRUM": 
		i_s = tuple([tuple([tuple([tf.placeholder("float", [None, fast_size]),
			tf.placeholder("float", [None, fast_size])]),
			tf.placeholder("float", [None, slow_size])]) for _ in range(n_layers)])

	else:
		i_s = tuple([tf.placeholder("float", [None, n_hidden]) for _ in range(n_layers)])	
	input_data = tf.one_hot(x, n_input, dtype=tf.float32)
	if keep_prob != None: 
		tf.nn.dropout(input_data, keep_prob)

	if model == "HyperDRUM":
		def hyperdrum_cell(): 
			return HyperDRUMCell(
						n_hidden, 
						hyper_num_units = n_hyper_hidden, 
						use_recurrent_dropout = layer_norm,
						normalization = norm) 
		mcell = MultiRNNCell([hyperdrum_cell() for _ in range(n_layers)], state_is_tuple = True)
	if model == "RUM":
		def rum_cell(): 
			return RUMCell(
				n_hidden, 
				T_norm = 1.0, 
				use_zoneout = True,
				use_layer_norm = True
				)
		mcell = MultiRNNCell([rum_cell() for _ in range(n_layers)], state_is_tuple = True)
	if model == "FSRUM":
		def rum_cell(): 
			return RUMCell(
				slow_size, 
				T_norm = 1.0, 
				use_zoneout = True,
				use_layer_norm = True
				)
		def ln_lstm_cell(): 
			return LN_LSTMCell(fast_size, use_zoneout=True, 
									is_training=True,
                                    zoneout_keep_h=True, 
                                    zoneout_keep_c=True)
		def fs_rum_cell(): 
			return FSRNNCell([ln_lstm_cell(), ln_lstm_cell()], rum_cell(), 0.65, training = True)
		mcell = MultiRNNCell([fs_rum_cell() for _ in range(n_layers)], state_is_tuple = True)
	if model == "LSTM":
		def lstm_cell():
			return LSTMCell(n_hidden, initializer = tf.orthogonal_initializer())
		mcell = MultiRNNCell([lstm_cell() for _ in range(n_layers)], state_is_tuple = True)
	if model == "GRU": 
		def gru_cell(): 
			return GRUCell(n_hidden, kernel_initializer = tf.orthogonal_initializer())
		mcell = MultiRNNCell([gru_cell() for _ in range(n_layers)], state_is_tuple = True)
	hidden_out, states = tf.nn.dynamic_rnn(mcell, input_data, dtype=tf.float32, initial_state = i_s)

	V_init_val = np.sqrt(6.) / np.sqrt(n_output + n_input)
	V_weights = tf.get_variable("V_weights", shape = [n_hidden, n_output], 
			                    dtype = tf.float32, 
			                    initializer = tf.orthogonal_initializer(gain = V_init_val))
	V_bias = tf.get_variable("V_bias", shape = [n_output],
			                 dtype = tf.float32, 
			                 initializer = tf.constant_initializer(0.01))
	hidden_out_list = tf.unstack(hidden_out, axis = 1)
	temp_out = tf.stack([tf.matmul(i, V_weights) for i in hidden_out_list])
	output_data = tf.nn.bias_add(tf.transpose(temp_out, [1,0,2]), V_bias) 
	if keep_prob != None: 
		tf.nn.dropout(output_data, keep_prob)


	cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = output_data, 
																		 labels = y))
	correct_pred = tf.equal(tf.argmax(output_data, 2), y)
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	optimizer = tf.train.AdamOptimizer(learning_rate = lr)
	
	train_op = optimizer.minimize(cost)
	
	init = tf.global_variables_initializer()
	for i in tf.global_variables():
		print(i.name)
	tmp_filename = "./output/character/"
	if grid_name is not None: 
		tmp_filename += grid_name + "/"
	tmp_filename += "T=" + str(T) + "/"
	filename = tmp_filename + str(n_layers) + model  +  "_N=" + str(n_hidden) + \
			   "_B=" + str(n_batch) + "_nb_v=" + str(nb_v) + \
			   "_numEpochs=" + str(n_epochs) + "_lr=" + str(learning_rate) 
	if norm is not None: 
		filename += "_norm=" + str(norm)
	if keep_prob is not None: 
		filename += "_keepProb=" + str(keep_prob)
	filename = filename + ".txt"
	research_filename = tmp_filename + "researchModels" + "/" + \
						str(n_layers) + model  +  "_N=" + str(n_hidden) + \
			   			"_B=" + str(n_batch) + "_nb_v=" + str(nb_v) + \
			   			"_numEpochs=" + str(n_epochs) + "_lr=" + str(learning_rate)
	if not os.path.exists(os.path.dirname(filename)):
		try:
			os.makedirs(os.path.dirname(filename))
		except OSError as exc:
			if exc.errno != errno.EEXIST:
				raise
	if not os.path.exists(os.path.dirname(research_filename)):
		try:
			os.makedirs(os.path.dirname(research_filename))
		except OSError as exc:
			if exc.errno != errno.EEXIST:
				raise
	if not os.path.exists(os.path.dirname(research_filename + "/modelCheckpoint/")):
		try:
			os.makedirs(os.path.dirname(research_filename + "/modelCheckpoint/"))
		except OSError as exc:
			if exc.errno != errno.EEXIST:
				raise
	f = open(filename, 'w')
	f.write("########\n\n")
	f.write("## \tModel: %s with N=%d" % (model, n_hidden))
	f.write("\n\n")
	f.write("########\n\n")

	def do_test(): 
		j = 0
		test_losses = []
		for test in epoch_test:
			j += 1 
			if j >= 2:
				break
			print("Running test...")
			if model == "LSTM":
				test_state = tuple([LSTMStateTuple(np.zeros((nb_v, n_hidden), dtype = np.float), 
											np.zeros((nb_v, n_hidden), dtype = np.float))
											for _ in range(n_layers)])
			elif model == "HyperDRUM":
				test_state = tuple([LSTMStateTuple(np.zeros((nb_v, n_hyper_hidden), dtype = np.float), 
											np.zeros((nb_v, n_hyper_hidden + n_hidden), dtype = np.float))
											for _ in range(n_layers)])
			elif model == "FSRUM":
				test_state = tuple([tuple([tuple([np.zeros((nb_v, fast_size), dtype = np.float),
					np.zeros((nb_v, fast_size), dtype = np.float)]),
					np.zeros((nb_v, slow_size), dtype = np.float)]) for _ in range(n_layers)])
			else:
				test_state = tuple([np.zeros((nb_v, n_hidden), dtype = np.float)
									for _ in range(n_layers)])
			for stepb, (X_test,Y_test) in enumerate(test):
				test_batch_x = X_test
				test_batch_y = Y_test
				test_dict = {x: test_batch_x, y: test_batch_y, i_s: test_state}
				test_acc, test_loss, test_state = sess.run([accuracy, cost,states],
					                                       feed_dict = test_dict)
				test_losses.append(test_loss)
		print("test:", )
		test_losses.append(sum(test_losses) / len(test_losses))
		print("test Loss= " +
				  "{:.6f}".format(test_losses[-1]))
		return test_losses[-1]

	def do_validation(loss, curr_epoch):
		curr_epoch = int(curr_epoch)
		j = 0
		val_losses = []
		val_max = 0
		val_norm_max = 0
		for val in epoch_val:
			j += 1 
			if j >= 2:
				break
			print("Running validation...")
			if model == "LSTM":
				val_state = tuple([LSTMStateTuple(np.zeros((nb_v, n_hidden), dtype = np.float), 
					               np.zeros((nb_v, n_hidden), dtype = np.float))
								   for _ in range(n_layers)])
			elif model == "HyperDRUM":
				val_state = tuple([LSTMStateTuple(np.zeros((nb_v, n_hyper_hidden), dtype = np.float), 
					               np.zeros((nb_v, n_hyper_hidden + n_hidden), dtype = np.float))
								   for _ in range(n_layers)])
			elif model == "FSRUM":
				val_state = tuple([tuple([tuple([np.zeros((nb_v, fast_size), dtype = np.float),
					np.zeros((nb_v, fast_size), dtype = np.float)]),
					np.zeros((nb_v, slow_size), dtype = np.float)]) for _ in range(n_layers)])
			else:
				val_state = tuple([np.zeros((nb_v, n_hidden), dtype = np.float)
								   for _ in range(n_layers)])
			for stepb, (X_val, Y_val) in enumerate(val):
				val_batch_x = X_val
				val_batch_y = Y_val
				val_dict = {x: val_batch_x, y: val_batch_y, i_s: val_state}				
				val_acc, val_loss, val_state = sess.run([accuracy, cost, states], feed_dict = val_dict)
				val_losses.append(val_loss)
		print("Validations:", )
		validation_losses.append(sum(val_losses) / len(val_losses))
		print("Validation Loss= " + "{:.6f}".format(validation_losses[-1]))
		test_loss = do_test ()
		lr = [v for v in tf.global_variables() if v.name == "learning_rate:0"][0]
		lr = sess.run(lr)
		f.write("Step: %d\t TrLoss: %f\t TestLoss: %f\t ValLoss: %f\t Epoch: %d\t Learning rate: %f\n" % (t, loss, test_loss, validation_losses[-1], curr_epoch, lr)) 
		f.flush()

	saver = tf.train.Saver()
	step = 0
	with tf.Session(config = tf.ConfigProto(log_device_placement = False, 
		                                    allow_soft_placement = False)) as sess:
		print("Session Created")
		steps = []
		losses = []
		accs = []
		validation_losses = []

		sess.run(init)
		if lr_decay == None: 
			sess.run(update, feed_dict={new_lr: learning_rate})

		if model == "LSTM":
			training_state = tuple([LSTMStateTuple(np.zeros((n_batch, n_hidden), dtype = np.float), 
							        np.zeros((n_batch, n_hidden), dtype = np.float)) 
								 	for _ in range(n_layers)])
		elif model == "HyperDRUM": 
			training_state = tuple([LSTMStateTuple(np.zeros((n_batch, n_hyper_hidden), dtype = np.float), 
							        np.zeros((n_batch, n_hyper_hidden + n_hidden), dtype = np.float)) 
								 	for _ in range(n_layers)])
		elif model == "FSRUM":
			training_state = tuple([tuple([tuple([np.zeros((n_batch, fast_size), dtype = np.float),
				np.zeros((n_batch, fast_size), dtype = np.float)]),
				np.zeros((n_batch, slow_size), dtype = np.float)]) for _ in range(n_layers)])
		else: 
			training_state = tuple([np.zeros((n_batch, n_hidden), dtype = np.float)
									for _ in range(n_layers)])
		i = 0
		t = 0
		val_cnt = 0 
		for epoch in epoch_train:
			print("Epoch: " , i)
			if lr_decay != None:
				sess.run(update, feed_dict={new_lr: learning_rate * (lr_decay ** max(i + 1 - max_n_epoch, 0.0))})

			for step, (X,Y) in enumerate(epoch):
				batch_x = X
				batch_y = Y
				myfeed_dict = {x: batch_x, y: batch_y, i_s: training_state}
				_, acc, loss, training_state = sess.run([train_op, accuracy, cost, states], 
														feed_dict = myfeed_dict)
				lr = [v for v in tf.global_variables() if v.name == "learning_rate:0"][0]
				lr = sess.run(lr)
				print("Iter " + str(t) + ", Minibatch Loss= " + 
					  "{:.6f}".format(loss) + ", Training Accuracy= " + 
				  	  "{:.5f}".format(acc) + ", Epoch " + str(i) + ", Learning rate= " + str(lr))
				steps.append(t)
				losses.append(loss)
				accs.append(acc)
				t += 1
				if step % 499 == 500:
					do_validation(loss, i) 
					if is_gates and (model == "GRU" or model == "DRUM") and (n_layers == 1): 
						if model == "GRU": tmp = "gru"
						if model == "DRUM": tmp = "drum"
						kernel = [v for v in tf.global_variables() if v.name == "rnn/multi_rnn_cell/cell_0/" + tmp + "_cell/gates/kernel:0"][0]
						bias = [v for v in tf.global_variables() if v.name == "rnn/multi_rnn_cell/cell_0/" + tmp + "_cell/gates/bias:0"][0]
						k, b = sess.run([kernel, bias])
						np.save(research_filename + "/kernel_" + str(val_cnt), k)
						np.save(research_filename + "/bias_" + str(val_cnt), b)
						val_cnt += 1 
			i += 1
			saver.save(sess, research_filename + "/modelCheckpoint/model")
		print("Optimization Finished!")

		test_loss = do_test()
		f.write("Test result: %d (step) \t%f (loss)\n" % (t, test_loss[-1]))

if __name__=="__main__":
	parser = argparse.ArgumentParser(
		description="NLP Character-level Task")
	parser.add_argument("model", default = 'LSTM', help = 'Model name: LSTM, EURNN, GRU, GORU')
	parser.add_argument('-T', type = int, default = 150, help = 'sequence length')
	parser.add_argument("--n_epochs", '-E', type = int, default = 50, help = 'num epochs')
	parser.add_argument('--n_batch', '-B', type = int, default = 128, help = 'batch size')
	parser.add_argument('--n_hidden', '-H', type = int, default = 730, help = 'hidden layer size')
	parser.add_argument('--slow_size', '-ss', type = int, default = 1500, help = 'slow cell size')
	parser.add_argument('--fast_size', '-fs', type = int, default = 730, help = 'fast cell size')
	parser.add_argument('--capacity', '-L', type = int, default = 2, help = 'Tunable style capacity, only for EURNN, default value is 2')
	parser.add_argument('--learning_rate', '-R', default = 0.001, type = float)
	parser.add_argument('--decay', '-D', default = 0.9, type = float)
	parser.add_argument('--nb_v', '-nbv', default = 32, type = int)
	parser.add_argument('--norm', '-norm', default = None, type = float)
	parser.add_argument('--n_layers', '-NL', default = 1, type = int, help = 'number of layers')
	parser.add_argument('--clip_threshold', '-CT', default = None, type = float, help = 'threshold of clipping')
	parser.add_argument('--keep_prob', '-KP', default = None, type = float, help = 'keep probability for dropout')
	parser.add_argument('--lr_decay', '-RD', default = None, type = float, help = 'learning rate decay for GradDescent')
	parser.add_argument('--max_n_epoch', '-ME', default = 12, type = int, help = 'maximum num. of epochs for reducing lr')
	parser.add_argument('--grid_name', '-GN', default = None, type = str, help = 'specify folder to save to')
	parser.add_argument('--is_gates', '-G', default = "False", type = str, help = 'ask to save gates (important for research)')
	parser.add_argument('--n_hyper_hidden', '-HH', type = int, default = 128, help = 'hidden layer size')
	parser.add_argument('--layer_norm', '-LN', type = str, default = "False", help = 'layer normalization?')

	args = parser.parse_args()
	dict = vars(args)
	for i in dict:
		if dict[i] == "False":
			dict[i] = False
		elif dict[i] == "True":
			dict[i] = True
	kwargs = {	
				'model': dict['model'],
				'T': dict['T'],
				'n_epochs': dict['n_epochs'],
			  	'n_batch': dict['n_batch'],
			  	'n_hidden': dict['n_hidden'],
			  	'learning_rate': dict['learning_rate'],
			  	'decay': dict['decay'],
				'nb_v': dict['nb_v'], 
				'norm': dict['norm'],
				'n_layers': dict['n_layers'],
				'clip_threshold': dict['clip_threshold'],
				'keep_prob': dict['keep_prob'],
				'lr_decay': dict['lr_decay'],
				'max_n_epoch': dict['max_n_epoch'],
				'grid_name': dict['grid_name'],
				'is_gates': dict['is_gates'], 
				'n_hyper_hidden': dict['n_hyper_hidden'],
				'layer_norm': dict['layer_norm'],
				'slow_size': dict['slow_size'],
				'fast_size': dict['fast_size']
			}
	print(kwargs)
	main(**kwargs)