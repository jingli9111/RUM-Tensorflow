import tensorflow as tf
import numpy as np 
import auxiliary as aux 

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.ops.rnn_cell_impl import RNNCell

sigmoid = math_ops.sigmoid 
tanh = math_ops.tanh
relu = nn_ops.relu
sign = math_ops.sign

def rotation_operator(x, y, eps = 1e-12): 
	"""Rotation between two tensors: R(x,y) is unitary and takes x to y. 
	
	Args: 
		x: a tensor from where we want to start 
		y: a tensor at which we want to finish 
		eps: the cutoff for the normalizations (avoiding division by zero)
	Returns: 
		a tensor, which is the orthogonal rotation operator R(x,y)
	"""
	
	size_batch = tf.shape(x)[0]
	hidden_size = tf.shape(y)[1]

	#construct the 2x2 rotation
	u = tf.nn.l2_normalize(x, 1, epsilon = eps)
	costh = tf.reduce_sum(u * tf.nn.l2_normalize(y, 1, epsilon = eps), 1)
	sinth = tf.sqrt(1 - costh ** 2)
	step1 = tf.reshape(costh, [size_batch, 1])
	step2 = tf.reshape(sinth, [size_batch, 1])
	Rth = tf.reshape(tf.concat([step1, -step2, step2, step1], axis = 1), [size_batch, 2, 2])

	#get v and concatenate u and v 
	v = tf.nn.l2_normalize(y - tf.reshape(tf.reduce_sum(u * y, 1), [size_batch,1]) * u, 1, epsilon = eps)
	step3 = tf.concat([tf.reshape(u, [size_batch, 1, hidden_size]),
					  tf.reshape(v, [size_batch, 1, hidden_size])], 
					  axis = 1)
	
	#do the batch matmul 
	step4 = tf.reshape(u, [size_batch, hidden_size, 1])
	step5 = tf.reshape(v, [size_batch, hidden_size, 1])
	
	return (tf.eye(hidden_size, batch_shape = [size_batch]) - 
		   tf.matmul(step4, tf.transpose(step4, [0,2,1])) - 
		   tf.matmul(step5, tf.transpose(step5, [0,2,1])) + 
		   tf.matmul(tf.matmul(tf.transpose(step3, [0,2,1]), Rth), step3))

def rotation_components(x, y, eps = 1e-12): 
	"""Components for the operator R(x,y)
	   Together with `rotate` achieves best memory complexity: O(N_batch * N_hidden)

	Args: 
		x: a tensor from where we want to start 
		y: a tensor at which we want to finish 
		eps: the cutoff for the normalizations (avoiding division by zero)
	Returns: 
		Four components: u, v, [u,v] and R'(theta)
	"""
	
	size_batch = tf.shape(x)[0]
	hidden_size = tf.shape(x)[1]

	#construct the 2x2 rotation
	u = tf.nn.l2_normalize(x, 1, epsilon = eps)
	costh = tf.reduce_sum(u * tf.nn.l2_normalize(y, 1, epsilon = eps), 1)
	sinth = tf.sqrt(1 - costh ** 2)
	step1 = tf.reshape(costh, [size_batch, 1])
	step2 = tf.reshape(sinth, [size_batch, 1])
	Rth = tf.reshape(tf.concat([step1, -step2, step2, step1], axis = 1), [size_batch, 2, 2])

	#get v and concatenate u and v 
	v = tf.nn.l2_normalize(y - tf.reshape(tf.reduce_sum(u * y, 1), [size_batch,1]) * u, 1, epsilon = eps)
	step3 = tf.concat([tf.reshape(u, [size_batch, 1, hidden_size]),
					  tf.reshape(v, [size_batch, 1, hidden_size])], 
					  axis = 1)
	
	#do the batch matmul 
	step4 = tf.reshape(u, [size_batch, hidden_size, 1])
	step5 = tf.reshape(v, [size_batch, hidden_size, 1])
	return step4, step5, step3, Rth 

def rotate(v1, v2, v):
	"""Rotates v via the rotation R(v1,v2)

	Args: 
		v: a tensor, which is the vector we want to rotate
		== to define R(v1,v2) == 
		v1: a tensor from where we want to start 
		v2: a tensor at which we want to finish 
		
	Returns: 
		A tensor: the vector R(v1,v2)[v]
	"""
	size_batch = tf.shape(v1)[0]
	hidden_size = tf.shape(v1)[1]

	U  = rotation_components(v1, v2)
	h = tf.reshape(v, [size_batch, hidden_size, 1])

	return	(v + tf.reshape(	
							- tf.matmul(U[0], tf.matmul(tf.transpose(U[0], [0,2,1]), h))
							- tf.matmul(U[1], tf.matmul(tf.transpose(U[1], [0,2,1]), h)) 
							+ tf.matmul(tf.transpose(U[2], [0,2,1]), tf.matmul(U[3], tf.matmul(U[2], h))),
							[size_batch, hidden_size]
						))

class RUMCell(RNNCell):
	"""Rotational Unit of Memory (RUM)"""

	def __init__(self,
				 hidden_size,
				 activation_tmp = None,
				 activation_tar = None,
				 activation_emb = None, 
				 reuse = None,
				 bias_initializer = None, 
				 T_norm = None, 
				 eps = 1e-12,
				 use_zoneout = False,
				 zoneout_keep_h = 0.9,
				 use_layer_norm = False,
				 is_training = False,
				 ):
		"""Initialization of the RUM cell.

		Args: 
			hidden_size: number of neurons in hidden state 
			acitvation_tmp: activation of the temporary new state 
			activation_tar: activation of the target 
			activation_emb: activation of the embedded input 
			T_norm: norm for time normalization, `eta` in the paper 
			eps: the cutoff for the normalizations
			use_zoneout: zoneout, True or False 
			use_layer_norm: batch normalization, True or False
			is_training: marker for the zoneout 

		"""
		super(RUMCell, self).__init__(_reuse = reuse)
		self._hidden_size = hidden_size
		self._activation_tmp = activation_tmp or relu
		self._activation_tar = activation_tar or sigmoid
		self._activation_emb = activation_emb or tf.identity 
		self._T_norm = T_norm 
		self._bias_initializer = bias_initializer
		self._T_norm = T_norm
		self._eps = eps 
		self._use_zoneout = use_zoneout 
		self._zoneout_keep_h = zoneout_keep_h
		self._use_layer_norm = use_layer_norm
		self._is_training = is_training 

	@property
	def state_size(self):
		return self._hidden_size
	
	@property
	def output_size(self):
		return self._hidden_size

	def call(self, inputs, state):
		with vs.variable_scope("gates"): 
			x_size = inputs.get_shape().as_list()[1]
			batch_size = inputs.get_shape().as_list()[0]

			w_init = aux.orthogonal_initializer(1.0)
			h_init = aux.rum_ortho_initializer(1.0)
			b_init = tf.constant_initializer(1.0)

			W_xh = tf.get_variable("W_xh", [x_size, 2 * self._hidden_size], initializer = w_init)
			W_hh = tf.get_variable("W_hh", [self._hidden_size, 2 * self._hidden_size], initializer = h_init)
			bias = tf.get_variable("bias_ker", [2 * self._hidden_size], initializer = b_init)
			
			xh = tf.matmul(inputs, W_xh)
			hh = tf.matmul(state, W_hh)
			ux, rx = tf.split(xh, 2, 1)
			uh, rh = tf.split(hh, 2, 1)
			ub, rb = tf.split(bias, 2, 0)
			u = sigmoid(ux + uh + ub)
			r = self._activation_tar(rx + rh + rb)
			if self._use_layer_norm: 
				concat = tf.concat([u, r], 1)
				concat = aux.layer_norm_all(concat, 2, self._hidden_size, "ln_all")
				u, r = tf.split(concat, 2, 1)
		with vs.variable_scope("candidate"):
			W_xh_emb = tf.get_variable("W_xh_emb", [x_size, self._hidden_size], initializer = w_init)
			x_emb = self._activation_emb(tf.matmul(inputs, W_xh_emb))
			bias_emb = tf.get_variable("bias_emb", self._hidden_size, initializer = self._bias_initializer)
			state_new = rotate(x_emb + bias_emb, r, state)
			c = self._activation_tmp(aux.layer_norm(x_emb + state_new, "ln_c"))

		new_h = u * state + (1 - u) * c
		if self._T_norm != None: 
			new_h = tf.nn.l2_normalize(new_h, 1, epsilon = self._eps) * self._T_norm 
		if self._use_zoneout:
			new_h = aux.rum_zoneout(new_h, state, self._zoneout_keep_h, self._is_training)
		return new_h, new_h

	def zero_state(self, batch_size, dtype):
		h = tf.zeros([batch_size, self._hidden_size], dtype = dtype)
		return h
