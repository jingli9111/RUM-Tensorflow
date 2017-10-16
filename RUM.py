import tensorflow as tf
import numpy as np 
import auxiliary as aux

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.ops.rnn_cell_impl import RNNCell
from tensorflow.python.ops.rnn_cell_impl import _linear

sigmoid = math_ops.sigmoid 
tanh = math_ops.tanh
matm = math_ops.matmul
mul = math_ops.multiply 
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
	"""Rotational Unit of Memory"""

	def __init__(self,
				 hidden_size,
				 activation = None,
				 reuse = None,
				 kernel_initializer = None,
				 bias_initializer = None, 
				 T_norm = None, 
				 eps = 1e-12,
				 use_zoneout = False,
				 zoneout_keep_h = 0.9,
				 use_layer_norm = False,
				 is_training = False
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
		self._activation = activation or relu
		self._T_norm = T_norm 
		self._kernel_initializer = kernel_initializer or aux.orthogonal_initializer(1.0)
		self._bias_initializer = bias_initializer
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
			bias_ones = self._bias_initializer
			if self._bias_initializer is None:
				dtype = [a.dtype for a in [inputs, state]][0]
				bias_ones = init_ops.constant_initializer(1.0, dtype = dtype)
			value = _linear([inputs, state], 2 * self._hidden_size, True, bias_ones,
						aux.rum_ortho_initializer())
			r, u = array_ops.split(value = value, num_or_size_splits = 2, axis = 1)
			u = sigmoid(u)
			if self._use_layer_norm: 
				concat = tf.concat([r, u], 1)
				concat = aux.layer_norm_all(concat, 2, self._hidden_size, "LN_r_u")
				r, u = tf.split(concat, 2, 1)
		with vs.variable_scope("candidate"):
			x_emb = _linear(inputs, self._hidden_size, True, self._bias_initializer, 
							self._kernel_initializer)
			state_new = rotate(x_emb, r, state)
			if self._use_layer_norm: 
				c = self._activation(aux.layer_norm(x_emb + state_new, "LN_c"))
			else:
				c = self._activation(x_emb + state_new)
		new_h = u * state + (1 - u) * c
		if self._T_norm != None: 
			new_h = tf.nn.l2_normalize(new_h, 1, epsilon = self._eps) * self._T_norm
		if self._use_zoneout:
			new_h = aux.rum_zoneout(new_h, state, self._zoneout_keep_h, self._is_training) 
		return new_h, new_h

	def zero_state(self, batch_size, dtype):
		h = tf.zeros([batch_size, self._hidden_size], dtype=dtype)
		return h

class ARUMCell(RNNCell):
	"""Associative Rotational Unit of Memory"""

	def __init__(self,
				 hidden_size,
				 activation = None,
				 reuse = None,
				 kernel_initializer = None,
				 bias_initializer = None, 
				 T_norm = None, 
				 eps = 1e-12,
				 use_zoneout = False,
				 zoneout_keep_h = 0.9,
				 use_layer_norm = False,
				 is_training = False,
				 lambda_pow = 0
				 ):
		"""Initialization of the Associative RUM cell.

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
			lambda_pow: the power for the associative memory (an integer)
		"""
		super(ARUMCell, self).__init__(_reuse = reuse)
		self._hidden_size = hidden_size
		self._activation = activation or relu
		self._T_norm = T_norm 
		self._kernel_initializer = kernel_initializer or aux.orthogonal_initializer(1.0)
		self._bias_initializer = bias_initializer
		self._eps = eps 
		self._use_zoneout = use_zoneout 
		self._zoneout_keep_h = zoneout_keep_h
		self._use_layer_norm = use_layer_norm
		self._is_training = is_training 
		self._lambda_pow = lambda_pow

	@property
	def state_size(self):
		return self._hidden_size * (self._hidden_size + 1)
	
	@property
	def output_size(self):
		return self._hidden_size

	def call(self, inputs, state):
		#extract the associative memory and the state
		size_batch = tf.shape(state)[0]
		assoc_mem, state = tf.split(state, [self._hidden_size * self._hidden_size, self._hidden_size], 1)
		assoc_mem = tf.reshape(assoc_mem, [size_batch, self._hidden_size, self._hidden_size])

		with vs.variable_scope("gates"): 
			bias_ones = self._bias_initializer
			if self._bias_initializer is None:
				dtype = [a.dtype for a in [inputs, state]][0]
				bias_ones = init_ops.constant_initializer(1.0, dtype = dtype)
			value = _linear([inputs, state], 2 * self._hidden_size, True, bias_ones,
						aux.rum_ortho_initializer())
			r, u = array_ops.split(value = value, num_or_size_splits = 2, axis = 1)
			u = sigmoid(u)
			if self._use_layer_norm: 
				concat = tf.concat([r, u], 1)
				concat = aux.layer_norm_all(concat, 2, self._hidden_size, "LN_r_u")
				r, u = tf.split(concat, 2, 1)
		with vs.variable_scope("candidate"):
			x_emb = _linear(inputs, self._hidden_size, True, self._bias_initializer, 
							self._kernel_initializer)
			tmp_rotation = rotation_operator(x_emb, r)
			Rt = tf.matmul(assoc_mem, tmp_rotation)
			state_new = tf.reshape(tf.matmul(Rt, tf.reshape(state, [size_batch, self._hidden_size, 1])), [size_batch, self._hidden_size])

			if self._use_layer_norm: 
				c = self._activation(aux.layer_norm(x_emb + state_new, "LN_c"))
			else:
				c = self._activation(x_emb + state_new)
		new_h = u * state + (1 - u) * c
		if self._T_norm != None: 
			new_h = tf.nn.l2_normalize(new_h, 1, epsilon = self._eps) * self._T_norm
		if self._use_zoneout:
			new_h = aux.rum_zoneout(new_h, state, self._zoneout_keep_h, self._is_training) 

		Rt = tf.reshape(Rt, [size_batch, self._hidden_size * self._hidden_size])
		new_state = tf.concat([Rt, new_h], 1)
		return new_h, new_state

	def zero_state(self, batch_size, dtype):
		e = tf.eye(self._hidden_size, batch_shape = [batch_size])
		e = tf.reshape(e, [batch_size, self._hidden_size * self._hidden_size])
		c = tf.zeros([batch_size, self._hidden_size], dtype=dtype)
		h = tf.concat([e, c], 1)
		return h