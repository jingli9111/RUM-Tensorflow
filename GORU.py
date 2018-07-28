from tensorflow.python.ops.rnn_cell_impl import RNNCell
from EUNN import *

def modReLU(z, b):
	z_norm = math_ops.abs(z) + 0.00001
	step1 = nn_ops.bias_add(z_norm, b)
	step2 = nn_ops.relu(step1)
	step3 = math_ops.sign(z)		
	return math_ops.multiply(step3, step2)

class GORUCell(RNNCell):


	def __init__(self, hidden_size, capacity=2, FFT=False, activation=modReLU):
		
		self._hidden_size = hidden_size
		self._activation = activation
		self._capacity = capacity
		self._FFT = FFT

		self.v1, self.v2, self.ind, _, self._capacity = EUNN_param(hidden_size, capacity, FFT, False)



	@property
	def state_size(self):
		return self._hidden_size

	@property
	def output_size(self):
		return self._hidden_size

	@property
	def capacity(self):
		return self._capacity

	def __call__(self, inputs, state, scope=None):
		with vs.variable_scope(scope or "GORU_cell"):

			U_init = init_ops.random_uniform_initializer(-0.01, 0.01)
			b_init = init_ops.constant_initializer(2.)

			U = vs.get_variable("U", [inputs.get_shape()[-1], self._hidden_size * 3], dtype=tf.float32, initializer = U_init)
			Ux = math_ops.matmul(inputs, U)
			U_cx, U_rx, U_gx = array_ops.split(Ux, 3, axis=1)

			W_r = vs.get_variable("W_r", [self._hidden_size, self._hidden_size], dtype=tf.float32, initializer = U_init)
			W_g = vs.get_variable("W_g", [self._hidden_size, self._hidden_size], dtype=tf.float32, initializer = U_init)
			W_rh = math_ops.matmul(state, W_r)
			W_gh = math_ops.matmul(state, W_g)

			bias_r = vs.get_variable("bias_r", [self._hidden_size], dtype=tf.float32, initializer = b_init)
			bias_g = vs.get_variable("bias_g", [self._hidden_size], dtype=tf.float32)
			bias_c = vs.get_variable("bias_c", [self._hidden_size], dtype=tf.float32, initializer = b_init)
		

			r_tmp = U_rx + W_rh + bias_r
			g_tmp = U_gx + W_gh + bias_g
			r = math_ops.sigmoid(r_tmp)

			g = math_ops.sigmoid(g_tmp)

			# GRU with modReLU
			# W_h = vs.get_variable("W_h", [self._hidden_size, self._hidden_size], dtype=tf.float32, initializer = U_init)
			# W_hh = math_ops.matmul(state, W_h)
			# c = modReLU(math_ops.multiply(r, W_hh) + U_cx, bias_c)
			
			Unitaryh = EUNN_loop(state, self._capacity, self.v1, self.v2, self.ind, None)
			c = modReLU(math_ops.multiply(r, Unitaryh) + U_cx, bias_c)
			# c = tf.tanh(math_ops.multiply(r, Unitaryh) + U_cx + bias_c)
			new_state = math_ops.multiply(g, state) +  math_ops.multiply(1 - g, c)

		return new_state, new_state

