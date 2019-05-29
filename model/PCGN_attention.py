# -*- coding: utf-8 -*-
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import collections
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import AttentionMechanism, _compute_attention
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.util import nest

_zero_state_tensors = rnn_cell_impl._zero_state_tensors


class PCGNWrapperState(collections.namedtuple("PCGNWrapperState",
	                ("cell_state", "attention", "user_feat_mem", "time", "alignments","alignment_history"))):
	"""`namedtuple` storing the state of a `PCGNWrapper`.
	Contains:
	  - `cell_state`: The state of the wrapped `RNNCell` at the previous time
		step.
	  - `attention`: The attention emitted at the previous time step.
	  - `time`: int32 scalar containing the current time step.
	  - `alignments`: A single or tuple of `Tensor`(s) containing the alignments
		 emitted at the previous time step for each attention mechanism.
	  - `alignment_history`: (if enabled) a single or tuple of `TensorArray`(s)
		 containing alignment matrices from all time steps for each attention
		 mechanism. Call `stack()` on each to convert to a `Tensor`.
	"""

	def clone(self, **kwargs):
		"""Clone this object, overriding components provided by kwargs.
		Example:
		```python
		initial_state = attention_wrapper.zero_state(dtype=..., batch_size=...)
		initial_state = initial_state.clone(cell_state=encoder_state)
		```
		Args:
		  **kwargs: Any properties of the state object to replace in the returned
			`PCGNWrapperState`.
		Returns:
		  A new `PCGNWrapperState` whose properties are the same as
		  this one, except any overridden properties as provided in `kwargs`.
		"""
		return super(PCGNWrapperState, self)._replace(**kwargs)


class PCGNWrapper(rnn_cell_impl.RNNCell):
	"""Wraps another `RNNCell` with attention.
	"""

	def __init__(self,
	             cell,
	             attention_mechanism,
	             user_feats,
	             user_embs,
	             user_feat_mem_units,  # memory size
	             user_feat_mem_embedding,  # user_feat_mem_embedding
	             read_gate=None,
	             write_gate=None,
	             use_gate_memory=False,
	             attention_layer_size=None,
	             read_atten_gate=None,
	             alignment_history=False,
	             cell_input_fn=None,
	             output_attention=True,
	             initial_cell_state=None,
	             name=None):
		"""Construct the `PCGNWrapper`.
		**NOTE** If you are using the `BeamSearchDecoder` with a cell wrapped in
		`PCGNWrapper`, then you must ensure that:
		- The encoder output has been tiled to `beam_width` via
		  @{tf.contrib.seq2seq.tile_batch} (NOT `tf.tile`).
		- The `batch_size` argument passed to the `zero_state` method of this
		  wrapper is equal to `true_batch_size * beam_width`.
		- The initial state created with `zero_state` above contains a
		  `cell_state` value containing properly tiled final state from the
		  encoder.
		An example:
		```
		tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(
			encoder_outputs, multiplier=beam_width)
		tiled_encoder_final_state = tf.conrib.seq2seq.tile_batch(
			encoder_final_state, multiplier=beam_width)
		tiled_sequence_length = tf.contrib.seq2seq.tile_batch(
			sequence_length, multiplier=beam_width)
		attention_mechanism = MyFavoriteAttentionMechanism(
			num_units=attention_depth,
			memory=tiled_inputs,
			memory_sequence_length=tiled_sequence_length)
		attention_cell = PCGNWrapper(cell, attention_mechanism, ...)
		decoder_initial_state = attention_cell.zero_state(
			dtype, batch_size=true_batch_size * beam_width)
		decoder_initial_state = decoder_initial_state.clone(
			cell_state=tiled_encoder_final_state)
		```
		Args:
		  cell: An instance of `RNNCell`.
		  attention_mechanism: A list of `AttentionMechanism` instances or a single
			instance.
		  attention_layer_size: A list of Python integers or a single Python
			integer, the depth of the attention (output) layer(s). If None
			(default), use the context as attention at each time step. Otherwise,
			feed the context and cell output into the attention layer to generate
			attention at each time step. If attention_mechanism is a list,
			attention_layer_size must be a list of the same length.
		  alignment_history: Python boolean, whether to store alignment history
			from all time steps in the final output state (currently stored as a
			time major `TensorArray` on which you must call `stack()`).
		  cell_input_fn: (optional) A `callable`.  The default is:
			`lambda inputs, attention: array_ops.concat([inputs, attention], -1)`.
		  output_attention: Python bool.  If `True` (default), the output at each
			time step is the attention value.  This is the behavior of Luong-style
			attention mechanisms.  If `False`, the output at each time step is
			the output of `cell`.  This is the beahvior of Bhadanau-style
			attention mechanisms.  In both cases, the `attention` tensor is
			propagated to the next time step via the state and is used there.
			This flag only controls whether the attention mechanism is propagated
			up to the next cell in an RNN stack or to the top RNN output.
		  initial_cell_state: The initial state value to use for the cell when
			the user calls `zero_state()`.  Note that if this value is provided
			now, and the user uses a `batch_size` argument of `zero_state` which
			does not match the batch size of `initial_cell_state`, proper
			behavior is not guaranteed.
		  name: Name to use when creating ops.
		Raises:
		  TypeError: `attention_layer_size` is not None and (`attention_mechanism`
			is a list but `attention_layer_size` is not; or vice versa).
		  ValueError: if `attention_layer_size` is not None, `attention_mechanism`
			is a list, and its length does not match that of `attention_layer_size`.
		"""
		super(PCGNWrapper, self).__init__(name=name)
		if not rnn_cell_impl._like_rnncell(cell):  # pylint: disable=protected-access
			raise TypeError(
				"cell must be an RNNCell, saw type: %s" % type(cell).__name__)
		if isinstance(attention_mechanism, (list, tuple)):
			self._is_multi = True
			print('is multi attention')
			attention_mechanisms = attention_mechanism
			for attention_mechanism in attention_mechanisms:
				if not isinstance(attention_mechanism, AttentionMechanism):
					raise TypeError(
						"attention_mechanism must contain only instances of "
						"AttentionMechanism, saw type: %s"
						% type(attention_mechanism).__name__)
		else:
			self._is_multi = False
			print('is single attention')
			if not isinstance(attention_mechanism, AttentionMechanism):
				raise TypeError(
					"attention_mechanism must be an AttentionMechanism or list of "
					"multiple AttentionMechanism instances, saw type: %s"
					% type(attention_mechanism).__name__)
			attention_mechanisms = (attention_mechanism,)

		if cell_input_fn is None:
			cell_input_fn = (
				lambda inputs, attention: array_ops.concat([inputs, attention], -1))
		else:
			if not callable(cell_input_fn):
				raise TypeError(
					"cell_input_fn must be callable, saw type: %s"
					% type(cell_input_fn).__name__)

		if attention_layer_size is not None:
			attention_layer_sizes = tuple(
				attention_layer_size
				if isinstance(attention_layer_size, (list, tuple))
				else (attention_layer_size,))
			if len(attention_layer_sizes) != len(attention_mechanisms):
				raise ValueError(
					"If provided, attention_layer_size must contain exactly one "
					"integer per attention_mechanism, saw: %d vs %d"
					% (len(attention_layer_sizes), len(attention_mechanisms)))
			self._attention_layers = tuple(
				layers_core.Dense(
					attention_layer_size, name="attention_layer", use_bias=False)
				for attention_layer_size in attention_layer_sizes)
			self._attention_layer_size = sum(attention_layer_sizes)
		else:
			self._attention_layers = None
			self._attention_layer_size = sum(
				attention_mechanism.values.get_shape()[-1].value
				for attention_mechanism in attention_mechanisms)

		self._cell = cell
		self._attention_mechanisms = attention_mechanisms
		self._cell_input_fn = cell_input_fn
		self._output_attention = output_attention
		self._alignment_history = alignment_history
		# ==================================================================
		# PCGN hyperparameters
		self._user_embs = user_embs
		self._user_feats = user_feats
		self._user_feat_mem_units = user_feat_mem_units  # internal gated memory num units
		# PCGN internal memory
		self._user_feat_mem_embedding = user_feat_mem_embedding

		self.read_g = read_gate  # PCGN memory read_gate
		self.write_g = write_gate  # PCGN memory write_gate
		self.use_gate_memory = use_gate_memory
		self.read_attn_g = read_atten_gate
		# ==================================================================
		with ops.name_scope(name, "AttentionWrapperInit"):
			if initial_cell_state is None:
				self._initial_cell_state = None
			else:
				final_state_tensor = nest.flatten(initial_cell_state)[-1]
				state_batch_size = (
						final_state_tensor.shape[0].value
						or array_ops.shape(final_state_tensor)[0])
				error_message = (
						"When constructing PCGNWrapper %s: " % self._base_name +
						"Non-matching batch sizes between the memory "
						"(encoder output) and initial_cell_state.  Are you using "
						"the BeamSearchDecoder?  You may need to tile your initial state "
						"via the tf.contrib.seq2seq.tile_batch function with argument "
						"multiple=beam_width.")
				with ops.control_dependencies(
						self._batch_size_checks(state_batch_size, error_message)):
					self._initial_cell_state = nest.map_structure(
						lambda s: array_ops.identity(
							s, name="check_initial_cell_state"),
						initial_cell_state)

	def _batch_size_checks(self, batch_size, error_message):
		return [check_ops.assert_equal(batch_size,
		                               attention_mechanism.batch_size,
		                               message=error_message)
		        for attention_mechanism in self._attention_mechanisms]

	def _item_or_tuple(self, seq):
		"""Returns `seq` as tuple or the singular element.
		Which is returned is determined by how the AttentionMechanism(s) were passed
		to the constructor.
		Args:
		  seq: A non-empty sequence of items or generator.
		Returns:
		   Either the values in the sequence as a tuple if AttentionMechanism(s)
		   were passed to the constructor as a sequence or the singular element.
		"""
		t = tuple(seq)
		if self._is_multi:
			return t
		else:
			return t[0]

	@property
	def output_size(self):
		if self._output_attention:
			return self._attention_layer_size
		else:
			return self._cell.output_size

	@property
	def state_size(self):
		"""The `state_size` property of `PCGNWrapper`.
		Returns:
		  An `PCGNWrapperState` tuple containing shapes used by this object.
		"""
		return PCGNWrapperState(
			cell_state=self._cell.state_size,
			time=tensor_shape.TensorShape([]),
			attention=self._attention_layer_size,
			# ==============================================
			user_feat_mem=self._user_feat_mem_units,  # 返回内部memory的size
			# ==============================================
			alignments=self._item_or_tuple(
				a.alignments_size for a in self._attention_mechanisms),
			alignment_history=self._item_or_tuple(
				() for _ in self._attention_mechanisms))  # sometimes a TensorArray

	def zero_state(self, batch_size, dtype):
		"""Return an initial (zero) state tuple for this `PCGNWrapper`.
		**NOTE** Please see the initializer documentation for details of how
		to call `zero_state` if using an `PCGNWrapper` with a
		`BeamSearchDecoder`.
		Args:
		  batch_size: `0D` integer tensor: the batch size.
		  dtype: The internal state data type.
		Returns:
		  An `PCGNWrapperState` tuple containing zeroed out tensors and,
		  possibly, empty `TensorArray` objects.
		Raises:
		  ValueError: (or, possibly at runtime, InvalidArgument), if
			`batch_size` does not match the output size of the encoder passed
			to the wrapper object at initialization time.
		"""
		with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
			if self._initial_cell_state is not None:
				cell_state = self._initial_cell_state
			else:
				cell_state = self._cell.zero_state(batch_size, dtype)
			error_message = (
					"When calling zero_state of PCGNWrapper %s: " % self._base_name +
					"Non-matching batch sizes between the memory "
					"(encoder output) and the requested batch size.  Are you using "
					"the BeamSearchDecoder?  If so, make sure your encoder output has "
					"been tiled to beam_width via tf.contrib.seq2seq.tile_batch, and "
					"the batch_size= argument passed to zero_state is "
					"batch_size * beam_width.")
			with ops.control_dependencies(
					self._batch_size_checks(batch_size, error_message)):
				cell_state = nest.map_structure(
					lambda s: array_ops.identity(s, name="checked_cell_state"),
					cell_state)

			# =====================================================================
			# print('self._user_feats', self._user_feats)
			feat_imem_0 = self._user_feat_mem_embedding(self._user_feats)  # memory init
			return PCGNWrapperState(
				cell_state=cell_state,
				time=array_ops.zeros([], dtype=dtypes.int32),
				attention=_zero_state_tensors(self._attention_layer_size, batch_size,
				                              dtype),
				user_feat_mem=feat_imem_0,  #feat memory init
				alignments=self._item_or_tuple(
					attention_mechanism.initial_alignments(batch_size, dtype)
					for attention_mechanism in self._attention_mechanisms),
				alignment_history=self._item_or_tuple(
					tensor_array_ops.TensorArray(dtype=dtype, size=0,
					                             dynamic_size=True)
					if self._alignment_history else ()
					for _ in self._attention_mechanisms))

	# ======================================================================

	def _read_user_feat_mem(self, feat_imem, read_inputs):
		"""
		Read the internal memory
			feat_imem: [batch_size, men_int_units]
			read_inputs: [batch_size, d]
		Returns:
			M_read: [batch_size, men_int_units]
		"""
		gate_read = tf.nn.sigmoid(self.read_g(read_inputs))
		return feat_imem * gate_read

	def _read_attn_memory(self, user_attn, read_inputs):

		gate_read = tf.nn.sigmoid(self.read_attn_g(read_inputs))
		return user_attn * gate_read

	def _write_user_feat_mem(self, feat_imem, new_cell_state):
		"""
		Write the internal memory
			M_imem: [batch_size, men_int_units]
			new_cell_state: [batch_size, num_units]
		Returns:
			M_write: [batch_size, men_int_units]
		"""
		gate_write = tf.nn.sigmoid(self.write_g(new_cell_state))
		return feat_imem * gate_write

	def call(self, inputs, state):
		"""Perform a step of attention-wrapped RNN.
		- Step 1: Mix the `inputs` and previous step's `attention` output via
		  `cell_input_fn`.
		- Step 2: Call the wrapped `cell` with this input and its previous state.
		- Step 3: Score the cell's output with `attention_mechanism`.
		- Step 4: Calculate the alignments by passing the score through the
		  `normalizer`.
		- Step 5: Calculate the context vector as the inner product between the
		  alignments and the attention_mechanism's values (memory).
		- Step 6: Calculate the attention output by concatenating the cell output
		  and context through the attention layer (a linear layer with
		  `attention_layer_size` outputs).
		Args:
		  inputs: (Possibly nested tuple of) Tensor, the input at this time step.
		  state: An instance of `PCGNWrapperState` containing
			tensors from the previous time step.
		Returns:
		  A tuple `(attention_or_cell_output, next_state)`, where:
		  - `attention_or_cell_output` depending on `output_attention`.
		  - `next_state` is an instance of `PCGNWrapperState`
			 containing the state calculated at this time step.
		Raises:
		  TypeError: If `state` is not an instance of `PCGNWrapperState`.
		"""
		if not isinstance(state, PCGNWrapperState):
			raise TypeError("Expected state to be instance of PCGNWrapperState. "
			                "Received type %s instead." % type(state))

		# Step 1: Calculate the true inputs to the cell based on the
		# previous attention value.
		# =====================================================================
		if self.use_gate_memory:
			r_cell_state = state.cell_state  # 首先取出上一个状态中的cell_state
			# print(r_cell_state)
			r_cell_state = r_cell_state[-1]  # 取cell_state的最后一层的状态
			# print(r_cell_state)
			if isinstance(r_cell_state, LSTMStateTuple):  # 如果是lstm就将c和h拼接起来
				print('read_gate concat LSTMState C and H')
				r_cell_state = tf.concat([r_cell_state.c, r_cell_state.h], axis=-1)
			read_inputs = tf.concat([inputs, r_cell_state, state.attention],
			                        axis=-1)  # user_feat_mem read_gate inputs
			M_read = self._read_user_feat_mem(state.user_feat_mem, read_inputs)  # user_feat_mem的读取过程

			cell_inputs = tf.concat([inputs, state.attention, self._user_embs, M_read], axis=-1)  # 当前时序rnn_cell的输入
		else:
			cell_inputs = tf.concat([inputs, state.attention, self._user_embs], axis=-1)
		cell_state = state.cell_state
		cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

		if self.use_gate_memory:
			next_cell_state_to_write = next_cell_state[-1]  # 取最后一层的状态
			if isinstance(next_cell_state_to_write, LSTMStateTuple):
				print('write gate concat LSTMState C and H')
				next_cell_state_to_write = tf.concat([next_cell_state_to_write.c, next_cell_state_to_write.h], axis=-1)
			new_M_imem = self._write_user_feat_mem(state.user_feat_mem,
			                                        next_cell_state_to_write)  # user_feat_mem的写入过程
		else:
			new_M_imem = self._user_feat_mem_embedding(self._user_feats)
		# =======================================================================
		cell_batch_size = (
				cell_output.shape[0].value or array_ops.shape(cell_output)[0])
		error_message = (
				"When applying PCGNWrapper %s: " % self.name +
				"Non-matching batch sizes between the memory "
				"(encoder output) and the query (decoder output).  Are you using "
				"the BeamSearchDecoder?  You may need to tile your memory input via "
				"the tf.contrib.seq2seq.tile_batch function with argument "
				"multiple=beam_width.")
		with ops.control_dependencies(
				self._batch_size_checks(cell_batch_size, error_message)):
			cell_output = array_ops.identity(
				cell_output, name="checked_cell_output")

		if self._is_multi:
			previous_alignments = state.alignments
			previous_alignment_history = state.alignment_history
		else:
			previous_alignments = [state.alignments]
			previous_alignment_history = [state.alignment_history]

		all_alignments = []
		all_attentions = []
		all_histories = []
		for i, attention_mechanism in enumerate(self._attention_mechanisms):
			attention, alignments = _compute_attention(
				attention_mechanism, cell_output, previous_alignments[i],
				self._attention_layers[i] if self._attention_layers else None)
			if i > 0 and self.use_gate_memory:
				attention = self._read_attn_memory(attention, read_inputs)
			alignment_history = previous_alignment_history[i].write(
				state.time, alignments) if self._alignment_history else ()

			all_alignments.append(alignments)
			all_histories.append(alignment_history)
			all_attentions.append(attention)

		attention = array_ops.concat(all_attentions, 1)

		# ========================================================
		# 新的状态传递给下一个时序
		next_state = PCGNWrapperState(
			time=state.time + 1,
			cell_state=next_cell_state,
			attention=attention,
			user_feat_mem=new_M_imem,
			alignments=self._item_or_tuple(all_alignments),
			alignment_history=self._item_or_tuple(all_histories))
		# =========================================================
		if self._output_attention:
			return attention, next_state
		else:
			return cell_output, next_state
