# -*- coding:utf-8 -*-
import math

import tensorflow as tf
from tensorflow.contrib.seq2seq import tile_batch, BahdanauAttention, \
	BasicDecoder, dynamic_decode, TrainingHelper, AttentionWrapper, BeamSearchDecoder
from tensorflow.python.ops import array_ops, nn_ops
from tensorflow.python.ops import math_ops

from .PCGN_attention import PCGNWrapper
from .PCGN_beamsearch import PCGNBeamSearchDecoder
from .cell import create_rnn_cell
from .encoder import build_encoder

PAD_ID = 0
UNK_ID = 1
SOS_ID = 3
EOS_ID = 2

class PCGNModel():
	def __init__(self, mode='train',
	             model_name='PCGN',
	             vocab_size=40000, embedding_size=300,
	             encode_num_layers=2, encode_num_units=512, encode_cell_type='LSTM', encode_bidir=True,
	             attn_num_units=512,decode_num_layers=2, decode_num_units=512, decode_cell_type='LSTM',
	             use_user_feat=False, use_gate_memory=False,
	             use_user_desc=False, use_blog_user_coattn=False,
	             use_external_desc_express=False,use_external_feat_express=False,

	             user_feat_dim=88,user_feat_unit=50,user_feat_mem_unit=256,
	             desc_rnn_unit=200,desc_attn_num_units=256,user_map_unit=100,

	             batch_size=64, beam_search=False, beam_size=1, infer_max_iter=20,target_max_length=20,
	             l2_regularize=0, learning_rate=0.001, max_to_keep=100, max_gradient_norm=5.0,  # 1.0
	             ):

		self.mode = mode
		self.model_name = model_name
		# embedding params
		self.vocab_size = vocab_size
		self.embedding_size = embedding_size
		# blog encoder params
		self.encode_num_layers = encode_num_layers
		self.encode_num_units = encode_num_units
		self.encode_cell_type = encode_cell_type
		self.encode_bidir = encode_bidir
		# blog attention params
		self.attn_num_units = attn_num_units
		# decoder params
		self.decode_num_layers = decode_num_layers
		self.decode_num_units = decode_num_units
		self.decode_cell_type = decode_cell_type

		# PCGN params
		# 1. user feature gated memory
		self.use_user_feat = use_user_feat
		self.feat_dim =user_feat_dim #88
		self.user_feat_unit = user_feat_unit
		self.use_gate_memory = use_gate_memory
		self.user_feat_mem_unit = user_feat_mem_unit  # memory size
		# 2. user description cross attention
		self.use_user_desc = use_user_desc
		self.desc_rnn_unit = desc_rnn_unit
		self.use_blog_user_coattn = use_blog_user_coattn
		self.desc_attn_num_units = desc_attn_num_units
		# 3. external personality expression
		self.use_external_desc_express = use_external_desc_express
		self.use_external_feat_express = use_external_feat_express
		self.user_map_unit = user_map_unit #100

		# others
		self.batch_size = batch_size
		self.beam_search = beam_search
		self.beam_size = beam_size
		self.l2_regularize = l2_regularize
		self.infer_max_iter = infer_max_iter
		self.learning_rate = learning_rate
		self.max_to_keep = max_to_keep
		self.max_gradient_norm = max_gradient_norm
		self.target_max_length=target_max_length

		# build model
		print('user numeric feature:', use_user_feat, ' user individual description:', use_user_desc)
		print('use gated memory for user numeric featue:', use_gate_memory)
		print('use blog user co-attention:', use_blog_user_coattn)
		print('use external personality expression with user individual description:', use_external_desc_express,
		      'and user numeric feature:', use_external_feat_express)

		self.build_model()

	def init_embedding(self, vocab_size, embed_size, dtype=tf.float32,
	                   initializer=None, initial_values=None,
	                   ):
		"""
		embeddings:
			initialize trainable embeddings or load pretrained from files
		"""
		if initial_values:
			embedding = tf.Variable(initial_value=initial_values,
			                        name="embedding", dtype=dtype)
		else:
			if initializer is None:
				initializer = tf.contrib.layers.xavier_initializer()

			embedding = tf.Variable(
				initializer(shape=(vocab_size, embed_size)),
				name="embedding", dtype=dtype)

		return embedding

	def build_user_embedding(self, user_feat, user_desc, desc_length, user_feat_unit, desc_rnn_unit, embeddings,
	                         use_user_desc=True, use_user_feat=True,):
		init = tf.contrib.layers.xavier_initializer()
		act = tf.nn.relu
		if use_user_feat:
			feat_emb_layer1 = tf.layers.Dense(user_feat_unit, use_bias=False, kernel_initializer=init, activation=act,
			                                  name="feat_emb_layer1")
			feat_emb_layer2 = tf.layers.Dense(user_feat_unit, use_bias=False, kernel_initializer=init, activation=act,
			                                  name="feat_emb_layer2")
			user_feats = feat_emb_layer1(user_feat)
			user_embs = feat_emb_layer2(user_feats)
			user_desc_outputs=None

		if use_user_desc:
			user_desc_outputs, user_desc_states = build_encoder(embeddings, user_desc, desc_length, 1, desc_rnn_unit,
		                                                    'LSTM',
		                                                    bidir=True, name="user_encoder")  # 2
			desc_output = tf.reduce_mean(user_desc_outputs, axis=1)
			if not use_user_feat:
				user_embs = desc_output
				user_feats = desc_output
		return user_feats, user_embs, user_desc_outputs

	def get_context(self, query, keys, blog_desc_inetract, batch_size, num_units):

		query = tf.matmul(tf.reshape(query, (batch_size, num_units)), blog_desc_inetract)
		query = array_ops.expand_dims(query, 1)
		score = math_ops.matmul(query, keys, transpose_b=True)
		score = array_ops.squeeze(score, [1])
		alignments = nn_ops.softmax(score)
		expanded_alignments = array_ops.expand_dims(alignments, 1)
		context = math_ops.matmul(expanded_alignments, keys)
		context = array_ops.squeeze(context, [1])
		return context

	def external_personality_express(self, decoder_logits_train, keys, blog_desc_inetract, user_feats=None,
	                               use_external_feat_express=False,
	                               user_map=None):  # keys Processed memory, shape `[batch_size, max_time, num_units]`.
		"""
		Args:
		  query: Tensor, shape `[batch_size, num_units]` to compare to keys.
		  keys: Processed memory, shape `[batch_size, max_time, num_units]`.
		  scale: Whether to apply a scale to the score function.
		Returns:
		  A `[batch_size, max_time]` tensor of unnormalized score values.

		"""
		batch_size, max_time, num_units = decoder_logits_train.get_shape()
		num_unit_key = keys.get_shape()[2]
		t_decoder_logits = tf.transpose(decoder_logits_train, [1, 0, 2])
		if use_external_feat_express:
			user_feats = tf.reshape(user_feats, [1, batch_size, user_feats.get_shape()[-1]])
		cnt_list = []
		for ii in range(self.target_max_length):
			dec = tf.gather(t_decoder_logits, ii)
			context = self.get_context(dec, keys, blog_desc_inetract, batch_size, num_units)
			context = tf.reshape(context, [1, batch_size, num_unit_key])
			dec = tf.reshape(dec, [1, batch_size, num_units])
			if use_external_feat_express:
				context = tf.concat([context, user_feats], axis=-1)
			if user_map is not None:
				context = tf.einsum('bij,jk->bik', context, user_map)
			cnt = tf.concat([dec, context], axis=-1)
			cnt_list.append(cnt)
		aligned_decoder_logit = tf.concat(cnt_list, axis=0)
		aligned_decoder_logit = tf.transpose(aligned_decoder_logit, [1, 0, 2])
		return aligned_decoder_logit

	def build_model(self):
		print('building model... ...')
		with tf.variable_scope('seq2seq_placeholder'):
			self.encoder_inputs = tf.placeholder(tf.int32, [None, None], name="encoder_inputs")
			self.decoder_inputs = tf.placeholder(tf.int32, [None, None], name="decoder_inputs")
			self.decoder_targets = tf.placeholder(tf.int32, [None, None], name="decoder_targets")
			self.decoder_targets_masks = tf.placeholder(tf.bool, [None, None], name="mask")
			self.encoder_length = tf.placeholder(tf.int32, [None], name="encoder_length")
			self.decoder_length = tf.placeholder(tf.int32, [None], name="decoder_length")

			# PCGN placeholder
			self.user_feat = tf.placeholder(tf.float32, [None, self.feat_dim], name="user_feat")
			self.user_desc = tf.placeholder(tf.int32, [None, None], name="user_desc")
			self.desc_length = tf.placeholder(tf.int32, [None], name="user_desc_length")
			self.max_target_sequence_length = tf.constant(value=self.target_max_length,name='max_target_len')  # 20# tf.reduce_max(self.decoder_length, name='max_target_len')

		with tf.variable_scope('seq2seq_embedding'):
			self.embedding = self.init_embedding(self.vocab_size, self.embedding_size)

		with tf.variable_scope('seq2seq_encoder'):
			encoder_outputs, encoder_states = build_encoder(
				self.embedding, self.encoder_inputs, self.encoder_length,
				self.encode_num_layers, self.encode_num_units, self.encode_cell_type,
				bidir=self.encode_bidir)

		if self.use_user_desc or self.use_user_feat:
			with tf.variable_scope('user_profile_encoder'):
				# create emotion category embeddings
				desc_initializer = tf.contrib.layers.xavier_initializer()
				self.user_feat_mem_embedding = tf.layers.Dense(self.user_feat_mem_unit, use_bias=False,
				                                                 activation=tf.nn.relu,
				                                                 kernel_initializer=desc_initializer,
				                                                 name="user_feat_mem_layer")
				self.user_feats, self.user_embs, self.user_desc_encode = self.build_user_embedding(
					self.user_feat, self.user_desc,
					self.desc_length, self.user_feat_unit,
					self.desc_rnn_unit,
					self.embedding, self.use_user_desc, self.use_user_feat)

				if self.use_external_desc_express:
					#self.embed_desc = self.user_desc_encode
					dim2 = self.desc_rnn_unit
					dim1 = self.decode_num_units
					if self.use_blog_user_coattn:
						dim1 = dim1 * 2
					self.blog_desc_inetract = tf.Variable(desc_initializer(shape=(dim1, dim2)),
					                                name="blog_desc_inetraction_layer",
					                                dtype=tf.float32)
					if self.use_external_feat_express:
						dim2 = dim2 + self.user_feat_unit
					self.user_map_layer = tf.Variable(desc_initializer(shape=(dim2, self.user_map_unit)),
					                                  name="user_map_layer", dtype=tf.float32)


		with tf.variable_scope('seq2seq_decoder'):
			encoder_length = self.encoder_length
			if self.use_user_desc or self.use_user_feat:
				user_feats = self.user_feats
				user_embs = self.user_embs
				if self.use_user_desc:
					desc_length = self.desc_length
					user_desc_encode = self.user_desc_encode
			if self.beam_search:
				# 如果使用beam_search，则需要将encoder的输出进行tile_batch，其实就是复制beam_size份。
				print("use beamsearch decoding..")
				encoder_outputs = tile_batch(encoder_outputs, multiplier=self.beam_size)
				encoder_states = tile_batch(encoder_states, multiplier=self.beam_size)
				encoder_length = tile_batch(encoder_length, multiplier=self.beam_size)
				if self.use_user_desc or self.use_user_feat:
					user_feats = tile_batch(user_feats, multiplier=self.beam_size)
					user_embs = tile_batch(user_embs, multiplier=self.beam_size)
					if self.use_user_desc:
						desc_length = tile_batch(desc_length, multiplier=self.beam_size)
						user_desc_encode = tile_batch(user_desc_encode, multiplier=self.beam_size)

			attention_mechanism = BahdanauAttention(num_units=self.attn_num_units,
			                                        memory=encoder_outputs,
			                                        memory_sequence_length=encoder_length)

			if self.use_blog_user_coattn:
				attention_mechanism_desc = BahdanauAttention(num_units=self.desc_attn_num_units,
				                                                memory=user_desc_encode,
				                                                memory_sequence_length=desc_length)

			decoder_cell = create_rnn_cell(self.decode_num_layers, self.decode_num_units, self.decode_cell_type)

			if self.use_blog_user_coattn:
				_attention_mechanism=(attention_mechanism, attention_mechanism_desc)
				_attention_layer_size = [self.decode_num_units, self.decode_num_units]
			else:
				_attention_mechanism=attention_mechanism
				_attention_layer_size = self.decode_num_units

			if self.use_user_feat:
				if self.use_gate_memory:
					_read_g = tf.layers.Dense(
					self.user_feat_mem_unit, use_bias=False, name="internal_read_gate")

					_write_g = tf.layers.Dense(
					self.user_feat_mem_unit, use_bias=False, name="internal_write_gate")
					if self.use_blog_user_coattn:
						_read_atten_gate = tf.layers.Dense(
							2 * self.desc_attn_num_units, use_bias=False, name="internal_read_attn_gate")
					else:
						_read_atten_gate = None
				else:
					_read_g = None
					_write_g = None
					_read_atten_gate = None
				decoder_cell = PCGNWrapper(cell=decoder_cell,
					                          attention_mechanism=_attention_mechanism,
					                          user_feats=user_feats,
					                          user_embs=user_embs,
					                          user_feat_mem_units=self.user_feat_mem_unit,
					                          # memory size
					                          user_feat_mem_embedding=self.user_feat_mem_embedding,
									          read_gate=_read_g,
					                          write_gate=_write_g,
					                          use_gate_memory=self.use_gate_memory,
					                          attention_layer_size=_attention_layer_size,
					                          read_atten_gate=_read_atten_gate,
					                          name='PCGNWrapper')

			else:
				decoder_cell = AttentionWrapper(cell=decoder_cell, attention_mechanism=_attention_mechanism,
					                                attention_layer_size=_attention_layer_size,
					                                name='Attention_Wrapper')

			batch_size = self.batch_size if not self.beam_search else self.batch_size * self.beam_size

			decoder_initial_state = decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32).clone(
				cell_state=encoder_states)

			output_layer = tf.layers.Dense(self.vocab_size,
			                               use_bias=False,
			                               name='output_projection')  #

			if self.mode == 'train':
				decoder_inputs_embedded = tf.nn.embedding_lookup(self.embedding, self.decoder_inputs)
				# training helper的作用就是决定下一个时序的decoder的输入为给定的decoder inputs, 而不是上一个时刻的输出
				training_helper = TrainingHelper(inputs=decoder_inputs_embedded,
				                                 sequence_length=self.decoder_length,
				                                 name='training_helper')

				training_decoder = BasicDecoder(cell=decoder_cell,
				                                helper=training_helper,
				                                initial_state=decoder_initial_state)

				self.decoder_outputs, self.final_state, self.final_sequence_length = dynamic_decode(
					decoder=training_decoder,
					impute_finished=True,
					maximum_iterations=self.max_target_sequence_length)

				self.decoder_logits_train = tf.identity(self.decoder_outputs.rnn_output)

				if self.use_external_desc_express:
					if self.use_external_feat_express:
						_user_feats=user_embs
					else:
						_user_feats=None
					self.decoder_logits_train = self.external_personality_express(self.decoder_logits_train,
						                                                            user_desc_encode, self.blog_desc_inetract,
						                                                            user_feats=_user_feats,
						                                                            use_external_feat_express=self.use_external_feat_express,
						                                                            user_map=self.user_map_layer)
				with tf.variable_scope('decoder'):
					self.generic_logits = output_layer(self.decoder_logits_train)  # 得到普通词的概率分布logits

					if self.use_gate_memory:
						self.feat_mem = self.final_state.user_feat_mem  # user_feat_mem的最终状态

				with tf.variable_scope('loss'):
					g_probs = tf.nn.softmax(self.generic_logits)
					train_log_probs = tf.log(g_probs)
					self.g_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
						logits=self.generic_logits, labels=self.decoder_targets)  # - tf.log(1 - self.alphas)
					losses = tf.boolean_mask(self.g_losses, self.decoder_targets_masks)
					self.loss = tf.reduce_mean(losses)

					if self.use_gate_memory:
						self.int_mem_reg = tf.reduce_mean(tf.norm(self.feat_mem + 1e-7, axis=1))
						self.loss += self.int_mem_reg  # + self.alpha_reg

				# prepare for perlexity computations
				# self.decoder_targets_masks=tf.cast(self.decoder_targets_masks,tf.bool)
				CE = tf.nn.sparse_softmax_cross_entropy_with_logits(
					logits=train_log_probs, labels=self.decoder_targets)
				CE = tf.boolean_mask(CE, tf.cast(self.decoder_targets_masks, tf.bool))
				# CE = tf.boolean_mask(CE, self.decoder_targets_masks)
				self.CE = tf.reduce_mean(CE)

				# optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)#tf.train.AdamOptimizer(self.learning_rate)
				optimizer = tf.train.AdamOptimizer(self.learning_rate)  # beta1=0.5,beta2=0.9
				trainable_params = tf.trainable_variables()
				gradients = tf.gradients(self.loss, trainable_params)
				clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
				self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))

			elif self.mode == 'infer':
				start_tokens = tf.ones([self.batch_size, ], tf.int32) * SOS_ID
				end_token = EOS_ID
				if self.use_user_feat or self.use_user_desc :
					if self.use_external_desc_express:
						_embed_desc=user_desc_encode
						_blog_desc_inetract=self.blog_desc_inetract
						_user_map=self.user_map_layer
						if self.use_external_feat_express:
							_feat_embed=user_embs
						else:
							_feat_embed=None
					else:
						_embed_desc=None
						_blog_desc_inetract=None
						_user_map=None
						_feat_embed = None

					inference_decoder = PCGNBeamSearchDecoder(cell=decoder_cell,
							                                embedding=self.embedding,
							                                start_tokens=start_tokens,
					                                         end_token=end_token,
					                                         initial_state=decoder_initial_state,
					                                         beam_width=self.beam_size,
					                                         output_layer=output_layer,
					                                         use_external_desc_express=self.use_external_desc_express,
					                                         embed_desc=_embed_desc,
					                                         blog_desc_inetract=_blog_desc_inetract,
					                                          feat_embed=_feat_embed,
					                                          use_external_feat_express=self.use_external_feat_express,
					                                         user_map=_user_map)

				else:
					inference_decoder = BeamSearchDecoder(cell=decoder_cell,
					                                      embedding=self.embedding,
					                                      start_tokens=start_tokens,
					                                      end_token=end_token,
					                                      initial_state=decoder_initial_state,
					                                      beam_width=self.beam_size,
					                                      output_layer=output_layer)
				decoder_outputs, _, _ = dynamic_decode(decoder=inference_decoder,
				                                       maximum_iterations=self.infer_max_iter)

				infer_outputs = decoder_outputs.predicted_ids  # [batch_size, decoder_targets_length, beam_size]
				self.infer_outputs = tf.transpose(infer_outputs, [0, 2, 1],
				                                  name='infer_outputs')  # [batch_size, beam_size, decoder_targets_length]

		self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.max_to_keep)

	def train(self, sess, batch):
		feed_dict = {
			self.encoder_inputs: batch[0],
			self.encoder_length: batch[1],
			self.decoder_inputs: batch[2],
			self.decoder_length: batch[3],
			self.decoder_targets: batch[4],
			self.decoder_targets_masks: batch[5],

			self.user_desc: batch[6],
			self.desc_length: batch[7],
			self.user_feat: batch[8],
		}
		_, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
		return loss


	def eval(self, sess, batch):
		feed_dict = {
			self.encoder_inputs: batch[0],
			self.encoder_length: batch[1],
			self.decoder_inputs: batch[2],
			self.decoder_length: batch[3],
			self.decoder_targets: batch[4],
			self.decoder_targets_masks: batch[5],

			self.user_desc: batch[6],
			self.desc_length: batch[7],
			self.user_feat: batch[8],
		}
		loss = sess.run(self.CE, feed_dict=feed_dict)
		return loss

	def infer(self, sess, batch):
		feed_dict = {
			self.encoder_inputs: batch[0],
			self.encoder_length: batch[1],
			self.user_desc: batch[6],
			self.desc_length: batch[7],
			self.user_feat: batch[8],
		}
		predict = sess.run(self.infer_outputs, feed_dict=feed_dict)
		return predict

	def compute_perplexity(self, sess, batch):
		feed_dict = {
			self.encoder_inputs: batch[0],
			self.encoder_length: batch[1],
			self.decoder_inputs: batch[2],
			self.decoder_length: batch[3],
			self.decoder_targets: batch[4],
			self.decoder_targets_masks: batch[5],

			self.user_desc: batch[6],
			self.desc_length: batch[7],
			self.user_feat: batch[8],
		}
		# loss = sess.run(self.loss, feed_dict=feed_dict)
		loss = sess.run(self.CE, feed_dict=feed_dict)
		perplexity = math.exp(float(loss))
		return perplexity
