# -*- coding:utf-8 -*-

import os
import argparse
import time
import yaml
import tensorflow as tf
import numpy as np

from utils.config_utils import get_pcgn_model_config,get_pcgn_training_config
from utils.data_utils import read_data,get_pcgn_batch
from utils.model_utils import load_model, save_model, setup_workpath,add_summary
from model.PCGN_model import PCGNModel

use_tensorboard=False

def parse_args():
	'''
	Parse Seq2seq with attention arguments.
	'''
	parser = argparse.ArgumentParser(description="Run PCGN training.")

	parser.add_argument('--config', nargs='?',
						default='./configs/pcgn_config.yaml',
						help='Configuration file for model specifications')

	return parser.parse_args()

def main(config):

	# set up workspace
	work_space = config["workspace"]
	tf_board = config["tf_board"]
	setup_workpath(work_space)
	name = config["Name"]

	# Construct or load embeddings
	print("Initializing embeddings ...")
	vocab_size = config["embeddings"]["vocab_size"]
	embed_size = config["embeddings"]["embed_size"]

	# Build the model and compute losses
	(encode_num_layers, encode_num_units, encode_cell_type, encode_bidir,
	 attn_num_units, decode_num_layers, decode_num_units, decode_cell_type,
	 use_user_feat,use_gate_memory,use_user_desc,use_blog_user_coattn,
	 use_external_desc_express,use_external_feat_express,
	 user_feat_dim,user_feat_unit,user_feat_mem_unit,
	 desc_rnn_unit,desc_attn_num_units,user_map_unit,
	 ) = get_pcgn_model_config(config)

	(train_file, dev_file,
	 source_max_length, target_max_length, desc_max_length,
	 gpu_fraction, gpu_id, train_steps, checkpoint_every, print_every,
	 batch_size,is_beam_search,beam_size,infer_max_iter,
	 l2_regularize,learning_rate,max_checkpoints,max_gradient_norm,
	  ) = get_pcgn_training_config(config)

	train_set=read_data(train_file)
	print(' # train data:',len(train_set))
	dev_set=read_data(dev_file)
	print(' # dev data:',len(dev_set))

	print("Building model architecture ")
	pcg_model = PCGNModel(
		mode='train', model_name=name,
		vocab_size=vocab_size, embedding_size=embed_size,
		encode_num_layers=encode_num_layers, encode_num_units=encode_num_units,
		encode_cell_type=encode_cell_type, encode_bidir=encode_bidir,
		attn_num_units=attn_num_units, decode_num_layers=decode_num_layers,
		decode_num_units=decode_num_units, decode_cell_type=decode_cell_type,
		use_user_feat=use_user_feat, use_gate_memory=use_gate_memory,
		use_user_desc=use_user_desc, use_blog_user_coattn=use_blog_user_coattn,
		use_external_desc_express=use_external_desc_express, use_external_feat_express=use_external_feat_express,

		user_feat_dim=user_feat_dim, user_feat_unit=user_feat_unit, user_feat_mem_unit=user_feat_mem_unit,
		desc_rnn_unit=desc_rnn_unit, desc_attn_num_units=desc_attn_num_units, user_map_unit=user_map_unit,

		batch_size=batch_size, beam_search=is_beam_search, beam_size=beam_size, infer_max_iter=infer_max_iter, target_max_length=target_max_length,
		l2_regularize=l2_regularize, learning_rate=learning_rate, max_to_keep=max_checkpoints, max_gradient_norm=max_gradient_norm,
	)

	print("\tDone.")


	logdir = '%s/nn_models/' % work_space

	# Set up session
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction, visible_device_list=gpu_id,allow_growth=True)

	sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
											gpu_options=gpu_options))
	init = tf.global_variables_initializer()
	sess.run(init)

	# tensorbord
	if use_tensorboard:
		train_writer = tf.summary.FileWriter(tf_board + 'train/', sess.graph)
		test_writer = tf.summary.FileWriter(tf_board + 'test/', sess.graph)

	try:
		saved_global_step = load_model(pcg_model.saver, sess, logdir)
		if saved_global_step is None:
			saved_global_step = -1

	except Exception:
		print("Something went wrong while restoring checkpoint. "
			  "Training is terminated to avoid the overwriting.")
		raise

	# ##### Training #####

	# Training
	last_saved_step = saved_global_step
	num_steps = saved_global_step + train_steps
	steps = []
	previous_losses=[]
	lr = pcg_model.learning_rate

	print("Start training ...")
	print('steps per epoch:',len(train_set)//batch_size)
	try:
		for step in range(saved_global_step + 1, num_steps):
			start_time = time.time()

			batch = get_pcgn_batch(train_set,'train', batch_size,source_max_length, target_max_length,desc_max_length)
			loss_value = pcg_model.train(sess, batch)
			previous_losses.append(loss_value)
			lr_decay_step = 10
			if step % 500 == 0 and len(previous_losses)-5 > lr_decay_step and np.mean(previous_losses[-5:]) >= np.mean(previous_losses[-lr_decay_step -5:-5]):
				lr=pcg_model.learning_rate
				if lr > 0.00001:
					pcg_model.learning_rate=lr*0.9
					print('learning rate decay:',lr*0.9)
			duration = (time.time() - start_time)
			if step % print_every == 0 and step != 0:
				# train perplexity
				t_perp = pcg_model.compute_perplexity(sess, batch)
				if use_tensorboard:
					add_summary(train_writer, step, 'train perplexity', t_perp)

				# eval perplexity
				dev_str = ""
				if dev_set is not None:
					eval_batch = get_pcgn_batch(dev_set,'train', batch_size,source_max_length, target_max_length,desc_max_length)
					eval_perp = pcg_model.compute_perplexity(sess, eval_batch)
					with open(logdir+'eval_perp.txt','a',encoding='utf-8') as f:
						f.write('{}\t{}\n'.format(str(step),str(eval_perp)))

					if use_tensorboard:
						add_summary(test_writer, step, 'eval perplexity', eval_perp)
					dev_str += "val_prep: {:.3f}\n".format(eval_perp)

				steps.append(step)
				ep=step//(len(train_set)//batch_size)
				info = 'epoch {:d}, step {:d},lr:{:.5f}, loss = {:.6f},perp: {:.3f}\n{}({:.3f} sec/step)'
				print(info.format(ep,step,lr, loss_value, t_perp, dev_str, duration))

			if step % checkpoint_every == 0:
				save_model(pcg_model.saver, sess, logdir, step)
				last_saved_step = step

	except KeyboardInterrupt:
		# Introduce a line break after ^C so save message is on its own line.
		print()

	finally:
		if step > last_saved_step:
			save_model(pcg_model.saver, sess, logdir, step)

if __name__ == "__main__":
	args=parse_args()
	with open(args.config) as f:
		config = yaml.safe_load(f)["configuration"]
	main(config)
