# -*- coding:utf-8 -*-

import os
import sys
import tensorflow as tf

def load_model(saver, sess, logdir, model_path=None):
	"""
	Load the latest checkpoint
	"""
	print("Trying to restore saved checkpoints from {} ...".format(logdir),
		  end="")
	if model_path:
		global_step = int(model_path.split("-")[-1])
		print("Global step was: {}".format(global_step))
		print("Restoring...",end="")
		saver.restore(sess, model_path)
		print("\tDone")
		return global_step
	else:
		ckpt = tf.train.get_checkpoint_state(logdir)
		if ckpt:
			print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
			global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
			print("  Global step was: {}".format(global_step))
			print("  Restoring...", end="")
			saver.restore(sess, ckpt.model_checkpoint_path)
			print(" Done.")
			return global_step
		else:
			print(" No checkpoint found.")
			return None


def save_model(saver, sess, logdir, step):
	"""
	Save the checkpoint
	"""
	model_name = 'model.ckpt'
	checkpoint_path = os.path.join(logdir, model_name)
	print('Storing checkpoint to {} ...'.format(logdir), end="")
	sys.stdout.flush()

	if not os.path.exists(logdir):
		os.makedirs(logdir)

	saver.save(sess, checkpoint_path, global_step=step)
	print(' Done.')


def setup_workpath(workspace):
	for p in ['data', 'nn_models', 'results']:
		wp = "{}/{}".format(workspace, p)
		if not os.path.exists(wp):
			os.mkdir(wp)

def add_summary(summary_writer, global_step, tag, value):
	"""
	Add a new summary to the current summary_writer.
	Useful to log things that are not part of the training graph, e.g., tag=BLEU.
	"""
	summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
	summary_writer.add_summary(summary, global_step)