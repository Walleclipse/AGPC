
import os
import argparse
import yaml
import tensorflow as tf
import pandas as pd

from utils.config_utils import get_pcgn_model_config, get_pcgn_infer_config
from utils.data_utils import read_data,get_pcgn_batch
from utils.model_utils import load_model
from model.PCGN_model import PCGNModel
from utils.infer_utils import create_vocab_tables, batch_token_to_str,featinds2df, calc_bleu2

def parse_args():
	'''
	Parse Seq2seq with attention arguments.
	'''
	parser = argparse.ArgumentParser(description="Run PCGN inference.")

	parser.add_argument('--config', nargs='?',
						default='./configs/pcgn_config.yaml',
						help='Configuration file for model specifications')

	return parser.parse_args()


def infer(config,test_bleu=True):

	work_space = config["workspace"]
	name = config["Name"]

	# Construct or load embeddings
	print("Initializing embeddings ...")
	vocab_size = config["embeddings"]["vocab_size"]
	embed_size = config["embeddings"]["embed_size"]
	vocab_file = config["inference"]["vocab_file"]

	# Build the model
	(encode_num_layers, encode_num_units, encode_cell_type, encode_bidir,
	 attn_num_units, decode_num_layers, decode_num_units, decode_cell_type,
	 use_user_feat,use_gate_memory,use_user_desc,use_blog_user_coattn,
	 use_external_desc_express,use_external_feat_express,
	 user_feat_dim,user_feat_unit,user_feat_mem_unit,
	 desc_rnn_unit,desc_attn_num_units,user_map_unit,
	 ) = get_pcgn_model_config(config)

	(infer_file, batch_size,is_beam_search, beam_size,
	 infer_source_max_length, infer_target_max_length,infer_desc_max_length,infer_max_iter,
	 output_path, gpu_fraction, gpu_id) = get_pcgn_infer_config(config)

	print("Building model architecture ...")
	pcg_model = PCGNModel(
		mode='infer', model_name=name,
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

		batch_size=batch_size, beam_search=is_beam_search, beam_size=beam_size, infer_max_iter=infer_max_iter, target_max_length=infer_target_max_length,
	)



	print("\tDone.")

	logdir = '%s/nn_models/' % work_space
	# Set up session
	gpu_fraction = config["training"]["gpu_fraction"]
	gpu_id = config["training"]["gpu_id"]
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction, visible_device_list=gpu_id,allow_growth=True)
	sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
											gpu_options=gpu_options))

	init = tf.global_variables_initializer()
	sess.run(init)
	try:
		saved_global_step = load_model(pcg_model.saver, sess, logdir)
		if saved_global_step is None:
			raise ValueError("Cannot find the checkpoint to restore from.")

	except Exception:
		print("Something went wrong while restoring checkpoint. ")
		raise

	# ##### Inference #####
	# Load data
	print("Loading inference data ...")

	# Load vocabularies.
	vocab_table, reverse_vocab_table = create_vocab_tables(vocab_file)

	infer_dataset = read_data(infer_file)
	print(' # infer data:',len(infer_dataset))
	print("\tDone.")

	# Inference
	print("Start inferring ...")
	final_result = pd.DataFrame()
	infer_step = int(len(infer_dataset) / batch_size)
	preds=[]
	for ith in range(infer_step):
		print('step:',ith)
		start = ith * batch_size
		end = (ith + 1) * batch_size
		batch = get_pcgn_batch(infer_dataset[start:end], 'infer',-1,infer_source_max_length, infer_target_max_length,infer_desc_max_length)

		result = pcg_model.infer(sess, batch)
		result1 = batch_token_to_str(result[:, 0, :], reverse_vocab_table)
		#result2 = batch_token_to_str(result[:, 1,:], reverse_vocab_table)
		#result3 = batch_token_to_str(result[:, 2,:], reverse_vocab_table)
		#result4 = batch_token_to_str(result[:, 3,:], reverse_vocab_table)
		#result5 = batch_token_to_str(result[:, 4,:], reverse_vocab_table)
		preds += list(result1)

		if test_bleu:
			blog = batch_token_to_str(batch[0],reverse_vocab_table)
			cmt = batch_token_to_str(batch[2],reverse_vocab_table)
			desc = batch_token_to_str(batch[6],reverse_vocab_table)
			feat_df = featinds2df(batch[8])


			df_result = pd.DataFrame({'Blog':blog,'Comment':cmt,'Individual_Description':desc,
									  'Prediction':result1,})
			df_result = pd.concat([df_result,feat_df],axis=1)
			final_result= pd.concat([final_result,df_result])

	out_path = config["inference"]["output_path"] + 'prediction' + '.txt'
	with open(out_path,'w') as f:
		f.write('\n'.join(preds))

	if test_bleu:
		bleu2=calc_bleu2(final_result['Prediction'].values, final_result['Comment'].values)
		print('test bleu:',bleu2)
		bleurecord='test_size:{}\trestore_step:{}\n'.format(str(int(infer_step*batch_size)),str(saved_global_step))
		bleurecord+='bleu2:{}\n\n'.format(str(bleu2[0]))
		with open(logdir+'bleu.txt','a') as f:
			f.write(bleurecord)

		out_path=config["inference"]["output_path"]+'prediction'+'.csv'
		final_result.to_csv(out_path, index=False)

	print("\tDone.")


if __name__ == "__main__":
	args = parse_args()
	with open(args.config) as f:
		config = yaml.safe_load(f)["configuration"]
	infer(config,test_bleu=True)
