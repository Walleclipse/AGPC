
def get_pcgn_model_config(config):
	encode_num_layers = config["encoder"]["num_layers"]
	encode_num_units = config["encoder"]["num_units"]
	encode_cell_type = config["encoder"]["cell_type"]
	encode_bidir = config["encoder"]["bidirectional"]

	attn_num_units = config["decoder"]["attn_num_units"]
	decode_num_layers = config["decoder"]["num_layers"]
	decode_num_units = config["decoder"]["num_units"]
	decode_cell_type = config["decoder"]["cell_type"]


	use_user_feat = config["user_profile"]["use_user_feat"]
	use_gate_memory = config["user_profile"]["use_gate_memory"]
	use_user_desc = config["user_profile"]["use_user_desc"]
	use_blog_user_coattn = config["user_profile"]["use_blog_user_coattn"]
	use_external_desc_express = config["user_profile"]["use_external_desc_express"]
	use_external_feat_express = config["user_profile"]["use_external_feat_express"]

	user_feat_dim = config["user_profile"]["user_feat_dim"]
	user_feat_unit = config["user_profile"]["user_feat_unit"]
	user_feat_mem_unit = config["user_profile"]["user_feat_mem_unit"]
	desc_rnn_unit = config["user_profile"]["desc_rnn_unit"]
	desc_attn_num_units = config["user_profile"]["desc_attn_num_units"]
	user_map_unit = config["user_profile"]["user_map_unit"]

	return (encode_num_layers, encode_num_units, encode_cell_type, encode_bidir,
	 attn_num_units, decode_num_layers, decode_num_units, decode_cell_type,
	 use_user_feat,use_gate_memory,use_user_desc,use_blog_user_coattn,
	 use_external_desc_express,use_external_feat_express,
	 user_feat_dim,user_feat_unit,user_feat_mem_unit,
	 desc_rnn_unit,desc_attn_num_units,user_map_unit,
	 )



def get_pcgn_training_config(config):
	train_file = config["training"]["train_file"]
	dev_file = config["training"]["dev_file"]
	source_max_length = config["training"]["source_max_length"]
	target_max_length = config["training"]["target_max_length"]
	desc_max_length = config["training"]["desc_max_length"]

	gpu_fraction = config["training"]["gpu_fraction"]
	gpu_id = config["training"]["gpu_id"]
	train_steps = config["training"]["train_steps"]  # 最大训练步数
	checkpoint_every = config["training"]["checkpoint_every"]  # 保存模型的步数
	print_every = config["training"]["print_every"]  # 打印信息

	batch_size = config["training"]["batch_size"]
	is_beam_search = False
	beam_size = 1
	infer_max_iter = config["training"]["infer_max_iter"]

	l2_regularize = config["training"]["l2_regularize"]
	learning_rate = config["training"]["learning_rate"]
	max_checkpoints = config["training"]["max_checkpoints"]  # 最大保留模型的个数
	max_gradient_norm = config["training"]["max_gradient_norm"]  # 最大保留模型的个数

	return (train_file, dev_file,
	 source_max_length, target_max_length, desc_max_length,
	 gpu_fraction, gpu_id, train_steps, checkpoint_every, print_every,
	 batch_size,is_beam_search,beam_size,infer_max_iter,
	 l2_regularize,learning_rate,max_checkpoints,max_gradient_norm,
	  )


def get_pcgn_infer_config(config):
	is_beam_search = config["inference"]["is_beam_search"]
	beam_size = config["inference"]["beam_size"]
	batch_size = config["inference"]["infer_batch_size"]
	infer_file = config["inference"]["infer_file"]

	infer_source_max_length = config["inference"]["infer_source_max_length"]
	infer_target_max_length = config["inference"]["infer_target_max_length"]
	infer_desc_max_length = config["inference"]["infer_desc_max_length"]

	infer_max_iter = config["inference"]["infer_max_iter"]
	output_path = config["inference"]["output_path"]

	gpu_fraction = config["training"]["gpu_fraction"]
	gpu_id = config["training"]["gpu_id"]

	return (infer_file, batch_size,is_beam_search, beam_size,
	 infer_source_max_length, infer_target_max_length,infer_desc_max_length,infer_max_iter,
	 output_path, gpu_fraction, gpu_id)
