
import tensorflow as tf
import numpy as np
import os
import sys
from collections import defaultdict
import random
import json

random.seed(1994)

PAD = "_PAD"
UNK = "_UNK"
SOS = "_GO"
EOS = "_EOS"

PAD_ID = 0
UNK_ID = 1
SOS_ID = 3
EOS_ID = 2

embed_shift = 4

def get_pcgn_batch(dataset, mode='train',batch_size=32,s_max_length=50, t_max_length=50,d_max_length=20):
	encoder_inputs, decoder_inputs, decoder_targets, decoder_targets_masks, \
	encoder_length, decoder_length, usr_descs, desc_length,usr_feats=  [], [], [], [], [], [], [],[],[]
	count = 0
	if mode!='train':
		batch_size=len(dataset)
	while count < batch_size:
		if mode=='train':
			encoder, decoder, feat, desc = random.choice(dataset)
		else:
			encoder, decoder, feat, desc = dataset[count]

		encoder = encoder[:s_max_length]
		encoder_input = encoder
		e_length = len(encoder_input)
		pads = PAD_ID * np.ones(s_max_length - e_length, dtype=np.int32)
		encoder_input = np.concatenate([encoder_input, pads])
		encoder_inputs.append(encoder_input)
		encoder_length.append(e_length)

		decoder = decoder[:t_max_length-1]
		decoder_input = [SOS_ID] + decoder
		d_length = len(decoder_input)
		pads = PAD_ID * np.ones(t_max_length  - d_length, dtype=np.int32)
		decoder_input = np.concatenate([decoder_input, pads])
		decoder_inputs.append(decoder_input)

		decoder_target = decoder + [EOS_ID]
		pads = PAD_ID * np.ones(t_max_length - d_length, dtype=np.int32)
		decoder_target = np.concatenate([decoder_target, pads])
		decoder_targets.append(decoder_target)
		decoder_length.append(t_max_length)

		desc = desc[:d_max_length]
		de_length = len(desc)
		pads = PAD_ID * np.ones(d_max_length - de_length, dtype=np.int32)
		desc = np.concatenate([desc, pads])
		usr_descs.append(desc)
		desc_length.append(de_length)


		usr_feats.append(feat)
		count += 1

	target_max_length = max(decoder_length)
	encoder_inputs = np.array(encoder_inputs)
	decoder_inputs = np.array(decoder_inputs)
	decoder_targets = np.array(decoder_targets)[:, :target_max_length]

	decoder_targets_masks = decoder_targets != PAD_ID # eos字符算在loss里
	usr_descs = np.array(usr_descs)
	usr_feats = np.array(usr_feats)

	encoder_length = np.array(encoder_length)
	decoder_length = np.array(decoder_length)
	desc_length = np.array(desc_length)


	return (encoder_inputs,encoder_length, decoder_inputs,decoder_length, decoder_targets, decoder_targets_masks,
			usr_descs, desc_length,usr_feats)


def read_data(path, max_size=None):
	data_set = []
	data = json.load(open(path,'r'))
	counter = 0
	for pair in data:
		post = pair[0][0]
		responses = pair[1]
		source_ids = [int(x) for x in post[0]]
		for response in responses: #[ 0.comment, 1.usr_feat, 2.individual_description]
			if not max_size or counter < max_size:
				counter += 1
				if counter % 100000 == 0:
					print("    reading data pair %d" % counter)
					sys.stdout.flush()
				target_ids = [int(x) for x in response[0]]
				feat=response[1]
				desc_ids = [int(x) for x in response[2]]
				data_set.append([source_ids, target_ids, feat, desc_ids])
	return data_set