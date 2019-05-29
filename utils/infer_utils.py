
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

PAD = "_PAD"
UNK = "_UNK"
SOS = "_GO"
EOS = "_EOS"

PAD_ID = 0
UNK_ID = 1
EOS_ID = 2
SOS_ID = 3

def create_vocab_tables(vocab_file):
	def load_vocab(vocab_file):
		vocab = []
		for _, word in enumerate(open(vocab_file, 'r',encoding='utf-8')):
			vocab.append(word.strip())#.decode('utf-8')
		return vocab
	vocab = load_vocab(vocab_file)
	vocab_table = {k: v for v, k in enumerate(vocab)}
	reverse_vocab_table = {k: v for k, v in enumerate(vocab)}
	return vocab_table, reverse_vocab_table

def batch_token_to_str(data_inds, reverse_vocab_table):
	data_word=[]
	for tokens in data_inds:
		word_list = [reverse_vocab_table[id] for id in list(tokens) if id >0 and id not in [EOS_ID,SOS_ID,PAD_ID]]
		sentence = " ".join(word_list)
		data_word.append(sentence)
	return data_word

def token_to_str(tokens, reverse_vocab_table):
	tokens = list(tokens)
	word_list = [reverse_vocab_table[id] for id in tokens if id>0 and id not in [EOS_ID,SOS_ID,PAD_ID]]
	sentence = " ".join(word_list)
	return sentence

def featinds2df(feats):
	df=pd.DataFrame()
	for ii, ff in enumerate(feats):
		temp={}
		temp['Age']=int(ff[0])
		temp['Birthday']=int(ff[1])
		temp['Gender'] = np.argmax(ff[2:4])
		temp['Marital_State'] = np.argmax(ff[4:12])
		temp['City'] = np.argmax(ff[12:59])
		temp['Province'] = np.argmax(ff[59:])

		temp=pd.DataFrame(temp,index=[ii])
		df=pd.concat([df,temp])
	return df

def calc_bleu2(hypotheis, refers):
	bleu = 0
	max_bleu = 0
	smoothie = SmoothingFunction()
	for h, r in zip(hypotheis, refers):
		rr = [x for x in r]
		hh = [x for x in h]

		try:
			hh = hh[: hh.index(EOS_ID)] # truncated to EOS
		except:
			hh = hh
		cur_bleu = sentence_bleu([rr], hh, weights=(0, 1, 0, 0),
								 smoothing_function=smoothie.method1)  # BLEU2
		bleu += cur_bleu
		if cur_bleu > max_bleu:
			max_bleu = cur_bleu
	return (bleu / len(hypotheis), max_bleu)


