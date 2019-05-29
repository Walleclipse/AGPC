import json
import re
from time import time

import numpy as np
import pandas as pd
from tensorflow.python.platform import gfile

_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _UNK, _EOS, _GO]

PAD_ID = 0
UNK_ID = 1
EOS_ID = 2
GO_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d")

def basic_tokenizer(sentence):
	"""Very basic tokenizer: split the sentence into a list of tokens."""
	words = []
	for space_separated_fragment in sentence.strip().split():
		words.extend(_WORD_SPLIT.split(space_separated_fragment))
	return [w for w in words if w]

def create_vocabulary(vocabulary_path, data, max_vocabulary_size,
                      tokenizer=None, normalize_digits=False):
	if not gfile.Exists(vocabulary_path):
		print("Creating vocabulary %s from data" % (vocabulary_path))
		print(len(data))
		vocab = {}
		counter = 0
		num = 0
		for line in data:
			counter += 1
			if counter % 100000 == 0:
				print("    processing line %d" % counter)
			# line = tf.compat.as_bytes(line)
			tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
			for w in tokens:
				num += 1
				word = _DIGIT_RE.sub("0", w) if normalize_digits else w
				if word in vocab:
					vocab[word] += 1
				else:
					vocab[word] = 1
		vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
		print('total words:', num, 'unique words:', len(vocab_list))
		if len(vocab_list) > max_vocabulary_size:
			vocab_list = vocab_list[:max_vocabulary_size]
			overlap = .0
			for key in vocab_list[len(_START_VOCAB):]:
				overlap += vocab[key]
			print("overlap %f" % (overlap / num))
		print('vocab size:',len(vocab_list))
		with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
			for w in vocab_list:
				vocab_file.write(w + "\n")


def initialize_vocabulary(vocabulary_path):
	if gfile.Exists(vocabulary_path):
		rev_vocab = []
		with gfile.GFile(vocabulary_path, mode="r") as f:
			rev_vocab.extend(f.readlines())
		rev_vocab = [line.strip() for line in rev_vocab]  # .decode('utf8')
		vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
		return vocab, rev_vocab
	else:
		raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary, tokenizer=None, normalize_digits=False):
	if tokenizer:
		words = tokenizer(sentence)
	else:
		words = basic_tokenizer(sentence)
	if not normalize_digits:
		return [vocabulary.get(w, UNK_ID) for w in words]
	# Normalize digits by 0 before looking words up in the vocabulary.
	return [vocabulary.get(_DIGIT_RE.sub("0", w), UNK_ID) for w in words]



def data_to_token_ids(data, vocab_path, tokenizer=None, normalize_digits=False):
	t1 = time()
	print("Tokenizing data")
	vocab, _ = initialize_vocabulary(vocab_path)
	vcol = ['Age', 'Birthday']
	onehotcol = ['Gender', 'Marital_State', 'City','Province']
	onehot_size = {'Province': 29, 'City': 47, 'Gender': 2, 'Marital_State': 8}
	counter = 0
	prep_data = []
	data = data.reset_index(drop=True)
	bloglist = data['Blog']
	blog2inds = {}
	for ii, uu in bloglist.items():
		if uu not in blog2inds:
			blog2inds[uu] = [ii]
		else:
			blog2inds[uu].append(ii)
	#print(len(bloglist), len(blog2inds))
	for raw_blog, inds in blog2inds.items():
		ablog = [[], []]
		blog = sentence_to_token_ids(raw_blog, vocab, tokenizer, normalize_digits)
		ablog[0].append([blog])
		pairs = data.iloc[inds]
		for ii, pair in pairs.iterrows():
			cmt = sentence_to_token_ids(pair['Comment'], vocab, tokenizer, normalize_digits)
			desc = sentence_to_token_ids(pair['Individual_Description'], vocab, tokenizer, normalize_digits)
			usr_feat = pair[vcol].values
			for feat in onehotcol:
				vonehot = np.eye(onehot_size[feat])[int(pair[feat])]
				usr_feat = np.concatenate([usr_feat, vonehot])
			usr_feat = list(usr_feat)
			ablog[1].append([cmt,  usr_feat, desc])
			counter += 1
			if counter % 100000 == 0:
				print("    processing line %d" % counter, (time() - t1) / 60)
		prep_data.append(ablog)
	return prep_data


def main(data_path='sample_data/sample_data.csv',vocab_size=40000,vocab_path='sample_data/sample_vocab.txt',prep_path='sample_data/sample_data.tokenids'):
	ssdata = pd.read_csv(data_path)
	blogs = pd.unique(ssdata['Blog'])
	comments = pd.unique(ssdata['Comment'])
	desc = pd.unique(ssdata['Individual_Description'])
	print('blogs:', len(blogs), 'comments:', len(comments), 'user description:', len(desc))

	data_vocab = np.concatenate([blogs, comments,desc])
	create_vocabulary(vocab_path, data_vocab, vocab_size)

	token_train = data_to_token_ids(ssdata, vocab_path)
	with open(prep_path, 'w') as output:
		output.write(json.dumps(token_train, ensure_ascii=False))

if __name__=='__main__':
	main(vocab_size=8000)


