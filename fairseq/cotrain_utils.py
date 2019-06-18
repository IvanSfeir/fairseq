#!/usr/bin/env python3 -u
# Copyright (c) 2019-present, Ivan Sfeir.
# All rights reserved.
"""
Library for co-training the double model architecture for coreference resolution.
Code written during my internship at GETALP team, LIG, Grenoble, from Feb to Jul 2019.
"""

from tqdm import tqdm
import pandas as pd
import itertools
import torch

"""
GENERAL UTILS
"""

def load_firsts(args, set='train'):
	firsts_list = []
	with open("{}/{}.firsts.txt".format(args.data[0], set), "r") as f:
		for first in f.readlines():
			firsts_list.append(int(first[:-1]))
	return firsts_list

"""
LABELS TO SPAN REPRESENTATIONS
"""

def labels_to_simple_span_representations(i, sentence, trainer, args):
	"""Returns the list of span objects of each span in one BIO sentence with id i"""
	# Span object is defined in my_data_utils.py and contains, ebtre autres
		# first: first index in the sentence
		# length: span length
		# representation: 1d tensor containing the span's representation
	# Uses simple representation by averaging
	# It is interesting to use either golden data or predictions

	def find_all_sub(beg_idx, end_idx):
		for sub_beg_idx in range(beg_idx, end_idx):
			for sub_end_idx in range(sub_beg_idx+1, end_idx+1):
				yield sub_beg_idx, sub_end_idx

	beg_idx = 0
	end_idx = 0
	idx = 0 # The index which goes through the sentence
	span_representations = []

	while idx < len(sentence):
		if (sentence[idx] in [6, 7]): # 'B-N or B-P'
			beg_idx = idx
			idx += 1
			while (idx < len(sentence)) and (sentence[idx] in [5, 8]): # 'I-N or I-P'
				idx += 1
			end_idx = idx
			# Compute all spans inside the detected span
			for sub_beg_idx, sub_end_idx in find_all_sub(beg_idx, end_idx):
				raw_left_representation = trainer.encoder_output_list[i][:sub_beg_idx, :]
				raw_span_representation = trainer.encoder_output_list[i][sub_beg_idx:sub_end_idx, :]
				raw_right_representation = trainer.encoder_output_list[i][sub_end_idx:, :]
				# Create span representation from raw span representation:
				# First word, last word, average, average left, average right
				span_representation = torch.cat((
					raw_span_representation[0],
					raw_span_representation[-1],
					torch.mean(raw_span_representation, 0),
					torch.mean(raw_left_representation, 0) if raw_left_representation.size()[0] != 0 \
						else torch.zeros(args.encoder_embed_dim).to(torch.device("cuda")),
					torch.mean(raw_right_representation, 0) if raw_right_representation.size()[0] != 0 \
						else torch.zeros(args.encoder_embed_dim).to(torch.device("cuda")),
					), 0)
				# Create Span object
				s = Span(i, sub_beg_idx, sub_end_idx-sub_beg_idx)
				s.update_representation(torch.squeeze(span_representation))
				span_representations.append(s)
		idx += 1

	return span_representations

def labels_to_attention_span_representations(i, sentence, trainer, args):
	#TODO
	return

"""
USING TRF1 PREDICTIONS
"""

def adjust_predictions():
	# Saves predictions in a CoNLL-friendly format in parent path
	# Output data is ready for conlleval.pl script

	parent_path = "/home/getalp/sfeirj/data/"
	# Build dataframe
	predictions = pd.read_csv("{}{}_predictions".format(parent_path, set), delimiter="\\t", \
		names=["source","target","prediction"])

	conll_list = []

	for row_idx,row in tqdm(predictions.iterrows()):
		splitted_source = row["source"].split(" ")
		splitted_target = row["target"].split(" ")
		splitted_prediction = row["prediction"].split(" ")
	    
	    # If prediction and target have same length
		if len(splitted_target) == len(splitted_prediction):
			for idx in range(len(splitted_target)):
				conll_list.append([row_idx, splitted_source[idx], \
	    									splitted_target[idx], splitted_prediction[idx]])
	            
	    # If target is longer
		elif len(splitted_target) > len(splitted_prediction):
			for idx in range(len(splitted_prediction)):
				conll_list.append([row_idx, splitted_source[idx], \
											splitted_target[idx], splitted_prediction[idx]])
	        # Fill prediction with "O" tags
			for idx in range(len(splitted_prediction), len(splitted_target)):
				conll_list.append([row_idx, splitted_source[idx], \
											splitted_target[idx], "O"])
	            
	    # If prediction is longer
		else:
			for idx in range(len(splitted_target)):
				conll_list.append([row_idx, splitted_source[idx], \
											splitted_target[idx], splitted_prediction[idx]])
	    
	    # Write empty line after every sentence
		conll_list.append(["", "", "", ""])

	with open("{}{}_predictions_for_eval".format(parent_path, set), "w") as f:
		for word_row in conll_list:
			f.write(str(word_row[0]))
			for e in word_row[1:]:
				f.write(" {}".format(str(e)))
			f.write("\n")


def get_length_differences():
	# Creates dataframe containing the sentences whose target and prediction 
	# don't have the same length
	# Returns dataframe containing exclusively wrongly predicted sizes sentences

	df_diff = pd.DataFrame(columns=["sentence_idx","source","target","prediction"])
	cter = 0
	for idx,row in predictions.iterrows():
		src_len = len(row["source"].split(" "))
		pred_len = len(row["prediction"].split(" "))
		if (src_len != pred_len):
			df_diff.loc[cter] = row
			cter += 1
	print("{} predicted sentences among {} are of different length than target".format(cter, \
		len(predictions)))
	return df_diff

"""
USING TRF2 CLUSTERING
"""