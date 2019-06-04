
from tqdm import tqdm
import pandas as pd
import torch



def load_firsts(args, set='train'):
	firsts_list = []
	with open("{}/{}.firsts.txt".format(args.data[0], set), "r") as f:
		for first in f.readlines():
			firsts_list.append(int(first[:-1]))
	return firsts_list

def labels_to_span_representations(i, sentence, args):
	"""Returns the list of span representations of all spans in one BIO sentence with id i"""
	# It is interesting to use either golden data or predictions
	beg_idx = 0
	end_idx = 0
	idx = 0 # The index which goes through the sentence
	span_representations = []

	while idx < len(sentence):
		if sentence[idx][0] == 'B':
			beg_idx = idx
			idx += 1
			while (idx < len(sentence)) and (sentence[idx][0] == 'I'):
				idx += 1
			end_idx = idx
			span_representations.append(torch.index_select(args.trainer.encoder_output_list(i), \
				 0, torch.tensor(range(beg_idx, end_idx)).to(torch.device("cuda"))))
		idx += 1

	return span_representations

def get_document_representation(i, args):
    """Return ith document"""
    for sentence in sentences:
    	return labels_to_span_representations(i, sentence, args)

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

