
from tqdm import tqdm
import pandas as pd
import torch



def load_firsts(args, set='train'):
	firsts_list = []
	with open("{}/{}.firsts.txt".format(args.data[0], set), "r") as f:
		for first in f.readlines():
			firsts_list.append(int(first[:-1]))
	return firsts_list

def labels_to_simple_span_representations(i, sentence, trainer, args):
	"""Returns the list of span representations of each span in one BIO sentence with id i"""
	# Uses simple representation by averaging
	# It is interesting to use either golden data or predictions
	beg_idx = 0
	end_idx = 0
	idx = 0 # The index which goes through the sentence
	span_representations = []

	while idx < len(sentence):
		if (sentence[idx][0] == 'B'): # 6/7
			beg_idx = idx
			idx += 1
			while (idx < len(sentence)) and (sentence[idx][0] == 'I'): # 5/8
				idx += 1
			end_idx = idx
			raw_left_representation = trainer.encoder_output_list[i][:beg_idx, :]
			raw_span_representation = trainer.encoder_output_list[i][beg_idx:end_idx, :]
			raw_right_representation = trainer.encoder_output_list[i][end_idx:, :]
			# Create span representation from raw span representation:
			# First word, last word, average, average left, average right
			span_representation = torch.cat((
				raw_span_representation[0],
				raw_span_representation[-1],
				torch.mean(raw_span_representation, 0),
				torch.mean(raw_left_representation, 0),
				torch.mean(raw_right_representation, 0),
				), 0)
			print("Size of span rep: {}".format(span_representation.size()))
			span_representations.append(torch.squeeze(span_representation))
		idx += 1

	return span_representations

def labels_to_attention_span_representations(i, sentence, trainer, args):
	#TODO
	return

def labels_to_span_representations(trainer, task, args, use_gold=True, use_attention=False):
	# returns a list containing in each index j the representations of all the mentions contained in the jth sentence
	sentences = []
	data_size = task.dataset('train').tgt.size

	if use_gold:
	    with open("/home/getalp/sfeirj/data/CoNLL/train.label", "r") as f:
	        for line in f.readlines():
	            gold_labels.append(line[:-1].split(" "))
	    assert len(gold_labels) == data_size
	else:
		#TODO sentences = predictions
		assert False

    span_representations = [] * data_size
    for i in range(data_size):
    	if use_attention == False:
			span_representations[i] = labels_to_simple_span_representations(i, sentences[i], trainer, args)
		else:
			span_representations[i] = labels_to_attention_span_representations(i, sentences[i], trainer, args)

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

