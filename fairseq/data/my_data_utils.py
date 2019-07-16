#!/usr/bin/env python3 -u
# Copyright (c) 2019-present, Ivan Sfeir.
# All rights reserved.
"""
Library for processing raw data in the CoNLL-2012 format, ie. for coreference resolution.
Code written during my internship at GETALP team, LIG, Grenoble, from Feb to Jul 2019.
"""

import pandas as pd
from collections import defaultdict, Counter
import itertools
import re


sets = ["dev", "test", "train"] #my datasets labels
splits = ["valid", "test", "train"] #fairseq's datasets labels
sides = ["input", "label"]



class Span():

    id_generator = itertools.count()

    def __init__(self, **kwargs):
        self.id = next(Span.id_generator)
        self.sentence_id = kwargs["sentence_id"]
        self.first = kwargs["first"]
        self.length = kwargs["length"]
        if (len(kwargs.items())) == 4:
            self.string = kwargs["string"]

    def update_representation(self, r):
        self.representation = r

    def update_ffnn_representation(self, r):
        self.ffnn_representation = r

    def add_antecedent(self, s):
        self.antecedent = s.id

    def no_antecedent(self):
        self.antecedent = None

    def reinitialize(self):
        Span.id_generator = itertools.count()


def read_conll_format_csv(data_source, split):
    return pd.read_csv("/data1/home/getalp/sfeirj/data/{}/csv/{}.csv".format(data_source, split))


def save_firsts(data_source="CoNLL", bin="maxmentions-bin"):
    #save the indices of the first sentence in each document or part

    firsts_list = [[], [], []]

    for i in range(3): #split index
        
        split_dataframe = read_conll_format_csv(data_source, sets[i])
        df_size = split_dataframe.shape[0]
        sentence_ctr = 0
        k = 0

        firsts_list[i].append(k)
        k += 1

        while k < df_size - 1:
            #no need to worry about last line, as no sentence is made of a single line
            
            #if end of sentence
            if (split_dataframe.loc[k+1, "col2"] == 0):
                sentence_ctr += 1
            
            #if end of a part or a document
            if (split_dataframe.loc[k, "col0"] != split_dataframe.loc[k+1, "col0"]) or \
            (split_dataframe.loc[k, "col1"] != split_dataframe.loc[k+1, "col1"]): #different doc or part ==> new first document line

                firsts_list[i].append(sentence_ctr)
            
            k += 1

    for i in range(3):
        with open("/data1/home/getalp/sfeirj/sfeirseq/{}/{}.firsts.txt".format(bin, splits[i]), "w") as f:
            for first in firsts_list[i]:
                f.write("{}\n".format(first))


def compute_gold_clusters_from_lists(l,  l_docs, l_parts, l_wordids, l_strings, data_source="CoNLL"):
    #function which translates last column annotation for coreference resolution to dictionary of coreferent mentions
    #INPUT: list of CoNLL format annotations along with their corresponding docs, parts, word id features and strings
    #OUTPUT: gold_clusters: list containing for each document, its entities and their coreferent mentions
    #note that this function computes previous coreferent spans as gold antecedents
    
    gold_clusters = []
    i = 0
    is_in_mention = False
    entity_id = -1
    sentence_id = 0
    nb_stacked_mentions = 0
    document_clusters = defaultdict(list)
    beginnings_re = '\([0-9]+'
    numbers_re = '[0-9]+'
    nb = 0
    
    while i < len(l):

        if l[i] != "-":
            for beginning in re.findall(beginnings_re, l[i]): #loop over mention beginnings
                #find entity id
                entity_id = re.findall(numbers_re, beginning)[0]
                #look for mention end (j being its index)
                is_in_mention = True
                j = i #j has to get outside the mention
                #computing nb_stacked_mentions is different for the annotation of the first word of each mention
                popped_annotation = l[i][l[i].find(entity_id):]
                nb_stacked_mentions = 1 + len(popped_annotation.split("(")) - len(popped_annotation.split(")"))
                j += 1
                if nb_stacked_mentions <= 0: #found one-word mention
                    is_in_mention = False
                while is_in_mention:
                    nb_stacked_mentions += len(l[j].split("(")) - len(l[j].split(")"))
                    j += 1
                    if nb_stacked_mentions <= 0: #stop condition: add mention to output dictionaries
                        is_in_mention = False
                span_string = " ".join(l_strings[i:j])
                s = Span(sentence_id=sentence_id, first=l_wordids[i], length=(j - i), string=span_string)
                #leme
                print(s.sentence_id, s.first, s.length)
                nb += 1
                #leme
                document_clusters[int(entity_id)].append(s)
                #add previous attribute to current span's antecedent attribute
                if len(document_clusters[int(entity_id)]) == 1:
                    s.no_antecedent()
                else:
                    s.add_antecedent(document_clusters[int(entity_id)][-2])
        
        #check if end of document
        if i < len(l) - 1:
            #if new document or new part
            if (l_docs[i] != l_docs[i + 1]) or (l_parts[i] != l_parts[i + 1]):
                gold_clusters.append(document_clusters)
                document_clusters = defaultdict(list)
        else:
            gold_clusters.append(document_clusters)

        #check if end of sentence
        if (i < len(l) - 1) and (l_wordids[i + 1] == 0):
            sentence_id += 1

        i += 1

    print(nb, " gold mentions extracted")

    return gold_clusters


def compute_gold_max_clusters_from_lists(l,  l_docs, l_parts, l_wordids, l_strings, data_source="CoNLL"):
    #function which translates last column annotation for coreference resolution to dictionary of coreferent max mentions
    #INPUT: list of CoNLL format annotations along with their corresponding docs, parts, word id features and strings
    #OUTPUT: gold_clusters: list containing for each document, its entities and their coreferent max mentions
    #note that this function does not compute gold antecedents
    
    gold_clusters = []
    i = 0
    is_in_max_mention = False
    entity_id = -1
    sentence_id = 0
    nb_stacked_mentions = 0
    document_clusters = defaultdict(list)
    beginnings_re = '\([0-9]+'
    numbers_re = '[0-9]+'
    nb = 0

    while i < len(l):

        if l[i] != "-": #found a max mention beginning
            is_in_max_mention = True
            #find entity id
            entity_id = re.findall(numbers_re, l[i])[0]
            j = i #j has to get outside the max mention
            while is_in_max_mention and j < len(l) - 1:
                nb_stacked_mentions += len(l[j].split("(")) - len(l[j].split(")"))
                is_in_max_mention = (nb_stacked_mentions != 0)
                j += 1
            span_string = " ".join(str(l_strings[i:j]))
            s = Span(sentence_id=sentence_id, first=l_wordids[i], length=(j - i), string=span_string)
            #leme
            print(s.sentence_id, s.first, s.length)
            nb += 1
            #leme
            document_clusters[int(entity_id)].append(s)
            i = j - 1
    
        #check if end of document
        if i < len(l) - 1:
            #if new document or new part
            if (l_docs[i] != l_docs[i + 1]) or (l_parts[i] != l_parts[i + 1]):
                gold_clusters.append(document_clusters)
                document_clusters = defaultdict(list)
        else:
            gold_clusters.append(document_clusters)

        #check if end of sentence
        if (i < len(l) - 1) and (l_wordids[i + 1] == 0):
            sentence_id += 1

        i += 1

    print(nb, " gold maximum mentions extracted")

    return gold_clusters


def compute_gold_clusters(split="train", data_source="CoNLL", use_only_max_mentions=True):
    df = read_conll_format_csv(data_source, split)
    #print(df.head())
    if use_only_max_mentions:
        gold_clusters = compute_gold_max_clusters_from_lists(list(df["col11"]), list(df["col0"]), list(df["col1"]), \
            list(df["col2"]), list(df["col3"]), data_source)
    else:    
        gold_clusters = compute_gold_clusters_from_lists(list(df["col11"]), list(df["col0"]), list(df["col1"]), \
            list(df["col2"]), list(df["col3"]), data_source)
    return gold_clusters

