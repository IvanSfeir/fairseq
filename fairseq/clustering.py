#!/usr/bin/env python3 -u
# Copyright (c) 2019-present, Ivan Sfeir.
# All rights reserved.
"""
This task corresponds to the second part of the coreference resolution architecture which I have
worked on during my internship at GETALP team, LIG, Grenoble, from Feb to Jul 2019.
It inherits from FairseqTask class but it isn't only a task. It also contains the model for the second step.
"""

import itertools
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


from fairseq import options, utils
from fairseq.data import (
    ConcatDataset,
    data_utils,
    Dictionary,
    IndexedCachedDataset,
    IndexedDataset,
    IndexedRawTextDataset,
    LanguagePairDataset,
    my_data_utils,
)

from fairseq.tasks import FairseqTask, register_task



class PredictedCluster():
    """
    Class defining clusters predicted in the second task of the co-training process
    """
    #For now, representation is computed the easy way: by averaging the spans in the cluster

    def __init__(self, **kwargs):

        if len(args) == 0: # initialize an empty cluster
            self.representation = torch.zeros(self.embed_dim * 2)

        elif "span" in kwargs.keys():
            s = kwargs["span"]
            assert type(s) == my_data_utils.Span
            self.spans = [s]
            self.representation = s.representation

    def add_span(self, s):
        assert type(s) == Span
        self.representation = ((self.representation * len(self.spans)) + s.representation) \
                                    / (len(self.spans) + 1)
        self.spans.append(s)

    def get_extended_representation(self):
        return torch.cat((self.representation, self.spans[-1].representation), dim=0)



@register_task('clustering')
class ClusteringTask(FairseqTask):
    """
    Encode spans and constitute coreferent clusters.
    """

    # total number of clusters has been computed via sum of train clusters and two times dev clusters
    self.nb_clusters = 43140
    # generate nb_clusters clusters and trash cluster
    self.predicted_clusters = [PredictedCluster() for i in range(self.nb_clusters + 1)]
    # dictionary of mentions to be used for mention verification
    self.gold_mentions_dict = {}

    def __init__(self, args):

        self.embed_dim = args.encoder_embed_dim
        self.lin_ffnn_1 = nn.Linear(self.embed_dim * 5, self.embed_dim * 4)
        self.lin_ffnn_2 = nn.Linear(self.embed_dim * 4, self.embed_dim * 2)
        self.lin_decision_maker = nn.Linear(self.embed_dim * 6, self.nb_clusters + 1)
        self.lin_loss_f = nn.Linear(self.embed_dim * 2, 1)
        self.lin_loss_g = nn.Linear(self.embed_dim * 4, self.embed_dim * 2)


        self.clusters_attn = MultiheadAttention(
            self.embed_dim * 2, args.encoder_attention_heads,
            dropout=args.attention_dropout,
        )

        #itr = self.prepare_batches_indices(args, set, firsts, data_size)
        #for batch in itr:
            

#    def add_trash_cluster(self):
#        self.predicted_clusters.append(?)

    def get_predicted_clusters(self):
        return self.predicted_clusters

    #function in the model's interface
    def forward(self, span, clusters_list):
        """
        Args:
            span (~my_data_utils.Span): the span to classify

        Returns:
            softmax
        """

        x = span.representation

        clusters = torch.Tensor([c.get_extended_representation() for c in clusters_list])

        x = F.relu(self.lin_ffnn_1(x))
        x = self.lin_ffnn_2(x)

        ffnn_representation = x
        x, _ = self.clusters_attn(query=x, key=clusters, value=clusters)
        x = ffnn_representation + x

        x = F.softmax(self.lin_decision_maker(x), 0)

        return x, ffnn_representation

    #function in the task's interface
    def train_step(self, sample, span_representations, criterion, optimizer, ignore_grad=False):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the model associated to the clustering task and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            span_representations (list): list of span representations for 
                the document at hand
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        beg = min(sample[0]["id"]) # difference between i and sentence idx
        for i in range(len(span_representations)):
            for j in range(len(span_representations[i])):
                # for every span, apply attention over the clusters to make decisions
                x, ffnn_representation = self.forward(span_representations[i][j], self.get_predicted_clusters())
                span_representations[i][j].update_ffnn_representation(ffnn_representation)

        loss, sample_size, logging_output = criterion(model, sample, span_representations)
        if ignore_grad:
            loss *= 0
        #optimizer.backward(loss) #delete and use line right under it without creating an optimizer object
        loss.backward(retain_graph=False)
        return loss, sample_size, logging_output

    #RELATED TO BATCHES AND DATA LOADING

    # USELESS FUNCTION
    def prepare_batches_indices(args, set, firsts, data_size):
        #yields index of first sentence in a batch/document and first sentence in the next one
        beg = firsts[0]
        for first in firsts[1:]:
            yield (beg, first)
            beg = first
        yield (beg, data_size)
