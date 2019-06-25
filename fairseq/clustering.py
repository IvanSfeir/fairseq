#!/usr/bin/env python3 -u
# Copyright (c) 2019-present, Ivan Sfeir.
# All rights reserved.
"""
This task corresponds to the second port of the coreference resolution architecture which I have
worked on during my internship at GETALP team, LIG, Grenoble, from Feb to Jul 2019.
"""

import itertools
import os

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

    def __init__(self, s):
        assert type(s) == Span
        self.spans = [s]
        self.representation = s.representation

    def add_span(self, s):
        assert type(s) == Span
        self.representation = ((self.representation * len(self.spans)) + s.representation) \
                                    / (len(self.spans) + 1)
        self.spans.append(s)



@register_task('clustering')
class ClusteringTask(FairseqTask):
    """
    Encode spans and constitute coreferent clusters.
    """

    self.predicted_clusters = []

    def __init__(self, args, set, firsts, data_size):
        itr = self.prepare_batches_indices(args, set, firsts, data_size)
        for batch in itr:

#    def add_trash_cluster(self):
#        self.predicted_clusters.append(?)

    def get_predicted_clusters(self):
        return self.predicted_clusters

    def add_cluster(self, c):
        self.predicted_clusters.append(c)


    #RELATED TO BATCHES AND DATA LOADING

    def prepare_batches_indices(args, set, firsts, data_size):
        beg = firsts[0]
        for first in firsts[1:]:
            yield (beg, first)
            beg = first
        yield (beg, data_size)

    def load_dataset(self, split, combine=False, **kwargs):
        return


