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
)

from . import FairseqTask, register_task



class PredictedCluster():
    """
    Class defining clusters predicted in the second task of the co-training process
    """



@register_task('clustering')
class ClusteringTask(FairseqTask):
    """
    Encode spans and constitute coreferent clusters.
    """

    self.predicted_clusters = []

    def __init__(self, args, set, firsts, data_size):
        itr = self.prepare_batches_indices(args, set, firsts, data_size)
        for batch in itr:


    def get_predicted_clusters(self):
        return self.predicted_clusters

    #RELATED TO BATCHES AND DATA LOADING

    def prepare_batches_indices(args, set, firsts, data_size):
        beg = firsts[0]
        for first in firsts[1:]:
            yield (beg, first)
            beg = first
        yield (beg, data_size)

    #WHAT WE HAVE LEFT FROM TRANSLATION TASK

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', nargs='+', help='path(s) to data directorie(s)')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        # fmt: on

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    def load_dataset(self, split, combine=False, **kwargs):
        return


