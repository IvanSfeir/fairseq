#!/usr/bin/env python3 -u
# Copyright (c) 2019-present, Ivan Sfeir.
# All rights reserved.
"""
Adapted implementation of Wiseman (2016)'s loss for coreference resolution.
Code written during my internship at GETALP team, LIG, Grenoble, from Feb to Jul 2019.
"""


from . import FairseqCriterion, register_criterion
import torch


@register_criterion('wiseman_loss')
class WisemanLoss(FairseqCriterion):
    """This is an implementation of the loss function accompanying the coreference resolution problem,
    described in the paper "Learning Global Features for Coreference Resolution"
    (http://arxiv.org/abs/1604.03035)."""

    def __init__(self, args, task):
        self.args = args

    def f(self, s_rep, t_rep, layer):
        """Compute the f function for two spans s and t using their vectorial representations
        and layer"""
        cat_rep = torch.cat((s_rep, t_rep))
        return layer(cat_rep)

    def g(self, s_rep, c_rep, layer):
        """Compute the g function for span s and cluster c using their vectorial representations
        and layer"""
        m = nn.Tanh()
        tanh_rep = m(s_rep)
        return torch.mm(tanh_rep, layer(c_rep))

    def forward(self, model, sample, span_representations):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        net_output = model(**sample['net_input'])
        orig_target = model.get_targets(sample, net_output)

        nsentences = orig_target.size(0)
        orig_target = orig_target.view(-1)

        bsz = orig_target.size(0)

        logits, target = adaptive_softmax(net_output[0], orig_target)
        assert len(target) == len(logits)

        loss = net_output[0].new(1 if reduce else bsz).zero_()

        for i in range(len(target)):
            if target[i] is not None:
                assert (target[i].min() >= 0 and target[i].max() <= logits[i].size(1))
                loss += F.cross_entropy(logits[i], target[i], size_average=False, ignore_index=self.padding_idx,
                                        reduce=reduce)

        for i in range(len(span_representations)):
            for j in range(len(span_representations[i])):
                

        orig = utils.strip_pad(orig_target, self.padding_idx)
        ntokens = orig.numel()
        sample_size = sample['target'].size(0) if self.args.sentence_avg else ntokens
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output