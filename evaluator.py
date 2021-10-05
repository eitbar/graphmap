# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
from copy import deepcopy
import numpy as np
from torch.autograd import Variable
from torch import Tensor as torch_tensor

from word_translation import get_word_translation_accuracy


logger = getLogger()


class Evaluator(object):

    def __init__(self, trainer):
        """
        Initialize evaluator.
        """
        self.src_emb = trainer.src_emb
        self.tgt_emb = trainer.tgt_emb
        self.src_dico = trainer.src_dico
        self.tgt_dico = trainer.tgt_dico
        self.mapping = trainer.mapping
        self.params = trainer.params

    def word_translation(self):
        """
        Evaluation on word translation.
        """
        # mapped word embeddings
        src_emb = self.mapping(self.src_emb.weight).data
        tgt_emb = self.tgt_emb.weight.data

        for method in ['nn', 'csls_knn_10']:
            results = get_word_translation_accuracy(
                self.src_dico, src_emb,
                self.tgt_dico, tgt_emb,
                method=method,
                dico_eval=self.params.dico_eval
            )
            print([('%s-%s' % (k, method), v) for k, v in results])

    def all_eval(self):
        """
        Run all evaluations.
        """
        self.word_translation()
