# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from logging import getLogger
import scipy
import scipy.linalg
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from cupy_utils import *
from dico_builder import build_dictionary
from word_translation import load_dictionary


logger = getLogger()

def _numpy_to_embedding(x):
    data = torch.from_numpy(asnumpy(x))
    emb = torch.nn.Embedding(x.shape[0], x.shape[1], sparse=True)
    emb.weight.data.copy_(data)
    return emb


class Trainer(object):

    def __init__(self, src_emb, tgt_emb, src_dico, tgt_dico, mapping, params):
        """
        Initialize trainer script.
        """
        self.src_emb = _numpy_to_embedding(src_emb)
        self.tgt_emb = _numpy_to_embedding(tgt_emb)
        self.src_dico = src_dico
        self.tgt_dico = tgt_dico
        self.mapping = mapping

        self.params = params

        # best validation score
        self.best_valid_metric = -1e12

    def load_training_dico(self, dico_train):
        """
        Load training dictionary.
        """
        word2id1 = self.src_dico
        word2id2 = self.tgt_dico

        self.dico = load_dictionary(dico_train, word2id1, word2id2)
        
        # cuda
        if self.params.cuda:
            self.dico = self.dico.cuda()


    def load_initial_dico(self, seed_dico):
        """
        Load training dictionary.
        """

        self.dico = torch.LongTensor(list([[int(a), int(b)] for (a, b) in seed_dico]))

        # cuda
        if self.params.cuda:
            self.dico = self.dico.cuda()

    def build_dictionary(self):
        """
        Build a dictionary from aligned embeddings.
        """
        src_emb = self.mapping(self.src_emb.weight).data
        tgt_emb = self.tgt_emb.weight.data
        src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
        tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)
        self.dico = build_dictionary(src_emb, tgt_emb, self.params)

    def procrustes(self):
        """
        Find the best orthogonal matrix mapping using the Orthogonal Procrustes problem
        https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
        """
        A = self.src_emb.weight.data[self.dico[:, 0]]
        B = self.tgt_emb.weight.data[self.dico[:, 1]]
        W = self.mapping.weight.data
        M = B.transpose(0, 1).mm(A).cpu().numpy()
        U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
        W.copy_(torch.from_numpy(U.dot(V_t)).type_as(W))

    def save_best(self, to_log, metric):
        """
        Save the best model for the given validation metric.
        """
        # best mapping for the given validation criterion
        if to_log[metric] > self.best_valid_metric:
            # new best mapping
            self.best_valid_metric = to_log[metric]
            logger.info('* Best value for "%s": %.5f' % (metric, to_log[metric]))
            # save the mapping
            W = self.mapping.weight.data.cpu().numpy()
            path = os.path.join(self.params.exp_path, 'best_mapping.pth')
            logger.info('* Saving the mapping to %s ...' % path)
            torch.save(W, path)

    def reload_best(self):
        """
        Reload the best mapping.
        """
        path = os.path.join(self.params.exp_path, 'best_mapping.pth')
        logger.info('* Reloading the best model from %s ...' % path)
        # reload the model
        assert os.path.isfile(path)
        to_reload = torch.from_numpy(torch.load(path))
        W = self.mapping.weight.data
        assert to_reload.size() == W.size()
        W.copy_(to_reload.type_as(W))

