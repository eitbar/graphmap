# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from torch import normal
import torch
from graph_utils import calc_csls_sim, word_graph
from art_wrapper import run_unsupervised_alignment
from refinement_trainer import Trainer
from evaluator import Evaluator

import embedding_utils
from cupy_utils import *

import argparse
import collections
import numpy as np
import re
import sys
import time
import pickle

def generate_seed_dict(src_vectors, tgt_vectors, build_method='S2T', seed_dict_size=0):

    xp = get_array_module(src_vectors)
    # 计算src -> tgt 和 tgt -> src 方向的最近邻
    if 'S2T' in build_method:
        print("Calculating SRC - TAR NNs ...")
        csls_sim_src = calc_csls_sim(src_vectors, tgt_vectors)
        nn_s2t = list(zip(xp.arange(src_vectors.shape[0]).tolist(), csls_sim_src.argmax(1).tolist()))
    
    if 'T2S' in build_method:
        print("Calculating TAR - SRC NNs ...")
        csls_sim_tgt = calc_csls_sim(tgt_vectors, src_vectors)
        nn_t2s = list(zip(xp.arange(tgt_vectors.shape[0]).tolist(), csls_sim_tgt.argmax(1).tolist()))
    
    if build_method == 'S2T':
        candidates = nn_s2t
    elif build_method == 'T2S':
        candidates = nn_t2s
    elif build_method == 'S2T&T2S':
        candidates = [st_pair for st_pair in nn_s2t if (st_pair[1], st_pair[0]) in nn_t2s]
    else:
        candidates = list(nn_t2s + [(st_pair[1], st_pair[0]) for st_pair in nn_t2s])

    candidates = list(set(candidates))
    # 按照src_word频率 + tgt_word频率排序，取频率最高的
    candidates_sorted = sorted(candidates, key = lambda  t: t[0] + t[1]) # sort by frequency but approximated by ranks in fasttext files

    # 取topN
    if seed_dict_size > 0:
        candidates_sorted = candidates_sorted[:seed_dict_size]

    return candidates_sorted


def main():
    src_input = 'data/wiki.en.vec'
    tgt_input = 'data/wiki.zh.vec'
    train_path = 'data/crosslingual/dictionaries/en-zh.0-5000.txt'
    eval_path = 'data/crosslingual/dictionaries/en-zh.5000-6500.txt'
    src_lang = 'en'
    tgt_lang = 'zh'
    cuda = True
    nonormalize_method = ['unit', 'center', 'unit']
    min_edge_weight = 0.1
    max_neighbor_number = 0
    min_cliques_number = 2
    vocab_size = 10000
    debug = True
    N_iter = 3
    n_refinement = 3


    srcfile = open(src_input, encoding='utf-8', errors='surrogateescape')
    tgtfile = open(tgt_input, encoding='utf-8', errors='surrogateescape')
    dtype = 'float32'
    src_words, src_embedding = embedding_utils.read(srcfile, vocab_size, dtype=dtype)
    tgt_words, tgt_embedding = embedding_utils.read(tgtfile, vocab_size, dtype=dtype)

    if cuda:
        if not supports_cupy():
            print('ERROR: Install CuPy for CUDA support', file=sys.stderr)
            sys.exit(-1)
        xp = get_cupy()
        src_embedding = xp.asarray(src_embedding)
        tgt_embedding = xp.asarray(tgt_embedding)
    else:
        xp = np

    xp.random.seed(2021)

    # Build word to index map
    src_word2ind = {word: i for i, word in enumerate(src_words)}
    tgt_word2ind = {word: i for i, word in enumerate(tgt_words)}
   
    src_ind2word = {i:word for i, word in enumerate(src_words)}
    tgt_ind2word = {i:word for i, word in enumerate(tgt_words)}
   
    for big_t in range(N_iter):
        # STEP 0: Normalization
        embedding_utils.normalize(src_embedding, nonormalize_method)
        embedding_utils.normalize(tgt_embedding, nonormalize_method)

        # STEP 1: Build Graph


        src_word_graph = word_graph(src_embedding, min_edge_weight, max_neighbor_number)
        tgt_word_graph = word_graph(tgt_embedding, min_edge_weight, max_neighbor_number)

        # STEP 2: Get Clique Center

        src_clique_center_word_index = src_word_graph.get_clique_center_lists(min_cliques_number)
        tgt_clique_center_word_index = tgt_word_graph.get_clique_center_lists(min_cliques_number)

        src_clique_center_word_embs = src_embedding[src_clique_center_word_index]
        tgt_clique_center_word_embs = tgt_embedding[tgt_clique_center_word_index]

        print(len(src_clique_center_word_index))
        print(len(tgt_clique_center_word_embs))

        if debug:
            print('clique words:')
            print('src lang:')
            src_clique_list = src_word_graph.cliques_list
            for i in range(min(10, len(src_clique_list))):
                print("clique ", i)
                print([src_ind2word[ind] for ind in src_clique_list[i]])
                print(src_ind2word[src_clique_center_word_index[i]])

            print('tgt lang:')
            tgt_clique_list = tgt_word_graph.cliques_list
            for i in range(min(10, len(tgt_clique_list))):
                print("clique ", i)
                print([tgt_ind2word[ind] for ind in tgt_clique_list[i]])
                print(tgt_ind2word[tgt_clique_center_word_index[i]])

        # STEP 3: Follow Artetxe et al. vecmap alignment
        src_clique_center_word_list = [src_ind2word[ind] for ind in src_clique_center_word_index]
        tgt_clique_center_word_list = [tgt_ind2word[ind] for ind in tgt_clique_center_word_index]
        """
        xw, zw = run_unsupervised_alignment(src_words, 
                                            tgt_words, 
                                            src_embedding, 
                                            tgt_embedding, 
                                            cuda)
        """
        xw, zw = run_unsupervised_alignment(src_clique_center_word_list, 
                                            tgt_clique_center_word_list, 
                                            src_clique_center_word_embs, 
                                            tgt_clique_center_word_embs, 
                                            cuda)


        # STEP 4: get seed dict
        clique_seed_dict = generate_seed_dict(xw, zw)
        word_seed_dict = [(src_clique_center_word_index[s], tgt_clique_center_word_index[t]) for s, t in clique_seed_dict]
        #word_seed_dict = [(s, t) for s, t in clique_seed_dict]
        word_seed_dict = list(set(word_seed_dict))
        
        
        # for debug
        print(len(word_seed_dict))
        if debug:
            print("seed dict:")
            for si, ti in word_seed_dict[:10]:
                print(src_ind2word[si], tgt_ind2word[ti])

        
        # STEP 5: refinement


        parser = argparse.ArgumentParser(description='Unsupervised training')

        parser.add_argument("--cuda", type=bool, default=True, help="Run on GPU")
        parser.add_argument("--n_refinement", type=int, default=n_refinement, help="Number of refinement iterations (0 to disable the refinement procedure)")
        # dictionary creation parameters (for refinement)

        parser.add_argument("--src_lang", type=str, default=src_lang, help="Source language")
        parser.add_argument("--tgt_lang", type=str, default=tgt_lang, help="Target language")
        parser.add_argument("--dico_eval", type=str, default=eval_path, help="Path to evaluation dictionary")

        parser.add_argument("--dico_method", type=str, default='csls_knn_10', help="Method used for dictionary generation (nn/invsm_beta_30/csls_knn_10)")
        parser.add_argument("--dico_build", type=str, default='S2T&T2S', help="S2T,T2S,S2T|T2S,S2T&T2S")
        parser.add_argument("--dico_threshold", type=float, default=0, help="Threshold confidence for dictionary generation")
        parser.add_argument("--dico_max_rank", type=int, default=15000, help="Maximum dictionary words rank (0 to disable)")
        parser.add_argument("--dico_min_size", type=int, default=0, help="Minimum generated dictionary size (0 to disable)")
        parser.add_argument("--dico_max_size", type=int, default=0, help="Maximum generated dictionary size (0 to disable)")
        args = parser.parse_args()

        mapping = torch.nn.Linear(src_embedding.shape[1], tgt_embedding.shape[1], bias=False)

        trainer = Trainer(src_embedding, tgt_embedding, src_word2ind, tgt_word2ind, mapping, args)

        evaluator = Evaluator(trainer)

        for n_iter in range(args.n_refinement):

            # build dictionary
            if n_iter == 0:
                trainer.load_initial_dico(word_seed_dict)
            else:
                trainer.build_dictionary()

            print(trainer.dico.shape)

            # apply the Procrustes solution
            trainer.procrustes()

            evaluator.word_translation()

        W = xp.asarray(trainer.mapping.weight.data.numpy()).T
        new_src_embedding = src_embedding.dot(W)
        src_embedding = new_src_embedding


if __name__ == "__main__":
    main()
