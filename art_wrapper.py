# Copyright (C) 2016-2018  Mikel Artetxe <artetxem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import embedding_utils as embeddings
from cupy_utils import *

import argparse
import collections
import numpy as np
import re
import sys
import time


def dropout(m, p):
    if p <= 0.0:
        return m
    else:
        xp = get_array_module(m)
        mask = xp.random.rand(*m.shape) >= p
        return m*mask


def topk_mean(m, k, inplace=False):  # TODO Assuming that axis is 1
    xp = get_array_module(m)
    n = m.shape[0]
    ans = xp.zeros(n, dtype=m.dtype)
    if k <= 0:
        return ans
    if not inplace:
        m = xp.array(m)
    ind0 = xp.arange(n)
    ind1 = xp.empty(n, dtype=int)
    minimum = m.min()
    for i in range(k):
        # 找当前值最大的index
        m.argmax(axis=1, out=ind1)
        ans += m[ind0, ind1]
        # 当值最大的位置改成最小值
        m[ind0, ind1] = minimum
    return ans / k

def run_unsupervised_alignment(src_word_list, tgt_word_list, src_embedding, tgt_embedding, use_cuda):
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Map word embeddings in two languages into a shared space')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('--precision', choices=['fp16', 'fp32', 'fp64'], default='fp32', help='the floating-point precision (defaults to fp32)')
    parser.add_argument('--cuda', action='store_true', help='use cuda (requires cupy)')
    parser.add_argument('--batch_size', default=10000, type=int, help='batch size (defaults to 10000); does not affect results, larger is usually faster but uses more memory')
    parser.add_argument('--seed', type=int, default=0, help='the random seed (defaults to 0)')

    recommended_group = parser.add_argument_group('recommended settings', 'Recommended settings for different scenarios')
    recommended_type = recommended_group.add_mutually_exclusive_group()
    recommended_type.add_argument('--supervised', metavar='DICTIONARY', help='recommended if you have a large training dictionary')
    recommended_type.add_argument('--semi_supervised', metavar='DICTIONARY', help='recommended if you have a small seed dictionary')
    recommended_type.add_argument('--identical', action='store_true', help='recommended if you have no seed dictionary but can rely on identical words')
    recommended_type.add_argument('--unsupervised', action='store_true', help='recommended if you have no seed dictionary and do not want to rely on identical words')
    recommended_type.add_argument('--acl2018', action='store_true', help='reproduce our ACL 2018 system')
    recommended_type.add_argument('--aaai2018', metavar='DICTIONARY', help='reproduce our AAAI 2018 system')
    recommended_type.add_argument('--acl2017', action='store_true', help='reproduce our ACL 2017 system with numeral initialization')
    recommended_type.add_argument('--acl2017_seed', metavar='DICTIONARY', help='reproduce our ACL 2017 system with a seed dictionary')
    recommended_type.add_argument('--emnlp2016', metavar='DICTIONARY', help='reproduce our EMNLP 2016 system')

    init_group = parser.add_argument_group('advanced initialization arguments', 'Advanced initialization arguments')
    init_type = init_group.add_mutually_exclusive_group()
    init_type.add_argument('-d', '--init_dictionary', default=sys.stdin.fileno(), metavar='DICTIONARY', help='the training dictionary file (defaults to stdin)')
    init_type.add_argument('--init_identical', action='store_true', help='use identical words as the seed dictionary')
    init_type.add_argument('--init_numerals', action='store_true', help='use latin numerals (i.e. words matching [0-9]+) as the seed dictionary')
    init_type.add_argument('--init_unsupervised', action='store_true', help='use unsupervised initialization')
    init_group.add_argument('--unsupervised_vocab', type=int, default=0, help='restrict the vocabulary to the top k entries for unsupervised initialization')

    mapping_group = parser.add_argument_group('advanced mapping arguments', 'Advanced embedding mapping arguments')
    mapping_group.add_argument('--normalize', choices=['unit', 'center', 'unitdim', 'centeremb', 'none'], nargs='*', default=[], help='the normalization actions to perform in order')
    mapping_group.add_argument('--whiten', action='store_true', help='whiten the embeddings')
    mapping_group.add_argument('--src_reweight', type=float, default=0, nargs='?', const=1, help='re-weight the source language embeddings')
    mapping_group.add_argument('--trg_reweight', type=float, default=0, nargs='?', const=1, help='re-weight the target language embeddings')
    mapping_group.add_argument('--src_dewhiten', choices=['src', 'trg'], help='de-whiten the source language embeddings')
    mapping_group.add_argument('--trg_dewhiten', choices=['src', 'trg'], help='de-whiten the target language embeddings')
    mapping_group.add_argument('--dim_reduction', type=int, default=0, help='apply dimensionality reduction')
    mapping_type = mapping_group.add_mutually_exclusive_group()
    mapping_type.add_argument('-c', '--orthogonal', action='store_true', help='use orthogonal constrained mapping')
    mapping_type.add_argument('-u', '--unconstrained', action='store_true', help='use unconstrained mapping')

    self_learning_group = parser.add_argument_group('advanced self-learning arguments', 'Advanced arguments for self-learning')
    self_learning_group.add_argument('--self_learning', action='store_true', help='enable self-learning')
    self_learning_group.add_argument('--vocabulary_cutoff', type=int, default=0, help='restrict the vocabulary to the top k entries')
    self_learning_group.add_argument('--direction', choices=['forward', 'backward', 'union'], default='union', help='the direction for dictionary induction (defaults to union)')
    self_learning_group.add_argument('--csls', type=int, nargs='?', default=0, const=10, metavar='NEIGHBORHOOD_SIZE', dest='csls_neighborhood', help='use CSLS for dictionary induction')
    self_learning_group.add_argument('--threshold', default=0.000001, type=float, help='the convergence threshold (defaults to 0.000001)')
    self_learning_group.add_argument('--validation', default=None, metavar='DICTIONARY', help='a dictionary file for validation at each iteration')
    self_learning_group.add_argument('--stochastic_initial', default=0.1, type=float, help='initial keep probability stochastic dictionary induction (defaults to 0.1)')
    self_learning_group.add_argument('--stochastic_multiplier', default=2.0, type=float, help='stochastic dictionary induction multiplier (defaults to 2.0)')
    self_learning_group.add_argument('--stochastic_interval', default=50, type=int, help='stochastic dictionary induction interval (defaults to 50)')
    self_learning_group.add_argument('--log', help='write to a log file in tsv format at each iteration')
    self_learning_group.add_argument('-v', '--verbose', action='store_true', help='write log information to stderr at each iteration')
    args = parser.parse_args()
    
    # 无监督用的这个参数
    parser.set_defaults(init_unsupervised=True, 
                        unsupervised_vocab=4000, 
                        normalize=['unit', 'center', 'unit'], 
                        whiten=True, 
                        src_reweight=0.5, 
                        trg_reweight=0.5, 
                        src_dewhiten='src', 
                        trg_dewhiten='trg', 
                        self_learning=True, 
                        vocabulary_cutoff=20000, 
                        csls_neighborhood=10)

    args = parser.parse_args()

    # Check command line arguments
    if (args.src_dewhiten is not None or args.trg_dewhiten is not None) and not args.whiten:
        print('ERROR: De-whitening requires whitening first', file=sys.stderr)
        sys.exit(-1)

    # Choose the right dtype for the desired precision
    if args.precision == 'fp16':
        dtype = 'float16'
    elif args.precision == 'fp32':
        dtype = 'float32'
    elif args.precision == 'fp64':
        dtype = 'float64'

    # Read input embeddings
    # srcfile = open(args.src_input, encoding=args.encoding, errors='surrogateescape')
    # trgfile = open(args.trg_input, encoding=args.encoding, errors='surrogateescape')
    
    # src_words 是单词list，x是embedding矩阵
    src_words = src_word_list
    trg_words = tgt_word_list
    x = src_embedding
    z = tgt_embedding
    # src_words, x = embeddings.read(srcfile, dtype=dtype)
    # trg_words, z = embeddings.read(trgfile, dtype=dtype)

    # NumPy/CuPy management
    # 用CuPy处理x, z
    xp = get_array_module(x)
    xp.random.seed(args.seed)

    # Build word to index map
    src_word2ind = {word: i for i, word in enumerate(src_words)}
    trg_word2ind = {word: i for i, word in enumerate(trg_words)}

    # STEP 0: Normalization
    # normalize=['unit', 'center', 'unit']
    embeddings.normalize(x, args.normalize)
    embeddings.normalize(z, args.normalize)

    # Build the seed dictionary
    src_indices = []
    trg_indices = []
    if args.init_unsupervised:
        # sim_size 4000
        sim_size = min(x.shape[0], z.shape[0]) if args.unsupervised_vocab <= 0 else min(x.shape[0], z.shape[0], args.unsupervised_vocab)
        # size(x[:sim_size]) = n * d
        # size(u) = n * d size(s) = d * d size(vt) = d * d
        u, s, vt = xp.linalg.svd(x[:sim_size], full_matrices=False)
        # u * s ?
        xsim = (u*s).dot(u.T)
        u, s, vt = xp.linalg.svd(z[:sim_size], full_matrices=False)
        zsim = (u*s).dot(u.T)
        del u, s, vt
        xsim.sort(axis=1)
        zsim.sort(axis=1)
        embeddings.normalize(xsim, args.normalize)
        embeddings.normalize(zsim, args.normalize)
        sim = xsim.dot(zsim.T)

        # csls_neighborhood = 10
        if args.csls_neighborhood > 0:
            # size n * 1
            knn_sim_fwd = topk_mean(sim, k=args.csls_neighborhood)
            # 反向再找一次
            knn_sim_bwd = topk_mean(sim.T, k=args.csls_neighborhood)
            # n * n csls相似度矩阵
            sim -= knn_sim_fwd[:, xp.newaxis]/2 + knn_sim_bwd/2
        if args.direction == 'forward':
            src_indices = xp.arange(sim_size)
            trg_indices = sim.argmax(axis=1)
        elif args.direction == 'backward':
            src_indices = sim.argmax(axis=0)
            trg_indices = xp.arange(sim_size)
        elif args.direction == 'union':
            # 双向求并集
            # y->x (arange, x_index)
            src_indices = xp.concatenate((xp.arange(sim_size), sim.argmax(axis=0)))
            # x->y (y_index, arange)
            trg_indices = xp.concatenate((sim.argmax(axis=1), xp.arange(sim_size)))
            # src_indices 和 trg_indices 是一一对应的关系表示src的翻译是trg
        del xsim, zsim, sim
    elif args.init_numerals:
        numeral_regex = re.compile('^[0-9]+$')
        src_numerals = {word for word in src_words if numeral_regex.match(word) is not None}
        trg_numerals = {word for word in trg_words if numeral_regex.match(word) is not None}
        numerals = src_numerals.intersection(trg_numerals)
        for word in numerals:
            src_indices.append(src_word2ind[word])
            trg_indices.append(trg_word2ind[word])
    elif args.init_identical:
        identical = set(src_words).intersection(set(trg_words))
        for word in identical:
            src_indices.append(src_word2ind[word])
            trg_indices.append(trg_word2ind[word])
    else:
        f = open(args.init_dictionary, encoding=args.encoding, errors='surrogateescape')
        for line in f:
            src, trg = line.split()
            try:
                src_ind = src_word2ind[src]
                trg_ind = trg_word2ind[trg]
                src_indices.append(src_ind)
                trg_indices.append(trg_ind)
            except KeyError:
                print('WARNING: OOV dictionary entry ({0} - {1})'.format(src, trg), file=sys.stderr)



    # Allocate memory
    xw = xp.empty_like(x)
    zw = xp.empty_like(z)

    # vocabulary_cutoff 20000
    src_size = x.shape[0] if args.vocabulary_cutoff <= 0 else min(x.shape[0], args.vocabulary_cutoff)
    trg_size = z.shape[0] if args.vocabulary_cutoff <= 0 else min(z.shape[0], args.vocabulary_cutoff)

    # size bs * n
    simfwd = xp.empty((args.batch_size, trg_size), dtype=dtype)
    simbwd = xp.empty((args.batch_size, src_size), dtype=dtype)


    # 1 * n 
    best_sim_forward = xp.full(src_size, -100, dtype=dtype)
    src_indices_forward = xp.arange(src_size)
    trg_indices_forward = xp.zeros(src_size, dtype=int)
    best_sim_backward = xp.full(trg_size, -100, dtype=dtype)
    src_indices_backward = xp.zeros(trg_size, dtype=int)
    trg_indices_backward = xp.arange(trg_size)

    knn_sim_fwd = xp.zeros(src_size, dtype=dtype)
    knn_sim_bwd = xp.zeros(trg_size, dtype=dtype)

    # Training loop
    best_objective = objective = -100.
    it = 1
    last_improvement = 0
    keep_prob = args.stochastic_initial
    t = time.time()
    # args.self_learning true
    end = not args.self_learning
    while True:

        # Increase the keep probability if we have not improve in args.stochastic_interval iterations
        # stochastic_interval  50
        # 这个意思是，最近50次都没有提升的话，就进入这个条件分支
        if it - last_improvement > args.stochastic_interval:
            if keep_prob >= 1.0:
                end = True
            # stochastic_multiplier 2
            # 调高keep_prob为原来的2倍，然后重新开始看是否有提升
            print(keep_prob)
            keep_prob = min(1.0, args.stochastic_multiplier*keep_prob)
            last_improvement = it

        # Update the embedding mapping
        if args.orthogonal or not end:  # orthogonal mapping

            # 根据之前构造的dict，计算新的w
            # size(w) = (2n * d)^T * (2n * d) = d * d
            u, s, vt = xp.linalg.svd(z[trg_indices].T.dot(x[src_indices]))
            w = vt.T.dot(u.T)

            # xw是x * w
            x.dot(w, out=xw)
            
            zw[:] = z
        elif args.unconstrained:  # unconstrained mapping
            x_pseudoinv = xp.linalg.inv(x[src_indices].T.dot(x[src_indices])).dot(x[src_indices].T)
            w = x_pseudoinv.dot(z[trg_indices])
            x.dot(w, out=xw)
            zw[:] = z
        else:  # advanced mapping
            # end = true整个迭代过程结束的时候会进入这个分支

            # TODO xw.dot(wx2, out=xw) and alike not working
            xw[:] = x
            zw[:] = z

            # STEP 1: Whitening
            def whitening_transformation(m):
                u, s, vt = xp.linalg.svd(m, full_matrices=False)
                return vt.T.dot(xp.diag(1/s)).dot(vt)

            # whiten true
            if args.whiten:
                wx1 = whitening_transformation(xw[src_indices])
                wz1 = whitening_transformation(zw[trg_indices])
                xw = xw.dot(wx1)
                zw = zw.dot(wz1)

            # STEP 2: Orthogonal mapping
            wx2, s, wz2_t = xp.linalg.svd(xw[src_indices].T.dot(zw[trg_indices]))
            wz2 = wz2_t.T
            xw = xw.dot(wx2)
            zw = zw.dot(wz2)

            # STEP 3: Re-weighting
            # src_reweight 0.5
            xw *= s**args.src_reweight
            zw *= s**args.trg_reweight

            # STEP 4: De-whitening
            if args.src_dewhiten == 'src':
                xw = xw.dot(wx2.T.dot(xp.linalg.inv(wx1)).dot(wx2))
            elif args.src_dewhiten == 'trg':
                xw = xw.dot(wz2.T.dot(xp.linalg.inv(wz1)).dot(wz2))
            if args.trg_dewhiten == 'src':
                zw = zw.dot(wx2.T.dot(xp.linalg.inv(wx1)).dot(wx2))
            elif args.trg_dewhiten == 'trg':
                zw = zw.dot(wz2.T.dot(xp.linalg.inv(wz1)).dot(wz2))

            # STEP 5: Dimensionality reduction
            # dim_reduction 0 没有做降维
            if args.dim_reduction > 0:
                xw = xw[:, :args.dim_reduction]
                zw = zw[:, :args.dim_reduction]

        # Self-learning
        if end:
            break
        else:
            # Update the training dictionary
            if args.direction in ('forward', 'union'):
                if args.csls_neighborhood > 0:
                    # 按照batch_size来更新y->x的csls相似度矩阵，这步是在计算公式种的tao y
                    for i in range(0, trg_size, simbwd.shape[0]):
                        j = min(i + simbwd.shape[0], trg_size)
                        zw[i:j].dot(xw[:src_size].T, out=simbwd[:j-i])
                        knn_sim_bwd[i:j] = topk_mean(simbwd[:j-i], k=args.csls_neighborhood, inplace=True)
                
                for i in range(0, src_size, simfwd.shape[0]):
                    j = min(i + simfwd.shape[0], src_size)
                    # 按照batch_size来更新x->y的相似度矩阵
                    xw[i:j].dot(zw[:trg_size].T, out=simfwd[:j-i])
                    # best_sim_forward 用于保存xw->y每个单词最高的相似度分数
                    simfwd[:j-i].max(axis=1, out=best_sim_forward[i:j])
                    simfwd[:j-i] -= knn_sim_bwd/2  # Equivalent to the real CSLS scores for NN

                    # 随机dropout掉一些相似度变为0，更新dict tgt index
                    dropout(simfwd[:j-i], 1 - keep_prob).argmax(axis=1, out=trg_indices_forward[i:j])
            if args.direction in ('backward', 'union'):
                if args.csls_neighborhood > 0:
                    for i in range(0, src_size, simfwd.shape[0]):
                        j = min(i + simfwd.shape[0], src_size)
                        xw[i:j].dot(zw[:trg_size].T, out=simfwd[:j-i])
                        knn_sim_fwd[i:j] = topk_mean(simfwd[:j-i], k=args.csls_neighborhood, inplace=True)
                for i in range(0, trg_size, simbwd.shape[0]):
                    j = min(i + simbwd.shape[0], trg_size)
                    zw[i:j].dot(xw[:src_size].T, out=simbwd[:j-i])
                    simbwd[:j-i].max(axis=1, out=best_sim_backward[i:j])
                    simbwd[:j-i] -= knn_sim_fwd/2  # Equivalent to the real CSLS scores for NN
                    dropout(simbwd[:j-i], 1 - keep_prob).argmax(axis=1, out=src_indices_backward[i:j])
            if args.direction == 'forward':
                src_indices = src_indices_forward
                trg_indices = trg_indices_forward
            elif args.direction == 'backward':
                src_indices = src_indices_backward
                trg_indices = trg_indices_backward
            elif args.direction == 'union':
                src_indices = xp.concatenate((src_indices_forward, src_indices_backward))
                trg_indices = xp.concatenate((trg_indices_forward, trg_indices_backward))

            # Objective function evaluation
            if args.direction == 'forward':
                objective = xp.mean(best_sim_forward).tolist()
            elif args.direction == 'backward':
                objective = xp.mean(best_sim_backward).tolist()
            elif args.direction == 'union':
                objective = (xp.mean(best_sim_forward) + xp.mean(best_sim_backward)).tolist() / 2
            
            # 判断平均cos相似度是否更高，差距大于0.000001的话认位模型更好，迭代成功
            if objective - best_objective >= args.threshold:
                last_improvement = it
                best_objective = objective

            # Logging
            duration = time.time() - t

        t = time.time()
        it += 1

    return xw, zw
