import collections
import math

import torch


def bleu_for_sentence(pred_seq, label_seq, k=3):
    """compute BLEU score for two sentence"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / (len_pred+eps)))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1 + 1e-6), math.pow(0.5, n))
    return score


def bleu(pred_seq, label_seq, k=3, eps=1e-6):
    """compute BLEU score for two list of int(token)"""
    if isinstance(pred_seq, torch.Tensor):
        pred_seq = pred_seq.tolist()
    if isinstance(label_seq, torch.Tensor):
        label_seq = label_seq.tolist()
    len_pred, len_label = len(pred_seq), len(label_seq)
    score = math.exp(min(0, 1 - len_label / (len_pred+eps)))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[tuple(label_seq[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[tuple(pred_seq[i: i + n])] > 0:
                num_matches += 1
                label_subs[tuple(pred_seq[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1 + 1e-6), math.pow(0.5, n))
    return score
