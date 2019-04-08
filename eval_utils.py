from collections import Counter
from utils.bleu_metric.bleu import Bleu
from utils.rouge_metric.rouge import Rouge

def compute_bleu_rouge(pred_dict, ref_dict, bleu_order=4):
    """
    Compute bleu and rouge scores.
    """
    assert set(pred_dict.keys()) == set(ref_dict.keys()), \
        "missing keys: {}".format(set(ref_dict.keys()) - set(pred_dict.keys()))
    scores = {}
    bleu_scores, _ = Bleu(bleu_order).compute_score(ref_dict, pred_dict)
    for i, bleu_score in enumerate(bleu_scores):
        scores['Bleu-%d' % (i + 1)] = bleu_score
    rouge_score, _ = Rouge().compute_score(ref_dict, pred_dict)
    scores['Rouge-L'] = rouge_score
    return scores


def find_best_answer(sample, start_prob, end_prob):
    #  start_prob: max_passage_num x padded_p_len
    """
    Finds the best answer for a sample given start_prob and end_prob for each position.
    This will call find_best_answer_for_passage because there are multiple passages in a sample
    """
    max_p_len = 500
    best_p_idx, best_span, best_score = None, None, 0
    for p_idx, passage in enumerate(sample['passages']):
        if p_idx >= start_prob.size(0):
            continue
        passage_len = min(max_p_len, len(passage['passage_tokens']))
        answer_span, score = find_best_answer_for_passage(start_prob[p_idx], end_prob[p_idx], passage_len)
        if score > best_score:
            best_score = score
            best_p_idx = p_idx
            best_span = answer_span
    if best_p_idx is None or best_span is None:
        best_answer = ''
    else:
        best_answer = ''.join(
            sample['passages'][best_p_idx]['passage_tokens'][best_span[0]: best_span[1] + 1])
    return best_answer, best_p_idx


def find_best_answer_for_passage(start_probs, end_probs, passage_len=None):
    """
    Finds the best answer with the maximum start_prob * end_prob from a single passage
    """
    max_a_len = 200
    if passage_len is None:
        passage_len = len(start_probs)
    else:
        passage_len = min(len(start_probs), passage_len)
    best_start, best_end, max_prob = -1, -1, 0
    for start_idx in range(passage_len):
        for ans_len in range(max_a_len):
            end_idx = start_idx + ans_len
            if end_idx >= passage_len:
                continue
            prob = start_probs[start_idx] * end_probs[end_idx]
            # print("prob.data[0] is ")
            # print(prob.data[0])
            # print("max_prob is ")
            # print(max_prob)
            if prob.data[0] > max_prob:
                best_start = start_idx
                best_end = end_idx
                max_prob = prob.data[0]
    return (best_start, best_end), max_prob


def normalize(s):
    """
    Normalize strings to space joined chars.

    Args:
        s: a list of strings.

    Returns:
        A list of normalized strings.
    """
    if not s:
        return s
    normalized = []
    for ss in s:
        tokens = [c for c in list(ss) if len(c.strip()) != 0]
        normalized.append(' '.join(tokens))
    return normalized
