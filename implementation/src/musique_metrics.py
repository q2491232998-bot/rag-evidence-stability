from collections import Counter

from common import normalize


def score_answer(prediction, answer, aliases=()):
    reference_answers = [answer] + list(aliases)
    p = normalize(prediction)
    prediction_tokens = p.split()
    exact = []
    f1_values = []
    for reference in reference_answers:
        g = normalize(reference)
        gold_tokens = g.split()
        exact.append(float(p == g))
        if not prediction_tokens or not gold_tokens:
            f1_values.append(float(prediction_tokens == gold_tokens))
            continue
        matches = sum((Counter(prediction_tokens) & Counter(gold_tokens)).values())
        if matches == 0:
            f1_values.append(0.)
            continue
        precision = matches / len(prediction_tokens)
        recall = matches / len(gold_tokens)
        f1_values.append(2 * precision * recall / (precision + recall))
    return {'em': max(exact), 'f1': max(f1_values)}


def score_support(predicted, reference):
    p, g = set(map(int, predicted)), set(map(int, reference))
    if not p and not g:
        return {'support_em': 1., 'support_f1': 1.}
    matches = len(p & g)
    precision = matches / len(p) if p else 0.
    recall = matches / len(g) if g else 0.
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    return {'support_em': float(p == g), 'support_f1': f1}
