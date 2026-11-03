import importlib
import itertools
import json
import sys

from common import ROOT, atomic_json
from musique_metrics import score_answer, score_support


def main():
    sys.path.insert(0, str(ROOT / 'references/musique_official'))
    official_answers = importlib.import_module('metrics.answer')
    official_supports = importlib.import_module('metrics.support')
    examples = ['', 'the', 'yes', 'no', 'noanswer', 'yes indeed', 'John F. Kennedy', 'JFK', 'Paris, France', 'Paris', 'An orange', 'orange orange', 'Émile Zola']
    count = 0
    for prediction, answer in itertools.product(examples, repeat=2):
        for aliases in [[], ['JFK', 'Paris'], ['', 'the']]:
            reference = official_answers.AnswerMetric()
            reference(prediction, [answer] + aliases)
            em, f1 = reference.get_metric()
            ours = score_answer(prediction, answer, aliases)
            assert abs(ours['em'] - em) < 1e-12 and abs(ours['f1'] - f1) < 1e-12
            count += 1
    support_cases = [[], [0], [1], [0, 1], [1, 1], ['1', '2'], [0, 2, 4]]
    for predicted, reference in itertools.product(support_cases, repeat=2):
        metric = official_supports.SupportMetric()
        metric(predicted, reference)
        em, f1 = metric.get_metric()
        ours = score_support(predicted, reference)
        assert abs(ours['support_em'] - em) < 1e-12 and abs(ours['support_f1'] - f1) < 1e-12
    assert score_answer('JFK', 'John F. Kennedy', ['JFK'])['em'] == 1.
    assert score_answer('', '')['f1'] == 1.
    assert score_answer('yes indeed', 'yes')['f1'] > 0.
    result = {'answer_comparisons': count, 'support_comparisons': len(support_cases) ** 2, 'official_metric_agreement': True, 'official_revision': '922ac98f19a201998dbdae6d7f2887a5258dbdeb', 'third_party_reference_code_kept_unmodified': True}
    atomic_json(ROOT / 'logs/musique_metric_checks.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
