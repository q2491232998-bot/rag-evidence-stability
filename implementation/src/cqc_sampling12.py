from transformers import LogitsProcessor, LogitsProcessorList, TemperatureLogitsWarper, TopPLogitsWarper


class CpuTopP(LogitsProcessor):
    def __init__(self, probability):
        self.warper = TopPLogitsWarper(probability)

    def __call__(self, input_ids, scores):
        return self.warper(input_ids.cpu(), scores.cpu()).to(scores.device)


def deterministic_sampling(options):
    effective = dict(options)
    if options['do_sample']:
        assert options['top_k'] == 0
        effective['logits_processor'] = LogitsProcessorList([TemperatureLogitsWarper(options['temperature']), CpuTopP(options['top_p'])])
        effective['temperature'] = 1.0
        effective['top_p'] = 1.0
    return effective
