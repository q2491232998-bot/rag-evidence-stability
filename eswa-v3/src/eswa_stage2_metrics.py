"""Pre-scoring frozen metric definitions; no fitting and no candidate selection."""
import collections
import html
import re
import string
import unicodedata


def ws(s):return ' '.join(s.split())


def evidence_norm(s):return ws(unicodedata.normalize('NFKC',html.unescape(s))).casefold()


def answer_norm(s):
    s=''.join(c for c in s.lower() if c not in string.punctuation)
    return ws(re.sub(r'\b(a|an|the)\b',' ',s))


def hotpot_answer(pred,gold):
    a,b=answer_norm(pred),answer_norm(gold);em=float(a==b)
    if (a in ['yes','no','noanswer'] or b in ['yes','no','noanswer']) and a!=b:return {'answer_f1':0.,'answer_em':em}
    at,bt=a.split(),b.split();common=sum((collections.Counter(at)&collections.Counter(bt)).values())
    f1=2*common/(len(at)+len(bt)) if common else 0.
    return {'answer_f1':f1,'answer_em':em}


def qasper_evidence(documents,refs,official):
    got={ws(d['text']) for d in documents}
    valid=[{ws(e) for e in r['evidence'] if e.strip()} for r in refs]
    valid=[r for r in valid if r]
    return {'gold_evidence_recall':max(len(got&r)/len(r) for r in valid) if valid else None,
            'gold_metric':'qasper_max_reference_text_paragraph_recall',
            'official_evidence_f1':max(official.paragraph_f1_score(list(dict.fromkeys(d['text'] for d in documents)),r['evidence']) for r in refs),
            'support_title_recall':None}


def hotpot_gold(label):
    paragraphs={title:sentences for title,sentences in label['context']};facts=[]
    for title,index in label['supporting_facts']:
        sentences=paragraphs.get(title,[])
        sentence=sentences[index] if 0<=index<len(sentences) else None
        normalized=evidence_norm(sentence) if sentence is not None else ''
        facts.append({'title':title,'sentence_index':index,'text':sentence,'normalized':normalized,'resolvable':bool(normalized)})
    return facts


def hotpot_evidence(documents,label):
    facts=hotpot_gold(label);titles={evidence_norm(d['title']) for d in documents}
    text_by_title=collections.defaultdict(list)
    for d in documents:text_by_title[evidence_norm(d['title'])].append(evidence_norm(d['text']))
    hits=[int(f['resolvable'] and any(f['normalized'] in t for t in text_by_title[evidence_norm(f['title'])])) for f in facts]
    want={evidence_norm(f['title']) for f in facts}
    return {'gold_evidence_recall':sum(hits)/len(facts) if facts else None,'gold_metric':'hotpot_support_sentence_text_coverage',
            'support_title_recall':len(titles&want)/len(want) if want else None,'official_evidence_f1':None,
            'support_facts_total':len(facts),'support_facts_covered':sum(hits),'support_facts_unresolvable':sum(not f['resolvable'] for f in facts)}
