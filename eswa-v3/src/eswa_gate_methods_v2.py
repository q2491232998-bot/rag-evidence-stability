"""Frozen pure selection rules; no tensors, model state, labels or cached serial scores."""
import itertools


def order(scores):return sorted(range(len(scores)),key=lambda i:(-scores[i],i))


def flagged(scores, already, packer, tau, version):
    ranking=order(scores);base=packer(ranking)['context'];window=min(12,len(ranking))
    pairs=zip(range(window-1),range(1,window)) if version=='v1' else itertools.combinations(range(window),2)
    fresh=set();triggers=[]
    for i,j in pairs:
        a,b=ranking[i],ranking[j]
        if scores[a]-scores[b]>=tau or (a in already and b in already):continue
        swap=list(ranking);swap[i],swap[j]=swap[j],swap[i]
        if packer(swap)['context']!=base:
            fresh.update({a,b}-already);triggers.append([a,b])
    return sorted(fresh),triggers
