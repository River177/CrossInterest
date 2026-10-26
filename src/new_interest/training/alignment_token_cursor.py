"""Strict per-example supervised-token counts for mixed completion lengths."""
from __future__ import annotations


def expected_supervised_tokens(counts, lengths):
    if len(counts)!=len(lengths) or any(type(v) is not int or v<0 for v in counts):
        raise ValueError('Invalid exposure counts')
    if any(type(v) is not int or v<=0 for v in lengths):
        raise ValueError('Invalid frozen completion lengths')
    return [count*length for count,length in zip(counts,lengths)]
