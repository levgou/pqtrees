from itertools import islice
from typing import List, Optional, Sequence, Dict
from pqtrees.common_intervals.common_interval import CommonInterval
from pqtrees.common_intervals.proj_types import Interval

CharIndex = Dict[Sequence, Dict[frozenset, tuple]]


def window(seq, n=2):
    """
    Returns a sliding window (of width n) over data from the iterable
      s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...
    """
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def index_perm(perm: Sequence, length: int):
    return {
        frozenset(w): (idx, idx + length - 1)
        for idx, w in enumerate(window(perm, length))
    }


def index_perms(perms: Sequence[Sequence], length: int):
    return {p: index_perm(p, length) for p in perms}


def find_in_others(k_index: CharIndex, charset: frozenset, others: Sequence[Sequence]) -> Optional[List[Interval]]:
    in_others = [k_index[perm].get(charset) for perm in others]
    if None in in_others:
        return None
    return in_others


def common_k(perm_id: Sequence, perms: Sequence[Sequence], k: int):
    other_perms = perms[1:]
    k_index = index_perms(other_perms, k)

    for w1 in window(perm_id, k):
        charset = frozenset(w1)
        if in_others := find_in_others(k_index, charset, perms[1:]):
            yield CommonInterval((w1[0], w1[-1]), *in_others)


def common_k_indexed(*perms: Sequence) -> List[CommonInterval]:
    t_perms = tuple(tuple(p) for p in perms)
    commons = []
    perm_id = t_perms[0]

    for l in range(2, len(perm_id) + 1):
        commons.extend(common_k(perm_id, t_perms, l))

    return commons


def common_k_indexed_with_singletons(*perms: Sequence) -> List[CommonInterval]:
    t_perms = tuple(tuple(p) for p in perms)
    commons = list(common_k(t_perms[0], t_perms, 1))
    commons.extend(common_k_indexed(*t_perms))

    return commons
