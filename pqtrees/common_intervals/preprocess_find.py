from itertools import islice
from typing import List, Optional, Sequence, Dict
from pqtrees.common_intervals.common_interval import CommonInterval
from pqtrees.proj_types import Interval

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


def find_in_others(k_index: CharIndex, charset: frozenset, others: Sequence[Sequence]) -> Optional[List[Interval]]:
    in_others = []
    chars_len = len(charset)
    for perm in others:

        if perm not in k_index:
            k_index[perm] = index_perm(perm, chars_len)

        if not (in_other := k_index[perm].get(charset)):
            return None
        in_others.append(in_other)

    return in_others


def common_k(perm_id: Sequence, perms: Sequence[Sequence], sub_len: int):
    k_index = {}  # index will be lazily updated on request

    for w1 in window(perm_id, sub_len):
        charset = frozenset(w1)
        if in_others := find_in_others(k_index, charset, perms[1:]):
            yield CommonInterval((w1[0], w1[-1]), *in_others)


def common_k_indexed(*perms: Sequence) -> List[CommonInterval]:
    commons = []
    perm_id = perms[0]

    for l in range(2, len(perm_id) + 1):
        current = common_k(perm_id, perms, l)
        commons.extend(current)

    return commons


def common_k_indexed_with_singletons(*perms: Sequence) -> List[CommonInterval]:
    t_perms = tuple(tuple(p) for p in perms)
    commons = list(common_k(t_perms[0], t_perms, 1))
    commons.extend(common_k_indexed(*t_perms))

    return commons
