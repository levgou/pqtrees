from typing import List, Optional, Sequence

from pqtrees.common_intervals.common_interval import CommonInterval
from itertools import islice

from pqtrees.proj_types import Interval


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


def find_in_other(chars1: set, other: Sequence) -> Optional[Interval]:
    l = len(chars1)

    for idx2, w2 in enumerate(window(other, l)):
        chars2 = set(w2)

        if chars1 == chars2:
            return idx2, idx2 + l - 1

    return None


def trivial_common(perm_a: Sequence, perm_b: Sequence) -> List[CommonInterval]:
    commons = []
    for l in range(2, len(perm_a) + 1):
        for w1 in window(perm_a, l):
            chars1 = set(w1)

            if in_other := find_in_other(chars1, perm_b):
                commons.append(CommonInterval((w1[0], w1[-1]), in_other))

    return commons


def find_in_others(chars1: set, others: Sequence[Sequence]) -> Optional[List[Interval]]:
    in_others = []
    for other in others:
        if in_other := find_in_other(chars1, other):
            in_others.append(in_other)
        else:
            return None

    return in_others


def common_k(perm_id: Sequence, perms: Sequence[Sequence], k: int):
    for w1 in window(perm_id, k):
        chars1 = set(w1)
        if in_others := find_in_others(chars1, perms[1:]):
            yield CommonInterval((w1[0], w1[-1]), *in_others)


def trivial_common_k(*perms: Sequence) -> List[CommonInterval]:
    commons = []
    perm_id = perms[0]

    for l in range(2, len(perm_id) + 1):
        commons.extend(common_k(perm_id, perms, l))

    return commons


def trivial_common_k_with_singletons(*perms: Sequence) -> List[CommonInterval]:
    commons = list(common_k(perms[0], perms, 1))
    commons.extend(trivial_common_k(*perms))

    return commons
