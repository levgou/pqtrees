import operator
from functools import partial
from typing import Iterable, Sized, Callable, Sequence

from funcy import group_by, flatten, chain, select_keys, merge_with

from pqtrees.proj_types import Permutations
from pqtrees.common_intervals.trivial import window


def neighbour_set(perm, char):
    indices = [i for i, x in enumerate(perm) if x == char]
    neighbour_indices = {
        x - 1 for x in indices if x > 0
    }.union({
        x + 1 for x in indices if x < len(perm) - 1
    })

    neighbours = frozenset(perm[idx] for idx in neighbour_indices)
    return neighbours


def is_list_consecutive(lst):
    sorted_list = list(sorted(lst))
    consecutive_list = list(range(sorted_list[0], sorted_list[-1] + 1))
    return consecutive_list == sorted_list


def iter_char_occurrence(seq: Sequence):
    """
    example:
    GeeeEEKKKss -> (G, 1), (e, 3), (E, 2), (K, 3), (s, 2)
    """
    idx = 0
    it = tuple(seq) + ("#@$@#$@#$@#",)
    while idx < len(it) - 1:
        count = 1
        while it[idx] == it[idx + 1]:
            idx += 1
            count += 1

            if idx + 1 == len(it):
                break

        yield it[idx], count
        idx += 1


def invert_dict_multi_val(d: dict):
    """
    example: {1:2, 3:2} -> {2, (1, 3)}
    """
    return merge_with(tuple, *({val: key} for key, val in d.items()))


def all_eq(*items):
    return items.count(items[0]) == len(items)


def same_abc(perms: Permutations):
    return all_eq(*map(frozenset, perms))


def diff_abc(perms: Permutations):
    return not same_abc(perms)


def same_len(collections: Iterable[Sequence]):
    return all_eq(*map(len, collections))


def diff_len(perms: Permutations):
    return not same_len(perms)


def subd_dicts_eq(keys: set, *objs: object):
    if not objs:
        return True

    eq_subset = partial(select_keys, keys)
    return all_eq(*map(eq_subset, objs))


def all_indices(perm, char):
    return [i for i, x in enumerate(perm) if x == char]


def num_appear(col, obj):
    return len(all_indices(col, obj))


def irange(itr: Sized):
    return range(len(itr))


def map1(f: Callable[[object, object], object], arg1: object, it: Iterable):
    return map(partial(f, arg1), it)


def map2(f: Callable[[object, object], object], arg1: object, arg2: object, it: Iterable):
    return map(partial(f, arg1, arg2), it)


def tmap(f: Callable[[object], object], it: Iterable):
    return tuple(map(f, it))


def tmap1(f: Callable[[object, object], object], arg1: object, it: Iterable):
    return tmap(partial(f, arg1), it)


def tmap2(f: Callable[[object, object], object], arg1: object, arg2: object, it: Iterable):
    return tmap(partial(f, arg1, arg2), it)


def smap(f: Callable[[object], object], it: Iterable):
    return set(map(f, it))


def smap1(f: Callable[[object, object], object], arg1: object, it: Iterable):
    return smap(partial(f, arg1), it)


def smap2(f: Callable[[object, object], object], arg1: object, arg2: object, it: Iterable):
    return smap(partial(f, arg1, arg2), it)


def filter_attr_eq(attr: str, eq_to: object, it: Iterable):
    return filter(lambda x: getattr(x, attr) == eq_to, it)


def lfilter_attr_eq(attr: str, eq_to: object, it: Iterable):
    return list(filter_attr_eq(attr, eq_to, it))


def tfilter_attr_eq(attr: str, eq_to: object, it: Iterable):
    return tuple(filter_attr_eq(attr, eq_to, it))


def filter_f_eq(f: str, eq_to: object, it: Iterable):
    return filter(lambda x: getattr(x, f)() == eq_to, it)


def lfilter_f_eq(f: str, eq_to: object, it: Iterable):
    return list(filter_f_eq(f, eq_to, it))


def tfilter_f_eq(f: str, eq_to: object, it: Iterable):
    return tuple(filter_f_eq(f, eq_to, it))


def filter_fx_eq(f: Callable, eq_to: object, it: Iterable):
    return filter(lambda x: f(x) == eq_to, it)


def lfilter_fx_eq(f: Callable, eq_to: object, it: Iterable):
    return list(filter_fx_eq(f, eq_to, it))


def tfilter_fx_eq(f: Callable, eq_to: object, it: Iterable):
    return tuple(filter_fx_eq(f, eq_to, it))


def tfilter(pred: Callable[[object], bool], it: Iterable):
    return tuple(filter(pred, it))


def tfilter1(pred: Callable[[object, object], bool], arg1: object, it: Iterable):
    return tfilter(partial(pred, arg1), it)


def sflatten(it: Iterable):
    return set(flatten(it))


def flatmap(f: Callable, it: Iterable):
    return flatten(map(f, it))


def flatmap1(f: Callable, arg1: object, it: Iterable):
    return flatten(map1(f, arg1, it))


def flatmap2(f: Callable, arg1: object, arg2: object, it: Iterable):
    return flatten(map2(f, arg1, arg2, it))


def lflatmap(f: Callable, it: Iterable):
    return list(flatmap(f, it))


def lflatmap1(f: Callable, arg1: object, it: Iterable):
    return list(flatmap1(f, arg1, it))


def lflatmap2(f: Callable, arg1: object, arg2: object, it: Iterable):
    return list(flatmap2(f, arg1, arg2, it))


def sflatmap(f: Callable, it: Iterable):
    return set(flatmap(f, it))


def sflatmap1(f: Callable, arg1: object, it: Iterable):
    return set(flatmap1(f, arg1, it))


def sflatmap2(f: Callable, arg1: object, arg2: object, it: Iterable):
    return set(flatmap2(f, arg1, arg2, it))


def group_by_attr(attr: str, it: Iterable):
    return dict(group_by(operator.attrgetter(attr), it))


def tsorted(it: Iterable, key=None):
    return tuple(sorted(it, key=key))


def geti(seq: Sequence, i: int, default=None):
    if i < 0 or i >= len(seq):
        return default
    return seq[i]


# noinspection PyTypeChecker
def perms_as_stream(perms: Sequence[Sequence], l_pad=None, r_pad=None, pad_num=1):
    """
    for a sequence of sequences return an item stream out of those sequences (like chain)
    padded by l_pad and r_pad:
    ex:
    for [[1,2], [3,4]] with default pads we would get:
    -> None, 1, 2, None, None, 3, 4, None

    this is useful when we want a context window over a collection,
    but dont want to miss the first / last chars as middle of the window
    """
    left_pad = [l_pad] * pad_num
    right_pad = [r_pad] * pad_num
    return chain(*[left_pad + list(p) + right_pad
                   for p in perms])


def assoc_cchars_with_neighbours(cchars, neighbour, context_perms):
    """
    will return a tuple for each cchar with specified neighbour at the appropriate place
    ex: for <1,2,3>, <3,2,1> and neighbour 1 the produced tuples are:
    (1,2), (2,1)
    """
    cc_set = set(cchars)
    res = []
    for left, cchar, right in window(perms_as_stream(context_perms), 3):
        if cchar in cc_set:
            if left and left.char == neighbour:
                res.append((left, cchar))
            else:
                res.append((cchar, right))

    return res
