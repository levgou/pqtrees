import operator
from collections import Counter
from functools import partial
from typing import Iterable, Sized, Callable, Collection

from funcy import group_by, flatten, lfilter, chain, lmap

from pqtrees.common_intervals.generalized_letters import ContextChar
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


def all_neighbours_list(cchars: Collection[ContextChar]) -> list:
    return lfilter(bool, flatmap(lambda cc: [cc.right_char, cc.left_char], cchars))


def filter_cchars(char, *cchars_collections: Collection[ContextChar]):
    return filter(lambda cc: cc.char == char, chain(*cchars_collections))


def neighbours_of(char, *cchars_collections: Collection[ContextChar]):
    return filter(lambda cc: char in [cc.left_char, cc.right_char], chain(*cchars_collections))


def char_neighbour_tuples(cchars_collections: Collection[ContextChar], char):
    return tmap(
        lambda cc: (cc.left_char, cc.char) if cc.left_char == char else (cc.right_char, cc.char),
        cchars_collections
    )


def iter_common_neighbours(*cchars: ContextChar):
    count = Counter(all_neighbours_list(cchars))
    for neighbour, freq in count.most_common():
        if freq == len(cchars):
            yield neighbour
        else:
            break


def assoc_cchars_with_neighbours(cchars, neighbour, context_perms):
    cc_set = set(cchars)
    res = []
    for left, cchar, right in window(chain([[None] + list(p) + [None] for p in context_perms]), 3):
        if cchar in cc_set:
            if left and left.char == neighbour:
                res.append((left, cchar))
            else:
                res.append((cchar, right))

    return res
