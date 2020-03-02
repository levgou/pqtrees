import operator
from functools import partial
from typing import Iterable, Sized, Callable

from funcy import group_by


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


def irange(itr: Sized):
    return range(len(itr))


def tmap(f: Callable[[object], object], it: Iterable):
    return tuple(map(f, it))


def tmap1(f: Callable[[object, object], object], arg1: object, it: Iterable):
    return tmap(partial(f, arg1), it)


def tmap2(f: Callable[[object, object], object], arg1: object, arg2: object, it: Iterable):
    return tmap(partial(f, arg1, arg2), it)


def group_by_attr(attr: str, it: Iterable):
    return group_by(operator.attrgetter(attr), it)
