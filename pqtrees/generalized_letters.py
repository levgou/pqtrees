from collections import Counter
from dataclasses import dataclass, asdict
from itertools import chain
from typing import Tuple, Optional, List, Collection, Mapping

from frozendict import frozendict
from funcy import select_keys, lfilter

from pqtrees.utilities.perm_helpers import flatmap, tmap, all_eq


@dataclass(frozen=True)
class MultipleOccurrenceChar:
    char: object
    count: int

    def __str__(self):
        return str(self.char) * self.count

    def __repr__(self):
        return f"MC({self.char},{self.count})"


@dataclass(frozen=True)
class MergedChar:
    char_orders: Tuple[Tuple[object, object], ...]
    counts: Mapping[object, int]
    # other_side_same: bool
    # context_same: bool

    def __str__(self):
        return f"{self.char_orders[0][0]}+{self.char_orders[0][1]}"

    def __repr__(self):
        return f"Merged{self.char_orders}"

    @classmethod
    def from_occurrences(cls,
                         # other_side_same: bool, context_same: bool,
                         *occurrences: Tuple[object, object]):
        counts = Counter(occurrences)
        char_orders = tuple(sorted(count[0] for count in counts.most_common()))
        return MergedChar(char_orders, frozendict(counts))


@dataclass(frozen=True)
class ContextChar:
    left_char: Optional[object]
    char: object
    right_char: Optional[object]
    index: Optional[int] = None
    perm: Optional[object] = None

    @classmethod
    def from_perm_index(cls, perm, index):
        left_char = perm[index - 1] if index > 0 else None
        right_char = perm[index + 1] if index < len(perm) - 1 else None
        char = perm[index]
        return ContextChar(left_char, char, right_char, index, perm)

    def char_info(self):
        return frozendict(select_keys({'left_char', 'right_char', 'char'}, asdict(self)))

    @classmethod
    def eq_to_info(cls, info: dict, cc: 'ContextChar') -> bool:
        return info == cc.char_info()

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, ContextChar):
            return False
        return all_eq(self.char_info(), o.char_info())

    def __repr__(self):
        return f"CC[{self.left_char}, <{self.char}>, {self.right_char}] - {self.perm}"


ContextPerm = List[ContextChar]


def all_neighbours_list(cchars: Collection[ContextChar]) -> list:
    return lfilter(bool, flatmap(lambda cc: [cc.right_char, cc.left_char], cchars))


def filter_cchars(char, *cchars_collections: Collection[ContextChar]):
    return filter(lambda cc: cc.char == char, chain(*cchars_collections))


def neighbours_of(char, *cchars_collections: Collection[ContextChar]):
    return filter(lambda cc: char in [cc.left_char, cc.right_char], chain(*cchars_collections))


def char_neighbour_tuples(cchars_collections: Collection[ContextChar], char) -> Tuple[Tuple[object, object], ...]:
    return tmap(
        lambda cc: (cc.left_char, cc.char) if cc.left_char == char else (cc.char, cc.right_char),
        cchars_collections
    )


def iter_common_neighbours(*cchars: ContextChar):
    count = Counter(all_neighbours_list(cchars))
    for neighbour, freq in count.most_common():
        if freq == len(cchars):
            yield neighbour
        else:
            break
