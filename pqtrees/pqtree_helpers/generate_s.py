from collections import defaultdict
from pprint import pprint
from typing import Dict, Set, Optional, Tuple, Callable, Iterable, List
from funcy import lfilter, lmap
from itertools import chain

from pqtrees.common_intervals.common_interval import CommonInterval
from pqtrees.common_intervals.preprocess_find import common_k_indexed
from pqtrees.pqtree_helpers.reduce_intervals import ReduceIntervals

Arrows = Tuple[Optional[CommonInterval], Optional[CommonInterval],
               Optional[CommonInterval], Optional[CommonInterval]]


class IntervalHierarchy:
    nesting_levels: Dict[int, list]
    reverse_index: Dict[CommonInterval, int]

    def __init__(self) -> None:
        super().__init__()
        self.nesting_levels = defaultdict(list)
        self.reverse_index = {}

    def _sort_lists_by_start(self):
        def sort_ci_list(lst): lst.sort(key=lambda ci: ci.first_start)

        lmap(sort_ci_list, self.nesting_levels.values())

    @classmethod
    def from_irreducible_intervals(cls, intervals: Set[CommonInterval]) -> 'IntervalHierarchy':
        ih = IntervalHierarchy()

        for interval in intervals:
            include_interval = lfilter(interval.included_in_other, intervals - {interval})
            nest_level = len(include_interval)
            ih.nesting_levels[nest_level].append(interval)

        lmap(lambda l: l.sort(key=lambda ci: ci.first_start),
             ih.nesting_levels.values())

        ih.reverse_index = {
            ci: lvl
            for lvl, ci_lst in ih.nesting_levels.items()
            for ci in ci_lst
        }

        return ih

    def iter_bottom_up(self, from_level=float('inf')) -> Iterable[CommonInterval]:
        keys_in_order = reversed(sorted(filter(lambda k: k <= from_level, self.nesting_levels.keys())))
        lst_in_order = [self.nesting_levels[k] for k in keys_in_order]
        return chain(*lmap(iter, lst_in_order))

    def iter_top_down(self, from_level=0):
        keys_in_order = sorted(filter(lambda k: k >= from_level, self.nesting_levels.keys()))
        lst_in_order = [self.nesting_levels[k] for k in keys_in_order]
        return chain(*lmap(iter, lst_in_order))

    def _start_or_end_with_at_level(self, stat_or_end_equal: Callable[[CommonInterval], bool], lvl: int, up: bool):
        if up:
            search_level = lvl - 1
            it = self.iter_bottom_up
        else:
            search_level = lvl + 1
            it = self.iter_top_down

        maybe_ci = filter(stat_or_end_equal, it(search_level))
        return next(iter(maybe_ci), None)

    def _start_with_at_level(self, start: int, lvl: int, up: bool) -> Optional[CommonInterval]:
        return self._start_or_end_with_at_level(lambda ci: ci.first_start == start, lvl, up)

    def _end_with_at_level(self, start: int, lvl: int, up: bool) -> Optional[CommonInterval]:
        return self._start_or_end_with_at_level(lambda ci: ci.first_end == start, lvl, up)

    def s_arrows_of(self, ci: CommonInterval) -> Arrows:
        lvl = self.reverse_index[ci]
        return (
            self._start_with_at_level(ci.first_start, lvl, True),
            self._start_with_at_level(ci.first_start, lvl, False),
            self._end_with_at_level(ci.first_end, lvl, True),
            self._end_with_at_level(ci.first_end, lvl, False),
        )

    def chain_starting_with(self, ci: CommonInterval) -> List[CommonInterval]:
        lvl = self.reverse_index[ci]
        chain = []

        ci_index = self.nesting_levels[lvl].index(ci)
        items_after_ci = self.nesting_levels[lvl][ci_index + 1:]

        cur = ci
        for other in items_after_ci:
            if CommonInterval.intersect(cur, other):
                chain.append(other)
                cur = other
            else:
                break

        return [ci] + chain if chain else []

    @property
    def boundaries(self):
        return (
            0,
            max(self.reverse_index, key=lambda ci: ci.first_end).first_end
        )


def produce_s_for_example():
    commons1 = [

        # len 1
        *[CommonInterval((i, i)) for i in range(9)],

        # len 2
        CommonInterval((0, 1), (6, 7), (0, 1)),
        CommonInterval((1, 2), (7, 8), (1, 2)),
        CommonInterval((3, 4), (2, 3), (5, 6)),
        CommonInterval((4, 5), (3, 4), (6, 7)),

        # len 3
        CommonInterval((0, 2), (6, 8), (0, 2)),
        CommonInterval((3, 5), (2, 4), (5, 7)),

        # len 4
        CommonInterval((3, 6), (2, 5), (4, 7)),

        # len 5
        CommonInterval((3, 7), (1, 5), (3, 7)),

        # len 6
        CommonInterval((3, 8), (0, 5), (3, 8)),

        # len 8
        CommonInterval((0, 7), (1, 8), (0, 7)),

        # len 9
        CommonInterval((0, 8), (0, 8), (0, 8)),
    ]

    ir_intervals = ReduceIntervals.reduce(commons1)
    s = IntervalHierarchy.from_irreducible_intervals(ir_intervals)

    pprint(s.nesting_levels)


if __name__ == '__main__':
    produce_s_for_example()

    perms = [
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        (9, 8, 7, 6, 3, 1, 5, 4, 2, 0)
    ]
    common_intervals = common_k_indexed(*perms)
    ir_intervals = ReduceIntervals.reduce(common_intervals)
    pprint(ir_intervals)
    print(common_intervals)
    s = IntervalHierarchy.from_irreducible_intervals(ir_intervals)
    pprint(s.nesting_levels)
