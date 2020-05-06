from typing import Collection, Set, List

import networkx as nx
from funcy import lmap, lfilter

from pqtrees.common_intervals.common_interval import CommonInterval

CommonIntervals = Collection[CommonInterval]
IntervalSet = Set[CommonInterval]
IntervalList = List[CommonInterval]


class ReduceIntervals:

    @classmethod
    def _intersects_with(cls, intervals: CommonIntervals, ci: CommonInterval) -> IntervalList:
        """Find intervals that intersect with ci and are 'After' ci"""
        return [other for other in intervals
                if CommonInterval.intersect(ci, other) and ci.first_end <= other.first_end]

    @classmethod
    def _add_successors(cls, g: nx.DiGraph, intervals: CommonIntervals, interval: CommonInterval) -> None:
        lmap(lambda ci: g.add_edge(interval, ci), cls._intersects_with(intervals, interval))

    @classmethod
    def _find_redundant_intervals(cls, intervals: CommonIntervals) -> IntervalSet:
        no_singleton_intervals = lfilter(lambda inter: inter.first_end != inter.first_start, intervals)
        intersection_dict = {ci: cls._intersects_with(no_singleton_intervals, ci) for ci in no_singleton_intervals}
        start_end_index = {(ci.first_start, ci.first_end): ci for ci in no_singleton_intervals}

        redundant = {

            start_end_index.get((src.first_start, dest.first_end))
            for src in no_singleton_intervals
            for dest in intersection_dict[src]
        }

        return redundant - {None}

    @classmethod
    def reduce(cls, intervals: Collection[CommonInterval]) -> Set[CommonInterval]:
        return set(intervals) - cls._find_redundant_intervals(intervals)
