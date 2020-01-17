from typing import Tuple

from pqtrees.common_intervals.proj_types import Index


class CommonInterval:
    intervals: Tuple[Tuple[Index, Index], ...]
    first_start: Index
    first_end: Index

    def __init__(self, *args: Tuple[Index, Index]) -> None:
        super().__init__()
        self._validate_intervals(args)
        self.intervals = tuple(args)
        self.first_start = self.intervals[0][0]
        self.first_end = self.intervals[0][1]

    def is_trivial(self) -> bool:
        return self.first_end == self.first_start

    def included_in_other(self, other: 'CommonInterval') -> bool:
        return self._left_included_in_right(self, other)

    @classmethod
    def _left_included_in_right(cls, left: 'CommonInterval', right: 'CommonInterval') -> bool:
        return left.first_end <= right.first_end and left.first_start >= right.first_start

    @classmethod
    def one_interval_includes_other(cls, ci1: 'CommonInterval', ci2: 'CommonInterval') -> bool:
        return cls._left_included_in_right(ci1, ci2) or cls._left_included_in_right(ci2, ci1)

    @classmethod
    def intervals_intersect(cls, ci1: 'CommonInterval', ci2: 'CommonInterval'):
        return (ci1.first_end >= ci2.first_start and ci1.first_start <= ci2.first_end) or \
               (ci1.first_start <= ci2.first_end and ci1.first_end >= ci2.first_start)

    @classmethod
    def intersect(cls, ci1: 'CommonInterval', ci2: 'CommonInterval') -> bool:
        """
        Test whether the intervals non trivially intersect by the first interval
        trivial i.e one interval includes the other e.g [1,4] & [2,3]
        """
        return not cls.one_interval_includes_other(ci1, ci2) and \
               cls.intervals_intersect(ci1, ci2)

    @staticmethod
    def _validate_intervals(intervals: Tuple[Tuple[Index, Index]]) -> None:
        not_sorted_intervals = [inter for inter in intervals if sorted(inter) != list(inter)]
        assert not len(not_sorted_intervals), f"All intervals should be sorted {not_sorted_intervals}"

        lens = {y - x for x, y in intervals}
        assert len(lens) == 1, f"All intervals should be of same len {intervals}"

    def __str__(self) -> str:
        return f'CI{list(self.intervals[0])}'
        # return f'CI{list(self.intervals)}'

    __repr__ = __str__

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, CommonInterval):
            return False
        return self.intervals == o.intervals

    def __hash__(self) -> int:
        return hash(self.intervals)

    def __len__(self) -> int:
        return self.intervals[1][1] - self.intervals[1][0] + 1


class CommonIntervalWeakEq(CommonInterval):

    def __init__(self, *args: Tuple[Index, Index]) -> None:
        super().__init__(*args)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, CommonIntervalWeakEq):
            return False
        return self.intervals[0] == o.intervals[0]

    def __hash__(self) -> int:
        return hash(self.intervals[0])

    def __str__(self) -> str:
        return "W" + super().__str__()

    __repr__ = __str__
