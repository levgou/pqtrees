from typing import Tuple

from pqtrees.common_intervals.proj_types import Index


class CommonInterval:

    def __init__(self, *args: Tuple[Index, Index]) -> None:
        super().__init__()
        self._validate_intervals(args)
        self.intervals = tuple(args)

    @staticmethod
    def _validate_intervals(intervals: Tuple[Tuple[Index, Index]]) -> None:
        not_sorted_intervals = [inter for inter in intervals if sorted(inter) != list(inter)]
        assert not len(not_sorted_intervals), f"All intervals should be sorted {not_sorted_intervals}"

        lens = {y - x for x, y in intervals}
        assert len(lens) == 1, f"All intervals should be of same len {intervals}"


    def __str__(self) -> str:
        return f'CI{list(self.intervals)}'

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
