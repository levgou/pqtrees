from typing import Collection, Set, List

from pqtrees.common_intervals.common_interval import CommonInterval
from pqtrees.pqtree_helpers.reduce_intervals import ReduceIntervals

CommonIntervals = Collection[CommonInterval]
IntervalSet = Set[CommonInterval]
IntervalList = List[CommonInterval]


def test_given_examples():
    # C = {[1, 2], [2, 3], [1, 3], [4, 5], [5, 6], [4, 6], [4, 7], [4, 8], [4, 9], [1, 8], [1, 9]}
    # I = {[1, 2], [2, 3],         [4, 5], [5, 6],         [4, 7], [4, 8], [4, 9], [1, 8]        }

    commons1 = [
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

    irreducible1 = {
        # len 2
        CommonInterval((0, 1), (6, 7), (0, 1)),
        CommonInterval((1, 2), (7, 8), (1, 2)),
        CommonInterval((3, 4), (2, 3), (5, 6)),
        CommonInterval((4, 5), (3, 4), (6, 7)),

        # len 4
        CommonInterval((3, 6), (2, 5), (4, 7)),

        # len 5
        CommonInterval((3, 7), (1, 5), (3, 7)),

        # len 6
        CommonInterval((3, 8), (0, 5), (3, 8)),

        # len 8
        CommonInterval((0, 7), (1, 8), (0, 7)),
    }

    commons2 = [
        # len 2
        CommonInterval((0, 1)),
        CommonInterval((1, 2)),

        # len 3
        CommonInterval((0, 2)),

        # len 5
        CommonInterval((2, 7)),
        CommonInterval((3, 8)),
        CommonInterval((4, 9)),

        # len 7
        CommonInterval((0, 7)),

        # len 8
        CommonInterval((0, 8)),

        # len 9
        CommonInterval((0, 9)),
    ]

    irreducible2 = {
        # len 2
        CommonInterval((0, 1)),
        CommonInterval((1, 2)),

        # len 5
        CommonInterval((2, 7)),
        CommonInterval((3, 8)),
        CommonInterval((4, 9)),
    }

    assert ReduceIntervals.reduce(commons1) == irreducible1
    assert ReduceIntervals.reduce(commons2) == irreducible2


if __name__ == '__main__':
    test_given_examples()
