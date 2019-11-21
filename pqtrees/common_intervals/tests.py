from pqtrees.common_intervals.bsc import CommonInterval, bsc
from pqtrees.common_intervals.lhp import lhp


def test_common_intervals(alg):
    pi1 = (0, 1, 2, 3, 4, 5, 6, 7, 8)
    pi2 = (8, 7, 3, 4, 5, 6, 0, 1, 2)

    commons = [
        # len 2
        CommonInterval((0, 1), (6, 7)),
        CommonInterval((1, 2), (7, 8)),
        CommonInterval((3, 4), (2, 3)),
        CommonInterval((4, 5), (3, 4)),
        CommonInterval((5, 6), (4, 5)),
        CommonInterval((7, 8), (0, 1)),

        # len 3
        CommonInterval((0, 2), (6, 8)),
        CommonInterval((3, 5), (2, 4)),
        CommonInterval((4, 6), (3, 5)),

        # len 4
        CommonInterval((3, 6), (2, 5)),

        # len 4
        CommonInterval((3, 7), (1, 5)),
        # len 6
        CommonInterval((3, 8), (0, 5)),

        # len 7
        CommonInterval((0, 6), (2, 8)),

        # len 8
        CommonInterval((0, 7), (1, 8)),

        # len 9
        CommonInterval((0, 8), (0, 8)),
    ]

    found_commons = alg(pi1, pi2)
    common_set = set(commons)
    found_common_set = set(found_commons)
    assert found_common_set == common_set, f'\nFor alg [{alg.__name__}]\n' \
                                           f'Only in found: {found_common_set - common_set}, ' \
                                           f'only in common: {common_set - found_common_set}'


if __name__ == '__main__':
    test_common_intervals(bsc)
    test_common_intervals(lhp)
