import random

from pqtrees.common_intervals.common_interval import CommonInterval
from pqtrees.pqtree_helpers.generate_s import IntervalHierarchy
from pqtrees.utilities.perm_helpers import tmap
from pqtrees.pqtree import PQTreeBuilder
from pqtrees.common_intervals.preprocess_find import common_k_indexed_with_singletons
from pqtrees.pqtree_helpers.reduce_intervals import ReduceIntervals
from pqtrees.utilities.string_mutations import mutate_collection
from pqtrees.common_intervals.trivial import trivial_common_k_with_singletons


def known_example():
    """
    the one from:
    Gene Proximity Analysis across Whole Genomes via PQ Trees1
    """
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

    pqtree = PQTreeBuilder._from_s(s, None)
    assert pqtree.to_parens() == "[[0 1 2] [[[3 4 5] 6] 7] 8]"

    # pqtree.show()


def known_e2e():
    """From wikipedia PQTree entry """

    perms = [(0, 1, 2, 3, 4), (0, 1, 3, 2, 4), (0, 2, 1, 3, 4),
             (0, 2, 3, 1, 4), (0, 3, 1, 2, 4), (0, 3, 2, 1, 4),
             (4, 1, 2, 3, 0), (4, 1, 3, 2, 0), (4, 2, 1, 3, 0),
             (4, 2, 3, 1, 0), (4, 3, 1, 2, 0), (4, 3, 2, 1, 0)]

    strs = {"".join(str(x) for x in p) for p in perms}

    common_intervals_trivial = trivial_common_k_with_singletons(*perms)
    common_intervals = common_k_indexed_with_singletons(*perms)

    assert common_intervals_trivial == common_intervals

    ir_intervals = ReduceIntervals.reduce(common_intervals)
    s = IntervalHierarchy.from_irreducible_intervals(ir_intervals)

    pqtree = PQTreeBuilder._from_s(s)
    frontier = list(pqtree.frontier())
    assert pqtree.approx_frontier_size() == len(frontier)
    assert pqtree.to_parens() == "[0 (1 2 3) 4]"
    assert strs.issubset(frontier), strs - frontier


def known_e2e_2():
    """pqtrees/docs/_static/images/pqtree-example-rat-human.png"""

    perms = [
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        (9, 8, 7, 6, 3, 1, 5, 4, 2, 0)
    ]

    strs = {"".join(str(x) for x in p) for p in perms}

    common_intervals = trivial_common_k_with_singletons(*perms)

    ir_intervals = ReduceIntervals.reduce(common_intervals)
    s = IntervalHierarchy.from_irreducible_intervals(ir_intervals)

    pqtree = PQTreeBuilder._from_s(s)
    frontier = pqtree.frontier()

    assert pqtree.to_parens() == "[0 (1 2 3 [4 5]) 6 7 8 9]"
    assert strs.issubset(frontier), strs - frontier


def simple_pq_tree_size_tests():
    tests = [
        (
            [
                (0, 1, 2, 3),
                (3, 1, 2, 0),
            ],
            "[0 [1 2] 3]",
            4
        ),
        (
            [
                (0, 1, 2, 3),
                (3, 1, 2, 0),
                (0, 3, 2, 1),
            ],
            "[0 [[1 2] 3]]",
            8
        ),
        (
            [
                (0, 1, 2, 3),
                (3, 1, 2, 0),
                (0, 3, 2, 1),
                (3, 0, 1, 2)
            ],
            '(0 [1 2] 3)',
            12
        )
    ]

    def run_tests(perms, paren_repr, front_size):
        strs = {"".join(str(x) for x in p) for p in perms}

        common_intervals = trivial_common_k_with_singletons(*perms)

        ir_intervals = ReduceIntervals.reduce(common_intervals)
        s = IntervalHierarchy.from_irreducible_intervals(ir_intervals)

        pqtree = PQTreeBuilder._from_s(s)
        frontier = pqtree.frontier()

        assert strs.issubset(frontier), strs - frontier
        assert pqtree.to_parens() == paren_repr
        assert pqtree.approx_frontier_size() == front_size

        # PQTreeVisualizer.show(pqtree)

    for t in tests:
        run_tests(*t)


def rand_size_tests():
    ITERATIONS = 1000

    for i in range(ITERATIONS):
        id_perm = list(range(1, 10))

        other_perms = [list(id_perm), list(id_perm)]
        for p in other_perms:
            mutate_collection(p, 2)

        ps = tmap(tuple, (id_perm, *other_perms))

        pq = PQTreeBuilder.from_perms(ps)
        assert pq.approx_frontier_size() == len(list(pq.frontier())), [pq.to_parens(),
                                                                       set(pq.frontier()),
                                                                       pq.approx_frontier_size(),
                                                                       len(list(pq.frontier())),
                                                                       len(set(pq.frontier()))]


def string_inputs_pqtree():
    def test_case(length, num_perm):
        assert length <= 10

        abc = "abcdefghijklmnopqrstuvwxyz"
        p1 = range(length)
        translation = dict(zip(p1, abc))

        perms = [list(p1) for _ in range(num_perm)]
        it = iter(perms)
        next(it)
        for lst in it:
            random.shuffle(lst)

        translated_perms = ["".join(translation[n] for n in l) for l in perms]

        p_repr = PQTreeBuilder.from_perms(perms).to_parens()
        str_key_translation = {str(k): v for k, v in translation.items()}
        p_repr_translated = "".join(str_key_translation.get(c, c) for c in p_repr)
        str_parens = PQTreeBuilder.from_perms(translated_perms).to_parens()
        str_parens = PQTreeBuilder.from_perms(translated_perms).to_parens()

        assert p_repr_translated == str_parens

    test_case(5, 2)
    test_case(5, 4)
    test_case(9, 4)


def compare_oren():
    perms_2851 = [
        (1, 2, 3, 4, 5, 6, 7, 8, 9),
        (6, 1, 2, 3, 4, 5, 7, 8, 9),
    ]
    perms_4545 = [
        (1, 2, 3, 4, 5, 6, 7),
        (2, 3, 4, 1, 5, 6, 7),
    ]

    perms_790 = [
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14),
        (7, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14),
    ]

    assert PQTreeBuilder.from_perms(perms_2851).to_parens() == "[[[1 2 3 4 5] 6] 7 8 9]"
    assert PQTreeBuilder.from_perms(perms_4545).to_parens() == "[[1 [2 3 4]] 5 6 7]"
    assert PQTreeBuilder.from_perms(perms_790).to_parens() == "[[[1 2 3 4 5 6] 7] 8 9 10 11 12 13 14]"


def compare_oren_from_rand():
    ps1 = [
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        (9, 3, 2, 6, 5, 1, 8, 10, 7, 4)
    ]

    ps2 = [
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
        (14, 10, 12, 7, 1, 3, 5, 13, 8, 2, 6, 11, 9, 4, 15)
    ]

    ps3 = [
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20),
        (3, 20, 6, 8, 5, 19, 13, 15, 11, 10, 14, 12, 17, 9, 7, 1, 4, 16, 2, 18)
    ]

    assert PQTreeBuilder.from_perms(ps1).to_parens() == "(1 [2 3] 4 [5 6] 7 8 9 10)"
    assert PQTreeBuilder.from_perms(ps2).to_parens() == "[[(1 2 3 4 5 6 7 8 9 10 11 12 13) 14] 15]"
    assert PQTreeBuilder.from_perms(ps3).to_parens() == "(1 2 3 4 5 6 7 8 9 ([10 11] 12 13 14 15) 16 17 18 19 20)"


def test_json_repr():
    perms = [
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        (9, 8, 7, 6, 3, 1, 5, 4, 2, 0)
    ]
    tree = PQTreeBuilder.from_perms(perms)
    assert tree.dict_repr() == {
        'approx_front_size': 96,
        'has_multi_chars': False,
        'root': {'children': [
            {'char': '0',
             'multi': False,
             'multi_stats': {1: '1:1'},
             'type': 'LEAF'},
            {'children': [
                {'char': '1',
                 'multi': False,
                 'multi_stats': {1: '1:1'},
                 'type': 'LEAF'},
                {'char': '2',
                 'multi': False,
                 'multi_stats': {1: '1:1'},
                 'type': 'LEAF'},
                {'char': '3',
                 'multi': False,
                 'multi_stats': {1: '1:1'},
                 'type': 'LEAF'},
                {'children': [
                    {'char': '4',
                     'multi': False,
                     'multi_stats': {1: '1:1'},
                     'type': 'LEAF'},
                    {'char': '5',
                     'multi': False,
                     'multi_stats': {1: '1:1'},
                     'type': 'LEAF'}],
                    'type': 'QNode'}],
                'type': 'PNode'},
            {'char': '6',
             'multi': False,
             'multi_stats': {1: '1:1'},
             'type': 'LEAF'},
            {'char': '7',
             'multi': False,
             'multi_stats': {1: '1:1'},
             'type': 'LEAF'},
            {'char': '8',
             'multi': False,
             'multi_stats': {1: '1:1'},
             'type': 'LEAF'},
            {'char': '9',
             'multi': False,
             'multi_stats': {1: '1:1'},
             'type': 'LEAF'}],
            'type': 'QNode'}}


if __name__ == '__main__':
    tests = [
        known_example,
        known_e2e,
        known_e2e_2,
        simple_pq_tree_size_tests,
        rand_size_tests,
        string_inputs_pqtree,
        compare_oren,
        compare_oren_from_rand,
        test_json_repr,
    ]

    for test in tests:
        print(f"----------- {test.__name__} ----------- ")
        test()
