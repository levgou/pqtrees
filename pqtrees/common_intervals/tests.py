import inspect
import random
from typing import Callable, Tuple

from pqtrees.common_intervals.bsc import bsc, bsc_k
from pqtrees.common_intervals.common_interval import CommonInterval
from pqtrees.common_intervals.generate_s import IntervalHierarchy
from pqtrees.common_intervals.lhp import lhp
from pqtrees.common_intervals.pqtree import PQTree, PQTreeBuilder, PQTreeVisualizer
from pqtrees.common_intervals.reduce_candidate import rc
from pqtrees.common_intervals.reduce_intervals import ReduceIntervals
from pqtrees.common_intervals.trivial import trivial_common, trivial_common_k, trivial_common_k_with_singletons
from pqtrees.common_intervals.preprocess_find import common_k_indexed_with_singletons, common_k_indexed
from timeit import default_timer as timer
from pprint import pprint

__DEBUG__ = False


def time_runtime(callable: Callable) -> Tuple[object, float]:
    start = timer()
    res = callable()
    end = timer()
    return res, end - start


def test_results(commons, alg, *perms):
    caller_name = inspect.stack()[1].function

    if __DEBUG__:
        print(f"Test: {caller_name}")

    found_commons = alg(*perms)
    common_set = set(commons)
    found_common_set = set(found_commons)

    perms_str = '\n'.join((str(perm) for perm in perms))
    summary = f"""

Test: {caller_name}
For alg [{alg.__name__}]
And strings:
{perms_str}
Found: {found_common_set}
Known: {common_set}
Only in found: {found_common_set - common_set}
Only in known: {common_set - found_common_set}
"""

    if __DEBUG__:
        print(summary)

    assert found_common_set == common_set, summary


def test_common_intervals_len_4(alg):
    sig1 = (0, 1, 2, 3)
    sig2 = (3, 0, 2, 1)

    commons = [
        # len 2
        CommonInterval((1, 2), (2, 3)),

        # len 3
        CommonInterval((0, 2), (1, 3)),

        # len 4
        CommonInterval((0, 3), (0, 3)),
    ]

    test_results(commons, alg, sig1, sig2)


def test_common_intervals_len_5(alg):
    sig1 = (0, 1, 2, 3, 4)
    sig2 = (4, 3, 0, 2, 1)

    commons = [
        # len 2
        CommonInterval((1, 2), (3, 4)),
        CommonInterval((3, 4), (0, 1)),

        # len 3
        CommonInterval((0, 2), (2, 4)),

        # len 4
        CommonInterval((0, 3), (1, 4)),

        # len 5
        CommonInterval((0, 4), (0, 4)),
    ]

    test_results(commons, alg, sig1, sig2)


def test_common_intervals_len_5_b(alg):
    sig1 = (0, 1, 2, 3, 4)
    sig2 = (3, 1, 4, 0, 2)

    commons = [
        # len 5
        CommonInterval((0, 4), (0, 4)),
    ]

    test_results(commons, alg, sig1, sig2)


def test_common_intervals_len_5_c(alg):
    sig1 = (0, 1, 2, 3, 4)
    sig2 = (0, 2, 4, 1, 3)

    commons = [
        # len 4
        CommonInterval((1, 4), (1, 4)),

        # len 5
        CommonInterval((0, 4), (0, 4)),
    ]

    test_results(commons, alg, sig1, sig2)


def test_common_intervals_len_9(alg):
    sig1 = (0, 1, 2, 3, 4, 5, 6, 7, 8)
    sig2 = (8, 7, 3, 4, 5, 6, 0, 1, 2)

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

    test_results(commons, alg, sig1, sig2)


def test_common_intervals_3_strings_len_9(alg):
    pi_1 = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    pi_2 = [8, 7, 3, 4, 5, 6, 0, 1, 2]
    pi_3 = [0, 1, 2, 7, 6, 3, 4, 5, 8]

    commons = [
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

    test_results(commons, alg, pi_1, pi_2, pi_3)


def test_common_intervals_3_strings_len_5(alg):
    pi_1 = [0, 1, 2, 3, 4]
    pi_2 = [1, 0, 3, 4, 2]
    pi_3 = [1, 4, 3, 0, 2]

    commons = [

        # len 2
        CommonInterval((3, 4), (2, 3), (1, 2)),

        # len 5
        CommonInterval((0, 4), (0, 4), (0, 4)),
    ]

    test_results(commons, alg, pi_1, pi_2, pi_3)


def test_common_intervals_2_perms(alg):
    print(f"\n--------------------------- Testing {alg.__name__} ---------------------------")
    test_common_intervals_len_4(alg)
    test_common_intervals_len_5(alg)
    test_common_intervals_len_5_b(alg)
    test_common_intervals_len_5_c(alg)
    test_common_intervals_len_9(alg)


def test_common_intervals_k_perms(alg):
    print(f"\n--------------------------- Testing[k] {alg.__name__} ---------------------------")
    test_common_intervals_3_strings_len_5(alg)
    test_common_intervals_3_strings_len_9(alg)


def test_rand_perms(algs: tuple,
                    min_len_perm: int,
                    max_len_perm: int,
                    perm_len_jump: int,
                    min_num_perms: int,
                    max_num_perms: int,
                    repeat_test_times: int
                    ):
    names = [alg.__name__ for alg in algs]
    other_algs = algs[1:]
    other_names = names[1:]

    total_times = {name: 0 for name in names}
    print(f"\n ->> Testing random k [{min_num_perms}-{max_num_perms}] inputs for {names}")

    first_alg = algs[0]
    first_alg_name = first_alg.__name__

    for length in range(min_len_perm, max_len_perm + 1, perm_len_jump):
        print(f"Len perm: {length}")
        for num_perms in range(min_num_perms, max_num_perms + 1):
            for _ in range(repeat_test_times):

                sig_a = list(range(length))
                other_perms = [list(sig_a) for _ in range(num_perms - 1)]
                for p in other_perms:
                    random.shuffle(p)

                first_alg_res, first_cur_rt = time_runtime(lambda: first_alg(sig_a, *other_perms))
                total_times[first_alg_name] += first_cur_rt

                for alg, alg_name in zip(other_algs, other_names):
                    _, cur_rt = time_runtime(lambda: test_results(first_alg_res, alg, sig_a, *other_perms))
                    total_times[alg_name] += cur_rt

    pprint(total_times)


def test_rand_k_perms_comp_all_algs(*algs):
    test_rand_perms(
        algs,
        min_len_perm=50,
        max_len_perm=60,
        perm_len_jump=10,
        min_num_perms=98,
        max_num_perms=100,
        repeat_test_times=3,
    )


def test_rand_perm_comp_all_algs(*algs):
    test_rand_perms(
        algs,
        min_len_perm=5,
        max_len_perm=65,
        perm_len_jump=10,
        min_num_perms=2,
        max_num_perms=2,
        repeat_test_times=3,
    )


def test_pq_tree_construction():
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
        pprint(s.nesting_levels)

        pqtree = PQTreeBuilder.from_s(s, None)
        print(pqtree.to_parens())
        assert pqtree.to_parens() == "[[0 1 2] [[[3 4 5] 6] 7] 8]"

        # pqtree.show()

    def known_e2e():
        """From wikipedia PQTree entry """

        perms = [(0, 1, 2, 3, 4), (0, 1, 3, 2, 4), (0, 2, 1, 3, 4),
                 (0, 2, 3, 1, 4), (0, 3, 1, 2, 4), (0, 3, 2, 1, 4),
                 (4, 1, 2, 3, 0), (4, 1, 3, 2, 0), (4, 2, 1, 3, 0),
                 (4, 2, 3, 1, 0), (4, 3, 1, 2, 0), (4, 3, 2, 1, 0)]

        strs = {"".join(str(x) for x in p) for p in perms}

        common_intervals = trivial_common_k_with_singletons(*perms)
        common_intervals = common_k_indexed_with_singletons(*perms)

        ir_intervals = ReduceIntervals.reduce(common_intervals)
        s = IntervalHierarchy.from_irreducible_intervals(ir_intervals)

        pprint(s.nesting_levels)

        pqtree = PQTreeBuilder.from_s(s)
        frontier = pqtree.frontier()
        print(pqtree.to_parens())
        assert pqtree.to_parens() == "[0 (1 2 3) 4]"
        assert strs.issubset(frontier), strs - frontier

        # pqtree.show()

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

        pprint(s.nesting_levels)

        pqtree = PQTreeBuilder.from_s(s)
        frontier = pqtree.frontier()

        print(pqtree.to_parens())
        assert pqtree.to_parens() == "[0 (1 2 3 [4 5]) 6 7 8 9]"
        assert strs.issubset(frontier), strs - frontier

        # pqtree.show()

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

            pprint(s.nesting_levels)

            pqtree = PQTreeBuilder.from_s(s)
            frontier = pqtree.frontier()

            print(pqtree.to_parens())
            assert strs.issubset(frontier), strs - frontier
            assert pqtree.to_parens() == paren_repr
            assert pqtree.approx_frontier_size() == front_size

            # PQTreeVisualizer.show(pqtree)

        for t in tests:
            run_tests(*t)

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

            print(">>>", p_repr_translated, str_parens)
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

        print(PQTreeBuilder.from_perms(ps1).to_parens())
        print(PQTreeBuilder.from_perms(ps2).to_parens())
        print(PQTreeBuilder.from_perms(ps3).to_parens())

        assert PQTreeBuilder.from_perms(ps1).to_parens() == "(1 [2 3] 4 [5 6] 7 8 9 10)"
        assert PQTreeBuilder.from_perms(ps2).to_parens() == "[[(1 2 3 4 5 6 7 8 9 10 11 12 13) 14] 15]"
        assert PQTreeBuilder.from_perms(ps3).to_parens() == "(1 2 3 4 5 6 7 8 9 ([10 11] 12 13 14 15) 16 17 18 19 20)"

    known_example()
    known_e2e()
    known_e2e_2()
    simple_pq_tree_size_tests()
    string_inputs_pqtree()
    compare_oren()
    compare_oren_from_rand()


if __name__ == '__main__':
    test_common_intervals_2_perms(trivial_common)
    test_common_intervals_2_perms(trivial_common_k)
    test_common_intervals_2_perms(bsc)
    test_common_intervals_2_perms(bsc_k)
    test_rand_perm_comp_all_algs(trivial_common, trivial_common_k, bsc, bsc_k, common_k_indexed)

    test_common_intervals_k_perms(trivial_common_k)
    test_common_intervals_k_perms(bsc_k)

    test_rand_k_perms_comp_all_algs(trivial_common_k, bsc_k, common_k_indexed)
    test_pq_tree_construction()
