import cProfile
import inspect
import random
from pprint import pprint
from timeit import default_timer as timer
from typing import Callable, Tuple

from pqtrees.common_intervals.bsc import bsc, bsc_k
from pqtrees.common_intervals.common_interval import CommonInterval
from pqtrees.common_intervals.preprocess_find import common_k_indexed
from pqtrees.common_intervals.trivial import trivial_common, trivial_common_k

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

    t_perms = map(tuple, perms)
    found_commons = alg(*t_perms)
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


def test_all_algs_2_perms():
    algs = [
        trivial_common,
        trivial_common_k,
        bsc,
        bsc_k,
    ]

    for alg in algs:
        test_common_intervals_2_perms(alg)


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

    for i, length in enumerate(range(min_len_perm, max_len_perm + 1, perm_len_jump)):
        print(f"{i + 1}. Len perm: {length}")

        for num_perms in range(min_num_perms, max_num_perms + 1):
            for _ in range(repeat_test_times):

                sig_a = list(range(length))
                other_perms = [list(sig_a) for _ in range(num_perms - 1)]
                for p in other_perms:
                    random.shuffle(p)

                t_others = tuple(map(tuple, other_perms))
                print((tuple(sig_a), *t_others))
                first_alg_res, first_cur_rt = time_runtime(lambda: first_alg(sig_a, *t_others))
                total_times[first_alg_name] += first_cur_rt

                for alg, alg_name in zip(other_algs, other_names):
                    _, cur_rt = time_runtime(lambda: test_results(first_alg_res, alg, sig_a, *other_perms))
                    total_times[alg_name] += cur_rt

    print("\nRuntimes:")
    pprint(total_times)


def test_rand_k_perms_comp_all_algs(*algs):
    # small amount of perms
    test_rand_perms(
        algs,
        min_len_perm=5,
        max_len_perm=70,
        perm_len_jump=1,
        min_num_perms=5,
        max_num_perms=10,
        repeat_test_times=3,
    )

    # large amount of perms
    test_rand_perms(
        algs,
        min_len_perm=50,
        max_len_perm=70,
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


if __name__ == '__main__':
    test_all_algs_2_perms()
    test_rand_perm_comp_all_algs(trivial_common, trivial_common_k, bsc, bsc_k, common_k_indexed)

    test_common_intervals_k_perms(trivial_common_k)
    test_common_intervals_k_perms(bsc_k)

    test_rand_k_perms_comp_all_algs(common_k_indexed, trivial_common_k , bsc_k)
