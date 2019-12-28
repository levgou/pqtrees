from typing import Sequence, List

from pqtrees.common_intervals.common_interval import CommonInterval
from pqtrees.common_intervals.proj_types import SigmaInvFunc, SigmaFunc
from pqtrees.y_lists import YLists

"""
sig_a = [1, 2, 3, 4, 5, 6, 7]
sig_b = [5, 3, 1, 4, 2, 7, 6]


"""


# def pairwise(iterable) -> Iterable[tuple]:
#     """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
#     a, b = tee(iterable)
#     next(b, None)
#     return zip(a, b)
#
#
# def calc_w(n: int, f: F_X_Y, x: Index) -> Set[Index]:
#     return {
#         y for y in range(x, n) if
#         all(f(x_prime, y) for x_prime in range(x))
#     }

#
# def trim_ylist(x: Index, y_list: deque, n: int, u_func: U_L_Func):
#     y_star = max(filter(
#         lambda y: u_func(x, y) < u_func(x - 1, y),
#         range(n)
#     ))
#
#     while y_list:
#         if u_func(x, y_list[0]) < u_func(x, y_star):
#             y_list.popleft()
#         else:
#             break
#
#     while len(y_list) >= 2:
#         for y_i, y_i_1 in pairwise(y_list):


def rc(a: Sequence, b: Sequence) -> List[CommonInterval]:
    n: int = len(a)
    sig_a: SigmaFunc = a.__getitem__
    b_inv_index = dict(zip(b, range(len(b))))
    sig_b_inv: SigmaInvFunc = b_inv_index.__getitem__

    output: List[CommonInterval] = []
    Y = YLists(sig_a, sig_b_inv, n)

    # we are not interested in the simple case where x = y, and the're both the end index
    Y = Y.decrease_x()

    for x in reversed(range(n - 1)):
        yul_fxy_zero = [yul for yul in Y.yuls_fxy_zero(x)]
        common_in_both = [((x, yul.y), (yul.l, yul.u)) for yul in yul_fxy_zero]
        common_intervals = [CommonInterval(a_cmn, b_cmn) for a_cmn, b_cmn in common_in_both]
        output.extend(common_intervals)
        Y = Y.decrease_x()

    output_no_singletons = list(filter(lambda ci: len(ci) > 1, output))

    if CommonInterval((0, len(a) - 1), (0, len(a) - 1)) not in output_no_singletons:
        output_no_singletons.append(CommonInterval((0, len(a) - 1), (0, len(a) - 1)))

    return output_no_singletons
