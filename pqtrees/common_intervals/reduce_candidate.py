from collections import Set, deque, Iterable
from functools import partial
from itertools import tee

from pqtrees.common_intervals.proj_types import F_X_Y, Index, U_L_Func

"""
sig_a = [1, 2, 3, 4, 5, 6, 7]
sig_b = [5, 3, 1, 4, 2, 7, 6]


"""


def pairwise(iterable) -> Iterable[tuple]:
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def calc_w(n: int, f: F_X_Y, x: Index) -> Set[Index]:
    return {
        y for y in range(x, n) if
        all(f(x_prime, y) for x_prime in range(x))
    }


def trim_ylist(x: Index, y_list: deque, n: int, u_func: U_L_Func):
    y_star = max(filter(
        lambda y: u_func(x, y) < u_func(x - 1, y),
        range(n)
    ))

    while y_list:
        if u_func(x, y_list[0]) < u_func(x, y_star):
            y_list.popleft()
        else:
            break

    while len(y_list) >= 2:
        for y_i, y_i_1 in pairwise(y_list):


def rc(n: int, f: F_X_Y):
    def f_zero(x: Index, y: Index): return f(x, y) == 0


    calc_w_ = partial(calc_w, n, f)
    output = []

    Y = set(range(n))

    for x in reversed(range(1, n)):
        output.extend(filter(f_zero, Y))
        W = calc_w_(x)
        Y -= W
