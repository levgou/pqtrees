from collections import Set
from functools import partial

from pqtrees.common_intervals.proj_types import F_X_Y, Index, U_L_Func


def calc_w(n: int, f: F_X_Y, x: Index) -> Set[Index]:
    return {
            y for y in range(x, n) if
            all(f(x_prime, y) for x_prime in range(x))
        }


def trim_ylist(x: Index, y: Index, n: int, u: U_L_Func):
    y_star = max(filter(
        lambda :,
        range(n)
    ))



def rc(n: int, f: F_X_Y):
    def f_zero(x: Index, y: Index): return f(x, y) == 0
    calc_w_ = partial(calc_w, n, f)
    output = []

    Y = set(range(n))

    for x in reversed(range(1, n)):
        output.extend(filter(f_zero, Y))
        W = calc_w_(x)
        Y -= W
