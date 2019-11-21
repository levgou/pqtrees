from typing import List

from pqtrees.common_intervals.bsc import CommonInterval, Interval
from pqtrees.common_intervals.proj_types import Index, IntSeq


def can_break_inner_loop(u: Index, l: Index, n: int, x: Index, interval: Interval) -> bool:
    if u - l > min(n - x, n - 3):
        return True

    hp_prime = {
        interval.pi_a_b(w) for w in
        {
            z for z in range(n)
            if z % n == (x - 2) % n or z % n == (x - 1) % n
        }
    }

    if any(l < h < u for h in hp_prime):
        return True

    return False


def add_edge_cases(perm_a: IntSeq, perm_b: IntSeq, n: int, output: List[CommonInterval]):
    output.append(CommonInterval((0, n - 1), (0, n - 1)))

    if perm_a[0] == perm_b[0]:
        output.append(CommonInterval((1, n - 1), (1, n - 1)))

    elif perm_a[0] == perm_b[-1]:
        output.append(CommonInterval((1, n - 1), (0, n - 2)))

    elif perm_a[-1] == perm_b[0]:
        output.append(CommonInterval((0, n - 2), (1, n - 1)))

    if perm_a[-1] == perm_b[-1]:
        output.append(CommonInterval((0, n - 2), (0, n - 2)))


def lhp(perm_a: IntSeq, perm_b: IntSeq) -> List[CommonInterval]:
    assert len(perm_a) == len(perm_b)

    n = len(perm_a)
    output = []
    interval = Interval(
        sig_a=perm_a.__getitem__,
        sig_a_inv=lambda v: perm_a.index(v),
        sig_b=perm_b.__getitem__,
        sig_b_inv=lambda v: perm_b.index(v),
    )

    add_edge_cases(perm_a, perm_b, n, output)

    for x in range(n - 1):
        l = u = interval.pi_a_b(x)

        upper_y_limit = min(n, x + n - 2)
        for y in range(x + 1, upper_y_limit):
            l = min(l, interval.pi_a_b(y))
            u = max(u, interval.pi_a_b(y))

            if can_break_inner_loop(u, l, n, x, interval):
                break

            if u - l - (y - x) == 0:
                output.append(CommonInterval((x, y), (l, u)))

    return output
