"""

The basic algorithm to find all common intervals of 2 permutation of {1...n}
"""

from typing import Callable, Iterable, List, Sequence, Set

from pqtrees.common_intervals.common_interval import CommonInterval, CommonIntervalWeakEq
from pqtrees.proj_types import Index, SigmaFunc, SigmaInvFunc


class IntervalAndFuncs:
    sig_a: SigmaFunc
    sig_a_inv: SigmaInvFunc
    sig_b: SigmaFunc
    sig_b_inv: SigmaInvFunc


    def __init__(self,
                 sig_a: SigmaFunc,
                 sig_a_inv: SigmaInvFunc,
                 sig_b: SigmaFunc,
                 sig_b_inv: SigmaInvFunc
                 ) -> None:
        super().__init__()

        self.sig_a = sig_a
        self.sig_a_inv = sig_a_inv
        self.sig_b = sig_b
        self.sig_b_inv = sig_b_inv


    def pi_a_b(self, i: Index) -> Index:
        return self.sig_b_inv(self.sig_a(i))


    def l_u(self, x: Index, y: Index, consumer: Callable[[Iterable[int]], int]) -> Index:
        return consumer(self.pi_a_b(i) for i in range(x, y + 1))


    def l(self, x: Index, y: Index) -> Index:
        return self.l_u(x, y, min)


    def u(self, x: Index, y: Index) -> Index:
        return self.l_u(x, y, max)


    def f(self, x: Index, y: Index) -> int:
        return self.u(x, y) - self.l(x, y) - (y - x)


def bsc(perm_a: Sequence, perm_b: Sequence, weak_ci: bool = False) -> List[CommonInterval]:
    assert len(perm_a) == len(perm_b)

    CI = CommonInterval if not weak_ci else CommonIntervalWeakEq

    n = len(perm_a)
    output = []
    interval = IntervalAndFuncs(
        sig_a=perm_a.__getitem__,
        sig_a_inv=lambda v: perm_a.index(v),
        sig_b=perm_b.__getitem__,
        sig_b_inv=lambda v: perm_b.index(v),
    )

    for x in range(n - 1):
        l = u = interval.pi_a_b(x)

        for y in range(x + 1, n):
            l = min(l, interval.pi_a_b(y))
            u = max(u, interval.pi_a_b(y))

            if u - l - (y - x) == 0:
                output.append(CI((x, y), (l, u)))

    return output


def add_indeces(output: Set[CommonInterval], commons: Set[CommonInterval]) -> Set[CommonInterval]:
    new_output = set()
    commons_dict = dict(zip(commons, commons))
    for ci in output:
        in_common = commons_dict[ci]
        assert len(in_common.intervals) == 2
        new_output.add(CommonIntervalWeakEq(*ci.intervals, in_common.intervals[-1]))

    return new_output


def bsc_k(*perms: Sequence) -> List[CommonInterval]:
    assert len({len(p) for p in perms}) == 1, perms

    perm_a = perms[0]
    perm_b = perms[1]

    output = set(bsc(perm_a, perm_b, weak_ci=True))

    for other_perm in perms[2:]:
        commons = set(bsc(perm_a, other_perm, weak_ci=True))
        output = {ci for ci in output if ci in commons}

        output = add_indeces(output, commons)

    return [CommonInterval(*ci.intervals) for ci in output]
