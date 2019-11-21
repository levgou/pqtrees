"""

The basic algorithm to find all common intervals of 2 permutation of {1...n}
"""

from typing import Callable, Iterable, Tuple, List, Sequence

from pqtrees.common_intervals.proj_types import Index, SigmaFunc, SigmaInvFunc


class Interval:
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


class CommonInterval:

    def __init__(self, *args: Tuple[Index, Index]) -> None:
        super().__init__()
        self.intervals = tuple(args)


    def __str__(self) -> str:
        return f'{self.intervals}'


    __repr__ = __str__


    def __eq__(self, o: object) -> bool:
        if not isinstance(o, CommonInterval):
            return False
        return self.intervals == o.intervals


    def __hash__(self) -> int:
        return hash(self.intervals)


def bsc(perm_a: Sequence, perm_b: Sequence) -> List[CommonInterval]:
    assert len(perm_a) == len(perm_b)

    n = len(perm_a)
    output = []
    interval = Interval(
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
                output.append(CommonInterval((x, y), (l, u)))

    return output
