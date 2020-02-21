class IterProduct:
    _SENTINEL = object()

    @classmethod
    def append(cls, it, elems: list):
        e = next(it, cls._SENTINEL)
        if e is not cls._SENTINEL:
            elems.append(e)

    @classmethod
    def iproduct2(cls, iterable1, iterable2):
        """Cartesian product of two possibly infinite iterables"""

        it1 = iter(iterable1)
        it2 = iter(iterable2)

        elems1 = []
        elems2 = []

        n = 0
        cls.append(it1, elems1)
        cls.append(it2, elems2)

        while n <= len(elems1) + len(elems2):
            for m in range(n - len(elems1) + 1, len(elems2)):
                yield elems1[n - m], elems2[m]
            n += 1
            cls.append(it1, elems1)
            cls.append(it2, elems2)

    @classmethod
    def iproduct(cls, *iterables):
        """Returns Cartesian product of iterables. Yields every element
        eventually"""
        if len(iterables) == 0:
            yield ()
            return
        elif len(iterables) == 1:
            for e in iterables[0]:
                yield e,
        elif len(iterables) == 2:
            for e12 in cls.iproduct2(*iterables):
                yield e12
        else:
            first, others = iterables[0], iterables[1:]
            for ef, eo in cls.iproduct2(first, cls.iproduct(*others)):
                yield (ef,) + eo


if __name__ == '__main__':

    def xx():
        for i in range(10):
            yield i


    for x in IterProduct.iproduct(xx(), range(300), range(300)):
        if not x[0] % 100 and not x[1] % 100 and not x[2] % 100:
            print(x)
