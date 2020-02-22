from collections import Counter, defaultdict
from itertools import product, permutations
from pprint import pprint
from random import shuffle

from funcy import lmap

from pqtrees.common_intervals.pqtree import PQTreeBuilder, PQTreeVisualizer
from pqtrees.iterator_product import IterProduct


class PQTreeDup:

    @classmethod
    def from_perms(cls, perms):
        if not cls.has_duplications(perms[0]):
            yield PQTreeBuilder.from_perms(perms)
            return

        for perm_set in cls.traverse_perm_space(perms):
            print(perm_set)
            yield PQTreeBuilder.from_perms(perm_set).approx_frontier_size()

    @classmethod
    def has_duplications(cls, perm):
        counter = Counter(perm)
        if counter.most_common(1)[0][1] > 1:
            return True
        return False

    @classmethod
    def traverse_perm_space(cls, perms):
        """
        example

        for 2 perms [(1, 1, 2), (1, 2, 1)]

        the first perm is translated to (0, 1, 2)
        translation: {1: [0,1], 2: [2]}

        yield w.l.o.g:
        a) 0 1 2 ; 0 2 1
        b) 0 1 2 ; 1 2 0

        Note: as seen in the example - perms[0] is static - to avoid isomorphic results
        """

        p1 = perms[0]
        norm_p1 = range(len(p1))

        translations = defaultdict(list)
        for i, val in enumerate(p1):
            translations[val].append(norm_p1[i])

        iters = lmap(lambda p: cls.perm_variations(p, translations), perms[1:])
        for perm_set in IterProduct.iproduct([tuple(norm_p1)], *iters):
            yield perm_set

    @classmethod
    def perm_variations(cls, perm, translations):
        """
        example:

        for (4, 5, 4, 5,) with translation [0, 1, 2, 3], i.e {4: [0, 2], 5: [1, 3]}

        yield:
        0 1 2 3
        2 1 0 3
        0 3 2 1
        2 3 0 1
        """
        for trans in cls.iter_translation_permutations(translations):
            indices_in_trans_order = {k: 0 for k in translations}
            variation = []

            for c in perm:
                index_in_trans = indices_in_trans_order[c]
                variation.append(trans[c][index_in_trans])
                indices_in_trans_order[c] += 1

            yield tuple(variation)

    @classmethod
    def iter_translation_permutations(cls, translations):
        """
        example:
        for {'a': [1], 'b': [2, 3], 'c': [4, 5]}

        yield:
        [
          {'a': (1,), 'b': (2, 3), 'c': (4, 5)},
          {'a': (1,), 'b': (2, 3), 'c': (5, 4)},
          {'a': (1,), 'b': (3, 2), 'c': (4, 5)},
          {'a': (1,), 'b': (3, 2), 'c': (5, 4)}
        ]
        """
        all_translation_orders = {k: permutations(v) for k, v in translations.items()}
        for vs in product(*all_translation_orders.values()):
            yield dict(zip(all_translation_orders.keys(), vs))


def test_perm_space():
    perms1 = [
        (1, 1, 2),
        (1, 2, 1)
    ]

    perms2 = [
        (1, 2, 3, 1, 2),
        (1, 2, 1, 2, 3)
    ]

    p = (0, 1, 1, 2, 2, 3, 3, 4)
    p1, p2, p3 = list(p), list(p), list(p)
    shuffle(p1), shuffle(p2), shuffle(p3)
    p1, p2, p3 = tuple(p1), tuple(p2), tuple(p3)

    perms3 = [
        p, p1, p2, p3
    ]

    ps1 = set(PQTreeDup.traverse_perm_space(perms1))
    ps2 = set(PQTreeDup.traverse_perm_space(perms2))
    ps3 = set(PQTreeDup.traverse_perm_space(perms3))

    assert len(ps1) == 2
    assert len(ps2) == 4
    assert len(ps3) == (2 * 2 * 2) ** 3

    assert ps1 == {
        ((0, 1, 2), (1, 2, 0)),
        ((0, 1, 2), (0, 2, 1))
    }

    assert ps2 == {
        ((0, 1, 2, 3, 4), (0, 1, 3, 4, 2)),
        ((0, 1, 2, 3, 4), (0, 4, 3, 1, 2)),
        ((0, 1, 2, 3, 4), (3, 1, 0, 4, 2)),
        ((0, 1, 2, 3, 4), (3, 4, 0, 1, 2)),
    }

    size = [s for s in PQTreeDup.from_perms(perms2)]
    print("5555555555555555555555555555")
    print(size)


if __name__ == '__main__':
    # PQTreeVisualizer.show(PQTreeBuilder.from_perms(((0, 1, 2, 3, 4), (0, 4, 3, 1, 2))))
    # PQTreeVisualizer.show(PQTreeBuilder.from_perms(((0, 4, 2, 3, 1), (0, 1, 3, 4, 2))))
    # PQTreeVisualizer.show(PQTreeBuilder.from_perms((('a', 'e', 'c', 'd', 'b'), ('a', 'b', 'd', 'e', 'c'))))

    test_perm_space()
