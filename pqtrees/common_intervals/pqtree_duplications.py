import operator
from collections import Counter, defaultdict
from functools import reduce
from itertools import product, permutations, chain
from pprint import pprint
from random import shuffle
from typing import Tuple, Collection, List, Dict, Optional

from funcy import lmap, flatten, select_values, lfilter, lflatten
from heapdict import heapdict

from pqtrees.common_intervals.perm_helpers import is_list_consecutive, all_indices, irange, tmap1, group_by_attr, \
    sflatmap1, num_appear, flatmap, lflatmap, filter_cchars, all_neighbours_list, neighbours_of, sflatten, \
    char_neighbour_tuples, iter_common_neighbours, assoc_cchars_with_neighbours
from pqtrees.common_intervals.pqtree import PQTreeBuilder
from pqtrees.iterator_product import IterProduct
from pqtrees.common_intervals.generalized_letters import MultipleOccurrenceChar as MultiChar, ContextChar, MergedChar, \
    ContextPerm

Translation = Dict[Tuple[ContextChar, ContextChar], MergedChar]


class PQTreeDup:

    @classmethod
    def from_perms(cls, perms):
        if not cls.has_duplications(perms[0]):
            yield PQTreeBuilder.from_perms(perms)
            return

        for perm_set in cls.traverse_perm_space(perms):
            # print(perm_set)
            yield PQTreeBuilder.from_perms(perm_set)

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

    @classmethod
    def find_multi_chars(cls, id_context_perm: Tuple[ContextChar]):
        count = defaultdict(list)
        for context_char in id_context_perm:
            count[context_char.char].append(context_char)
        return select_values(lambda lst: len(lst) > 1, count)

    @classmethod
    def can_merge_multi_chars(cls, context_perms):
        """
        Will return a dictionary of the structure:
        {
            char1: {
                        perm1: [ContextChar1, ..., ContextCharK]
                        perm2: [...]
                    }

            char2: {...}
        }

        each context char has common neighbour with at LEAST ONE context char for EACH other perm
        """
        more_than_once = cls.find_multi_chars(context_perms[0])

        def common_neighbour_set(char):
            perm_neighbours = lambda cperm: set(flatten(
                [cc.left_char, cc.right_char] for cc in cperm if cc.char == char
            ))

            common_neigbours = reduce(operator.__and__, map(perm_neighbours, context_perms))
            return common_neigbours - {None}

        def neighbours_of(char_col, char):
            return [cc for cc in char_col if cc.left_char == char or cc.right_char == char]

        mergable_chars = {}
        for char in more_than_once:
            neighbours = common_neighbour_set(char)
            if neighbours:
                cc_anywhere = lfilter(lambda cc: cc.char == char, chain(*context_perms))
                cc_with_common_neighbours = sflatmap1(neighbours_of, cc_anywhere, neighbours)
                mergable_chars_per_perm = group_by_attr('perm', cc_with_common_neighbours)
                mergable_chars[char] = mergable_chars_per_perm
                # mergable_chars[char] = neighbours_of(filter(lambda cc: cc.char == char, chain(*context_perms)), char)

        return mergable_chars

    @classmethod
    def merge_multi_chars(cls, perms):
        context_perms = cls.to_context_chars(perms)
        mergable_chars = cls.can_merge_multi_chars(context_perms)

        maybe_no_loss_mergable_chars = [
            char for char, cc_per_perm in mergable_chars.items()
            if len(cc_per_perm[perms[0]]) == num_appear(perms[0], char)
        ]

        return cls.try_merge_chars_no_loss(context_perms, mergable_chars, maybe_no_loss_mergable_chars)
        pprint(mergable_chars)

    @classmethod
    def can_reduce_chars(cls, id_perm, others):
        count = Counter(id_perm)
        more_than_once = {k: v for k, v in count.items() if v > 1}

        compactable_chars = []
        for char in more_than_once:
            for perm in others:
                indices = all_indices(perm, char)
                if not is_list_consecutive(indices):
                    break
            else:
                compactable_chars.append(char)

        return {char: n for char, n in count.items() if char in compactable_chars}

    @classmethod
    def reduce_multi_chars(cls, perms):
        reducibale_chars = cls.can_reduce_chars(perms[0], perms[1:])
        multi_chars = {char: MultiChar(char, count) for char, count in reducibale_chars.items()}

        def reduce_perm(perm):
            seen = set()
            new_perm = []
            for char in perm:
                if char not in reducibale_chars:
                    new_perm.append(char)
                else:
                    if char not in seen:
                        new_perm.append(multi_chars[char])
                    seen.add(char)
            return tuple(new_perm)

        reduced_perms = tuple(reduce_perm(p) for p in perms)
        return reduced_perms

    @classmethod
    def to_context_chars(cls, perms):
        return tuple(
            tmap1(ContextChar.from_perm_index, p, irange(p))
            for p in perms
        )

    @classmethod
    def try_merge_chars_no_loss(cls, context_perms: Collection[ContextPerm],
                                mergable_chars: Dict[object, Dict[Collection, List[ContextChar]]],
                                chars_to_merge: Collection) -> Optional[Translation]:
        translation = {}

        # def construct_neighbour_occurrence_heap(relevant_cchars):
        #     all_neighbours = all_neighbours_list(relevant_cchars)
        #     count = Counter(all_neighbours)
        #     return heapdict(count)

        # for char in chars_to_merge:
        # relevant_cchars = filter_cchars(char, *context_perms)
        # heap = construct_neighbour_occurrence_heap(relevant_cchars)
        # heap2 = heapdict(heap)

        for char in chars_to_merge:
            if char_trans := cls.try_merge_char(context_perms, mergable_chars[char]):
                translation.update(char_trans)
            else:
                return None

    @classmethod
    def try_merge_char(cls,
                       context_perms: Collection[ContextPerm],
                       cur_mergable_chars: Dict[Collection, List[ContextChar]]) -> Optional[Translation]:

        def iter_pairing_space():
            return IterProduct.iproduct(*[permutations(cchars) for cchars in cur_mergable_chars.values()])

        def try_translate_pairing(pairing, cur_translation=None, already_used=None):
            cur_translation = cur_translation or {}
            already_used = already_used or set()

            l_pairing = list(pairing)
            while cur_cchars := l_pairing.pop():
                for neighbour in iter_common_neighbours(*cur_cchars):

                    cchars_with_neighbours_tuples = assoc_cchars_with_neighbours(cur_cchars, neighbour, context_perms)

                    if any(cc in already_used for cc in chain(cchars_with_neighbours_tuples)):
                        continue

                    merged_char = MergedChar.from_occurrences(char_neighbour_tuples(cur_cchars, neighbour))
                    updated_translation = {
                        **cur_translation,
                        **{t: merged_char for t in cchars_with_neighbours_tuples}
                    }

                    new_used = already_used | set(chain(cchars_with_neighbours_tuples))

                    if trans := try_translate_pairing(l_pairing, updated_translation, new_used):
                        return trans

                return None

        for pairing in iter_pairing_space():
            if trans := try_translate_pairing(pairing):
                return trans
        return None


#         if cur_translation is None:
#             cur_translation = dict()
#
#         if not heapdict:
#             return cur_translation
#
#         heap = heapdict(neighbour_heap)
#         char, freq = heap.popitem()
#
#         if freq < len(cur_mergable_chars):
#             return None
#
#         cchars_per_perm = {perm: neighbours_of(char, perm) for perm in cur_mergable_chars}
#
#         if all(len(cchars) == 1 for cchars in cchars_per_perm.values()):
#             cchar_to_rm = sflatten(cchars_per_perm.values())
#             mergable_chars = {
#                 perm: lfilter(lambda cc: cc not in cchar_to_rm, cchars)
#                 for perm, cchars in cur_mergable_chars.items()
#             }
#             updated_translation = {
#                 **cur_translation,
#                 **{cchar: MergedChar.from_occurrences(char_neighbour_tuples(cchar_to_rm, char))
#                    for cchar in cchar_to_rm}
#             }
#
# # todo update heap
#             return cls.try_merge_char(heap, mergable_chars, updated_translation)
#
#         else:


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


def test_reduce_perms():
    tests = [
        [
            [(1, 2, 3), (3, 2, 1)],
            ((1, 2, 3), (3, 2, 1))
        ],

        [
            [(1, 2, 2, 3), (3, 2, 2, 1)],
            ((1, MultiChar(2, 2), 3), (3, MultiChar(2, 2), 1))
        ],

        [
            [(1, 2, 2, 2, 3), (3, 2, 2, 2, 1)],
            ((1, MultiChar(2, 3), 3), (3, MultiChar(2, 3), 1))
        ],

        [
            [(1, 1, 2, 2, 3), (3, 2, 2, 1, 1)],
            ((MultiChar(1, 2), MultiChar(2, 2), 3), (3, MultiChar(2, 2), MultiChar(1, 2)))
        ],

        [
            [(1, 1, 2, 2, 3, 3), (3, 3, 2, 2, 1, 1)],
            ((MultiChar(1, 2), MultiChar(2, 2), MultiChar(3, 2)),
             (MultiChar(3, 2), MultiChar(2, 2), MultiChar(1, 2)))
        ],

        [
            [(0, 0, 1, 2, 2, 3, 4, 5, 5), (5, 5, 4, 3, 2, 2, 1, 0, 0)],

            ((MultiChar(0, 2), 1, MultiChar(2, 2), 3, 4, MultiChar(5, 2)),
             (MultiChar(5, 2), 4, 3, MultiChar(2, 2), 1, MultiChar(0, 2)))
        ],

        [
            [(1, 2, 2, 3), (3, 2, 2, 1), (1, 3, 2, 2), (2, 2, 1, 3)],
            ((1, MultiChar(2, 2), 3), (3, MultiChar(2, 2), 1), (1, 3, MultiChar(2, 2)), (MultiChar(2, 2), 1, 3))
        ],
    ]

    for t_input, expected in tests:
        res = PQTreeDup.reduce_multi_chars(t_input)
        assert res == expected


def test_context_char_conversion():
    tests = [
        [tuple(tuple()), tuple(tuple())],
        [((1,),), ((ContextChar(None, 1, None),),)],
        [((1, 2),), ((ContextChar(None, 1, 2), ContextChar(1, 2, None)),)],
        [((1, 2, 3),), ((ContextChar(None, 1, 2), ContextChar(1, 2, 3), ContextChar(2, 3, None)),)],
        [((1, 2, 3), (1, 2, 3)),
         ((ContextChar(None, 1, 2), ContextChar(1, 2, 3), ContextChar(2, 3, None)),
          (ContextChar(None, 1, 2), ContextChar(1, 2, 3), ContextChar(2, 3, None)))],

    ]

    for t_in, expected in tests:
        assert PQTreeDup.to_context_chars(t_in) == expected, expected


def test_merge_chars():
    tests = [
        [
            (
                (1, 2, 3, 1),
                (1, 2, 1, 3)
            ),
            (
                (1, 2, MergedChar.from_occurrences((3, 1), (1, 3))),
                (1, 2, MergedChar.from_occurrences((3, 1), (1, 3)))
            ),
        ]
    ]
    for t_in, expect in tests:
        PQTreeDup.merge_multi_chars(t_in)


if __name__ == '__main__':
    # PQTreeVisualizer.show(PQTreeBuilder.from_perms(((0, 1, 2, 3, 4), (0, 4, 3, 1, 2))))
    # PQTreeVisualizer.show(PQTreeBuilder.from_perms(((0, 4, 2, 3, 1), (0, 1, 3, 4, 2))))
    # PQTreeVisualizer.show(PQTreeBuilder.from_perms((('a', 'e', 'c', 'd', 'b'), ('a', 'b', 'd', 'e', 'c'))))

    # test_perm_space()
    # test_reduce_perms()
    # test_context_char_conversion()
    test_merge_chars()
    trees = (list(PQTreeDup.from_perms([(1, 2, 3, 4, 1), (1, 3, 4, 2, 1), (1, 4, 3, 2, 1)])))
    print([t.approx_frontier_size() for t in trees][-1])

