import operator
from _operator import itemgetter
from collections import Counter, defaultdict, namedtuple
from copy import deepcopy
from dataclasses import dataclass
from functools import reduce
from itertools import product, permutations, chain
from typing import Tuple, Collection, List, Dict, Optional, Sequence, Mapping, Iterable

from frozendict import frozendict
from funcy import lmap, flatten, select_values, lfilter, merge, get_in

from pqtrees.common_intervals.common_interval import CommonInterval
from pqtrees.generalized_letters import (
    MultipleOccurrenceChar as MultiChar, ContextChar, MergedChar, ContextPerm,
    char_neighbour_tuples, iter_common_neighbours)

from pqtrees.utilities.perm_helpers import (
    irange, tmap1, group_by_attr, sflatmap1, num_appear,
    assoc_cchars_with_neighbours,
    tmap, iter_char_occurrence, diff_abc, diff_len, invert_dict_multi_val)

from pqtrees.pqtree import PQTreeBuilder, PQTree, LeafNode, QNode, PNode, PQTreeVisualizer
from pqtrees.proj_types import Permutations
from pqtrees.common_intervals.trivial import window
from pqtrees.utilities.iterator_product import IterProduct

Translation = Tuple[Mapping[Tuple[ContextChar, ContextChar], MergedChar]]
CPermutations = Sequence[Sequence[ContextChar]]
MultiCharIndex = Mapping[int, Dict[int, MultiChar]]

TranslatedChar = namedtuple('TranslatedChar', ['val', 'org'])


class PQTreeDup:

    @classmethod
    def from_perms(cls, perms):
        if not cls.has_duplications(perms[0]):
            yield PQTreeBuilder.from_perms(perms)
            return

        for perm_set in cls.traverse_perm_space(perms):
            yield PQTreeBuilder.from_perms(perm_set)

    @classmethod
    def from_perms_with_merge(cls, perms):
        for trans_perms, translation in cls.merge_multi_chars(perms):
            raw_tree = PQTreeBuilder.from_perms(trans_perms)
            final_tree = cls.process_merged_chars(raw_tree, translation, perms)
            yield final_tree

    @classmethod
    def from_perms_wth_multi(cls, perms) -> Optional[PQTree]:
        if len(set(perms[0])) == len(perms[0]):
            return PQTreeBuilder.from_perms(perms)

        multi_perms = cls.reduce_multi_chars(perms)
        norm_perms, multi_char_indices = cls.multi_chars_to_regular_chars(multi_perms)

        if diff_abc(norm_perms) or diff_len(norm_perms):
            return None

        norm_tree = PQTreeBuilder.from_perms(norm_perms)
        multi_tree = cls.update_multi_leafs(norm_tree, multi_char_indices)
        return multi_tree

    @classmethod
    def process_merged_chars(cls, raw_tree: PQTree,
                             translation: Tuple[Mapping[Tuple[ContextChar, ContextChar], MergedChar]],
                             perms):

        assert len(translation) == 1

        tree_copy = deepcopy(raw_tree)
        reverse_translation = invert_dict_multi_val(translation[0])
        for node, parent in tree_copy:
            if not isinstance(node, LeafNode):
                continue
            if not isinstance(node.ci.alt_sign, MergedChar):
                continue

            mc: MergedChar = node.ci.alt_sign
            cc1, cc2 = reverse_translation[mc][0]
            first_ci_start = node.ci.first_start
            first_leaf_ci = CommonInterval((first_ci_start, first_ci_start))
            first_leaf_ci.sign = cc1.char
            second_leaf_ci = CommonInterval((first_ci_start + 1, first_ci_start + 1))
            second_leaf_ci.sign = cc2.char
            first_leaf = LeafNode(first_leaf_ci)
            second_leaf = LeafNode(second_leaf_ci)
            new_qnode = QNode((first_ci_start, first_ci_start + 1)).with_children((first_leaf, second_leaf))
            # if parent is a P node just accept children as leafs,
            # same behaviour in case the chars always show up in the same order
            # otherwise add an Q node as a child - and split the node as its leaf children

            # if parent is a P node - transform merged char to child QNode
            # in case the parent is Q - if the merged char is first or last in children order
            # try lift leafs to parent -
            # otherwise, the merged char in the middle of its parent - same as P case

            # if isinstance(parent, PNode) or len(mc.char_orders) == 1:
            #     parent.replace_child(node, first_leaf, second_leaf)
            # else:
            #     parent.replace_child(node,
            #                          QNode((first_ci_start, first_ci_start + 1)).with_children(
            #                              (first_leaf, second_leaf)))
            if isinstance(parent, PNode):
                parent.replace_child(node, new_qnode)
            # elif node in [parent.children[0], parent.children[-1]] and mc.other_side_same:
            #     parent.replace_child(node, first_leaf, second_leaf)
            # elif mc.context_same:
            #     parent.replace_child(node, first_leaf, second_leaf)
            # else:
            #     parent.replace_child(node, new_qnode)
            elif cls.possible_raise_merged_char(perms, node, first_leaf, second_leaf, parent):
                parent.replace_child(node, first_leaf, second_leaf)
            else:
                parent.replace_child(node, new_qnode)

        return tree_copy

    @classmethod
    def possible_raise_merged_char(cls, perms, node: LeafNode, first_leaf: LeafNode, second_leaf: LeafNode,
                                   parent: QNode) -> bool:
        """
        This function could hurt perf - might be wiser way to figure whether we can lift merged char to parent
        """

        parent_copy = deepcopy(parent)
        parent_copy.replace_child(node, first_leaf, second_leaf)
        frontier = set(parent_copy.frontier())
        len_of_frontier = len(next(iter(frontier)))

        for perm in perms:
            if all(''.join(map(str, w)) not in frontier for w in window(perm, len_of_frontier)):
                return False
        return True

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
        translated_p1 = tuple(TranslatedChar(val, org) for val, org in zip(norm_p1, p1))

        translations = defaultdict(list)
        for i, val in enumerate(p1):
            translations[val].append(norm_p1[i])

        iters = lmap(lambda p: cls.perm_variations(p, translations), perms[1:])
        for perm_set in IterProduct.iproduct([translated_p1], *iters):
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
                cur_trans = trans[c][index_in_trans]
                variation.append(TranslatedChar(cur_trans, c))
                indices_in_trans_order[c] += 1

            yield tuple(variation)

    @classmethod
    def iter_translation_permutations(cls, translations):
        """
        example:
        for {'a': [1], 'b': [2, 3], 'c': [4, 5]}

        yield:
        [
          {'a': (1,), 'b': (2, 3), 'c': (5, 4)},
          {'a': (1,), 'b': (2, 3), 'c': (4, 5)},
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

    @dataclass
    class MergeSettings:
        prefer_same_order_merge: bool
        no_loss_only: bool

    @classmethod
    def merge_multi_chars(cls, perms, merge_settings: MergeSettings = None):
        if not cls.has_duplications(perms[0]):
            yield perms, frozendict()
            print("No duplication in perms")
            return

        context_perms = cls.to_context_chars(perms)
        mergable_chars = cls.can_merge_multi_chars(context_perms)

        # test that we can merge all but maybe one appearance for each char
        maybe_no_loss_mergable_chars = [
            char for char, cc_per_perm in mergable_chars.items()
            if len(cc_per_perm[perms[0]]) >= num_appear(perms[0], char) - 1
        ]

        # translation = cls.try_merge_chars_no_loss(context_perms, mergable_chars, maybe_no_loss_mergable_chars)
        # if not translation:
        #     return None, None
        #
        # return cls.translate(context_perms, translation), translation
        for translation in cls.try_merge_chars_no_loss(context_perms, mergable_chars, maybe_no_loss_mergable_chars):
            if not translation:
                # yield None, None
                # print(2222)
                return

            yield cls.translate(context_perms, translation), translation

    @classmethod
    def translate(cls, cperms: CPermutations, translation: Translation):
        dict_trans = merge(*map(dict, translation))

        def translate_perm(cperm):
            it2 = window(chain(cperm, [None]))
            perm = []

            for w in it2:
                if w in dict_trans:
                    perm.append(dict_trans[w])
                    next(it2)
                else:
                    perm.append(w[0].char)
            return tuple(perm)

        return tmap(translate_perm, cperms)

    @classmethod
    def reduce_multi_chars1(cls, perms: Permutations):
        reducibale_chars = cls.find_reduciable_chars(perms[0], perms[1:])
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
    def reduce_multi_chars(cls, perms: Permutations):
        final_perms = []
        for perm in perms:
            final_perm = []
            for char, occur_num in iter_char_occurrence(perm):
                if occur_num > 1:
                    final_perm.append(MultiChar(char, occur_num))
                else:
                    final_perm.append(char)

            final_perms.append(tuple(final_perm))
        return tuple(final_perms)

    @classmethod
    def multi_chars_to_regular_chars(cls, perms: Permutations) -> Tuple[Permutations, MultiCharIndex]:
        """
        remove multi char, but return a dict that will map perm:index to MultiChars
        {
            0: {0: MultiCHar(...), ...},
            ...,
            k: {..., k: MultiChar(...)}
        }

        This is done in order to construct the pqtree - where MultiChars cab be equal to their original chars
        But after construction we can retrieve the information in the MultiChars
        """
        multichar_index = defaultdict(dict)
        norm_perms = []

        for perm_idx, perm in enumerate(perms):
            norm_perm = []
            for char_idx, char in enumerate(perm):
                if not isinstance(char, MultiChar):
                    norm_perm.append(char)
                else:
                    multichar_index[perm_idx][char_idx] = char
                    norm_perm.append(char.char)

            else:
                norm_perms.append(tuple(norm_perm))

        return tuple(norm_perms), frozendict(multichar_index)

    @classmethod
    def to_context_chars(cls, perms: Permutations):
        return tuple(
            tmap1(ContextChar.from_perm_index, p, irange(p))
            for p in perms
        )

    @classmethod
    def try_merge_chars_no_loss(cls, context_perms: Collection[ContextPerm],
                                mergable_chars: Dict[object, Dict[Collection, List[ContextChar]]],
                                chars_to_merge: Collection,
                                cur_translation: tuple = None) -> Optional[Translation]:
        if not chars_to_merge:
            yield cur_translation or ()

        translation = cur_translation or ()

        for char in chars_to_merge:
            char_count = next(iter(mergable_chars[char])).count(char)
            for char_trans in cls.try_merge_char(context_perms, mergable_chars[char], char_count - 1):
                # if char_count == len(char_trans):
                    # we could compute all and remove merged char whose occurrences don't agree on the order
                    # reduced_trans = char_trans[:-1]
                # else:
                #     reduced_trans = char_trans

                yield from cls.try_merge_chars_no_loss(context_perms,
                                                       mergable_chars,
                                                       set(chars_to_merge) - {char},
                                                       (*translation, *char_trans))

    @classmethod
    def iter_pairing_space(cls, seqs_to_pair):
        return IterProduct.iproduct(*[permutations(cchars) for cchars in seqs_to_pair])

    @classmethod
    def try_merge_cchar(cls, already_used: set, translation: list, cchars: list, neighbour, context_perms):
        cchars_with_neighbours_tuples = assoc_cchars_with_neighbours(cchars, neighbour, context_perms)

        # todo: should try merge biggest context possible - this it will QNode for sure

        # if neighbour == cchars[0].char:
        # same characters should be merged with multi char to avoid tree reconstruction confusion
        # return None, None

        if any(cc in already_used for cc in chain(cchars_with_neighbours_tuples)):
            return None, None

        # best_replacement = cls.most_agreed_replacement(cchars_with_neighbours_tuples)

        # neighbours = lremove(cchars.__contains__, chain(*cchars_with_neighbours_tuples))
        # other_side_of_neighbours = {
        #     c.left_char if c.left_char != cchars[0].char else c.right_char
        #     for c in neighbours
        # }
        #
        # other_side_of_chars = {
        #     c.left_char if c.left_char != neighbour else c.right_char
        #     for c in cchars
        # }
        #
        # other_side_of_neighbour_always_same = len(other_side_of_neighbours) == 1
        # context_always_same = len(other_side_of_chars) == 1 and other_side_of_neighbour_always_same

        merged_char = MergedChar.from_occurrences(*char_neighbour_tuples(cchars, neighbour))

        updated_translation = (
            *translation,
            frozendict({t: merged_char for t in cchars_with_neighbours_tuples})
        )

        updated_used_set = already_used | set(chain(cchars_with_neighbours_tuples))

        return updated_used_set, updated_translation

    @classmethod
    def try_merge_char(cls,
                       context_perms: Collection[ContextPerm],
                       cur_mergable_chars: Dict[Collection, List[ContextChar]],
                       num_translations_to_do: int) -> Iterable[Translation]:

        def try_translate_pairing(char_pairing, cur_translation=None, already_used=None):
            cur_translation = cur_translation or ()
            already_used = already_used or set()

            l_pairing = list(char_pairing)
            if len(cur_translation) == num_translations_to_do:
                return cur_translation

            while cur_cchars := l_pairing.pop():
                for neighbour in iter_common_neighbours(*cur_cchars):

                    new_used, updated_translation = cls.try_merge_cchar(already_used, cur_translation, cur_cchars,
                                                                        neighbour, context_perms)

                    if new_used is None:
                        continue

                    if trans := try_translate_pairing(l_pairing, updated_translation, new_used):
                        return trans

                return None

        # l = list(cls.iter_pairing_space(cur_mergable_chars.values()))
        # pprint(l)
        for order_pairing in cls.iter_pairing_space(cur_mergable_chars.values()):
            char_wise_pairing = list(zip(*order_pairing))
            # print(char_wise_pairing)
            if trans := try_translate_pairing(char_wise_pairing):
                yield trans
        # return None

    @classmethod
    def update_multi_leafs(cls, norm_tree: PQTree, multi_char_indices: MultiCharIndex):
        def multi_char_from(_perm_index, _index_in_perm) -> Optional[MultiChar]:
            return get_in(multi_char_indices, [_perm_index, _index_in_perm])

        multi_tree = deepcopy(norm_tree)
        for leaf in multi_tree.iter_leafs():
            leaf.multi_occurrences = {1: 0}
            indeces_in_perms = map(itemgetter(0), leaf.ci.intervals)
            for perm_index, index_in_perm in enumerate(indeces_in_perms):
                if multi_char := multi_char_from(perm_index, index_in_perm):
                    leaf.multi_occurrences[multi_char.count] = leaf.multi_occurrences.get(multi_char.count, 0) + 1
                else:
                    leaf.multi_occurrences[1] += 1

            leaf.multi_occurrences = frozendict(leaf.multi_occurrences)

        return multi_tree


if __name__ == '__main__':
    # PQTreeVisualizer.show(PQTreeBuilder.from_perms(((0, 1, 2, 3, 4), (0, 4, 3, 1, 2))))
    # PQTreeVisualizer.show(PQTreeBuilder.from_perms(((0, 4, 2, 3, 1), (0, 1, 3, 4, 2))))
    # PQTreeVisualizer.show(PQTreeBuilder.from_perms((('a', 'e', 'c', 'd', 'b'), ('a', 'b', 'd', 'e', 'c'))))
    dup_perms = [
        (0, 1, 2, 3, 4, 5, 1, 7, 5, 9),
        (9, 5, 7, 1, 3, 1, 5, 4, 2, 0)
    ]
    l = list(PQTreeDup.from_perms(dup_perms))
    pqtree = l[0]

    print(pqtree.to_parens())
    print(pqtree.to_json(pretty=True))
    PQTreeVisualizer.show(pqtree)
