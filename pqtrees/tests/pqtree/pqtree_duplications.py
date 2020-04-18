from collections import namedtuple, defaultdict
import time

from pprint import pprint
from random import shuffle

from frozendict import frozendict
from funcy import lmap, lfilter

from pqtrees.common_intervals.generalized_letters import MultipleOccurrenceChar as MultiChar, ContextChar, MergedChar
from pqtrees.common_intervals.perm_helpers import tmap
from pqtrees.common_intervals.pqtree import LeafNode, PQTreeVisualizer
from pqtrees.common_intervals.pqtree_duplications import PQTreeDup
from pqtrees.common_intervals.string_mutations import duplication_mutations, mutate_collection


def test_reduce_perms():
    tests = [
        [
            [(1, 2, 3), (3, 2, 1)],
            ((1, 2, 3), (3, 2, 1)),
            ((1, 2, 3), (3, 2, 1)),
            {}
        ],

        [
            [(1, 2, 3, 2), (3, 2, 1)],
            ((1, 2, 3, 2), (3, 2, 1)),
            ((1, 2, 3, 2), (3, 2, 1)),
            {}
        ],

        [
            [(1, 2, 2, 3), (3, 2, 2, 1)],
            ((1, MultiChar(2, 2), 3), (3, MultiChar(2, 2), 1)),
            ((1, 2, 3), (3, 2, 1)),
            {
                0: {1: MultiChar(2, 2)}, 1: {1: MultiChar(2, 2)}
            }
        ],

        [
            [(1, 2, 2, 2, 3), (3, 2, 2, 2, 1)],
            ((1, MultiChar(2, 3), 3), (3, MultiChar(2, 3), 1)),
            ((1, 2, 3), (3, 2, 1)),
            {
                0: {1: MultiChar(2, 3)}, 1: {1: MultiChar(2, 3)}
            }
        ],

        [
            [(1, 1, 2, 2, 3), (3, 2, 2, 1, 1)],
            ((MultiChar(1, 2), MultiChar(2, 2), 3), (3, MultiChar(2, 2), MultiChar(1, 2))),
            ((1, 2, 3), (3, 2, 1)),

            {
                0: {0: MultiChar(1, 2), 1: MultiChar(2, 2)}, 1: {2: MultiChar(1, 2), 1: MultiChar(2, 2)}
            }
        ],

        [
            [(1, 1, 2, 2, 3, 3), (3, 3, 2, 2, 1, 1)],
            ((MultiChar(1, 2), MultiChar(2, 2), MultiChar(3, 2)),
             (MultiChar(3, 2), MultiChar(2, 2), MultiChar(1, 2))),
            ((1, 2, 3), (3, 2, 1)),

            {
                0: {0: MultiChar(1, 2), 1: MultiChar(2, 2), 2: MultiChar(3, 2)},
                1: {2: MultiChar(1, 2), 1: MultiChar(2, 2), 0: MultiChar(3, 2)}
            }
        ],

        [
            [(0, 0, 1, 2, 2, 3, 4, 5, 5), (5, 5, 4, 3, 2, 2, 1, 0, 0)],

            ((MultiChar(0, 2), 1, MultiChar(2, 2), 3, 4, MultiChar(5, 2)),
             (MultiChar(5, 2), 4, 3, MultiChar(2, 2), 1, MultiChar(0, 2))),

            ((0, 1, 2, 3, 4, 5), (5, 4, 3, 2, 1, 0)),
            {
                0: {0: MultiChar(0, 2), 2: MultiChar(2, 2), 5: MultiChar(5, 2)},
                1: {0: MultiChar(5, 2), 3: MultiChar(2, 2), 5: MultiChar(0, 2)}
            }
        ],

        [
            [(1, 2, 2, 3), (3, 2, 2, 1), (1, 3, 2, 2), (2, 2, 1, 3)],
            ((1, MultiChar(2, 2), 3), (3, MultiChar(2, 2), 1), (1, 3, MultiChar(2, 2)), (MultiChar(2, 2), 1, 3)),
            ((1, 2, 3), (3, 2, 1), (1, 3, 2), (2, 1, 3)),

            {
                0: {1: MultiChar(2, 2)}, 1: {1: MultiChar(2, 2)},
                2: {2: MultiChar(2, 2)}, 3: {0: MultiChar(2, 2)},
            }
        ],
    ]

    for t_input, expected, normalized_multi_char_perms, mc_index in tests:
        res = PQTreeDup.reduce_multi_chars(t_input)
        norm, multi_char_index = PQTreeDup.multi_chars_to_regular_chars(res)
        assert res == expected
        assert norm == normalized_multi_char_perms
        assert mc_index == multi_char_index


def test_pqtree_after_reduce_chars():
    tests = [
        [
            [(1, 2, 3), (3, 2, 1)],
            "[1 2 3]",

            [{'char': '1', 'multi': False, 'multi_stats': {1: '1:1'}, 'type': 'LEAF'},
             {'char': '2', 'multi': False, 'multi_stats': {1: '1:1'}, 'type': 'LEAF'},
             {'char': '3', 'multi': False, 'multi_stats': {1: '1:1'}, 'type': 'LEAF'}],

            {'123', '321'}
        ],

        [
            [(1, 2, 3, 2), (3, 2, 1)],
            None,
            None,
            None
        ],

        [
            [(1, 2, 2, 3, 1), (3, 2, 1)],
            None,
            None,
            None
        ],

        [
            [(1, 2, 2, 3), (3, 2, 1)],
            "[1 2 3]",

            [{'char': '1', 'multi': False, 'multi_stats': {1: '2:2'}, 'type': 'LEAF'},
             {'char': '2',
              'multi': True,
              'multi_stats': {1: '1:2', 2: '1:2'},
              'type': 'LEAF'},
             {'char': '3', 'multi': False, 'multi_stats': {1: '2:2'}, 'type': 'LEAF'}],

            {'1223', '123', '3221', '321'}
        ],

        [
            [(1, 2, 2, 3), (3, 3, 2, 1)],
            "[1 2 3]",

            [{'char': '1', 'multi': False, 'multi_stats': {1: '2:2'}, 'type': 'LEAF'},
             {'char': '2',
              'multi': True,
              'multi_stats': {1: '1:2', 2: '1:2'},
              'type': 'LEAF'},
             {'char': '3',
              'multi': True,
              'multi_stats': {1: '1:2', 2: '1:2'},
              'type': 'LEAF'}],

            {'123', '1223', '3321', '3221', '1233', '12233', '321', '33221'}
        ],

        [
            [(1, 1, 1, 1, 2, 2, 2, 2, 3), (3, 3, 2, 1)],
            "[1 2 3]",

            [{'char': '1',
              'multi': True,
              'multi_stats': {1: '1:2', 4: '1:2'},
              'type': 'LEAF'},
             {'char': '2',
              'multi': True,
              'multi_stats': {1: '1:2', 4: '1:2'},
              'type': 'LEAF'},
             {'char': '3',
              'multi': True,
              'multi_stats': {1: '1:2', 2: '1:2'},
              'type': 'LEAF'}],

            {'111122223', '1111222233', '111123', '1111233', '122223',
             '1222233', '123', '1233', '321', '321111', '322221',
             '322221111', '3321', '3321111', '3322221', '3322221111'}
        ],
    ]

    for perms, pq_parens, leaf_dicts, frontier in tests:
        pqtree = PQTreeDup.from_perms_wth_multi(perms)
        assert (not pq_parens and not pqtree) or pq_parens == pqtree.to_parens()
        if not pqtree:
            continue

        assert lmap(LeafNode.dict_repr, pqtree.iter_leafs()) == leaf_dicts
        assert pqtree.frontier() == frontier


def test_pqtree_after_reduce_chars_rand_examples():
    ITERATIONS = 1000

    merged = 0
    rts = []

    for i in range(ITERATIONS):
        iter_stats = {}

        # ------------------------- Generation of permutations -------------------------
        id_perm = list(range(1, 10))
        duplication_mutations(id_perm, 2)

        other_perms = [list(id_perm), list(id_perm)]
        for p in other_perms:
            mutate_collection(p, 2)

        ps = tmap(tuple, (id_perm, *other_perms))

        # ------------------------- Try to merge same adjacent chars -------------------------
        start_time = time.time()
        pq = PQTreeDup.from_perms_wth_multi(ps)
        iter_stats["merged"] = time.time() - start_time

        if not pq:
            continue
        else:
            merged += 1

        # ------------------------- find all the trees with minimal size -------------------------
        start_time = time.time()
        all_possibilities = list(PQTreeDup.from_perms(ps))
        iter_stats["no_merge"] = time.time() - start_time
        iter_stats["perms"] = ps

        best_size = all_possibilities[0].approx_frontier_size()
        only_best_sized = lfilter(lambda t: t.approx_frontier_size() == best_size, all_possibilities)

        # verify tree with multi chars contains in its frontier one of the best trees
        try:
            front = set(pq.frontier())
            assert any(front.issuperset(t.frontier()) for t in only_best_sized)
        except:
            print(ps)
            # PQTreeVisualizer.show_all(pq, *only_best_sized)
            raise
        else:
            rts.append(iter_stats)

    print(f"multi merged: {merged} / {ITERATIONS}")
    lmap(print, rts)


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
        res = PQTreeDup.to_context_chars(t_in)
        assert res == expected, expected


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

    def traverse_norm_perm_space_indexes(perms):
        """de-translate TranslatedChar - into representing indices"""
        return {
            tuple(tuple(char.val for char in t) for t in perm)
            for perm in PQTreeDup.traverse_perm_space(perms)
        }

    ps1 = traverse_norm_perm_space_indexes(perms1)
    ps2 = traverse_norm_perm_space_indexes(perms2)
    ps3 = traverse_norm_perm_space_indexes(perms3)

    assert len(ps1) == 2
    assert len(ps2) == 4
    assert len(ps3) == (2 * 2 * 2) ** 3
    assert ps2 == {
        ((0, 1, 2, 3, 4), (0, 1, 3, 4, 2)),
        ((0, 1, 2, 3, 4), (0, 4, 3, 1, 2)),
        ((0, 1, 2, 3, 4), (3, 1, 0, 4, 2)),
        ((0, 1, 2, 3, 4), (3, 4, 0, 1, 2)),
    }

    assert ps1 == {
        ((0, 1, 2), (1, 2, 0)),
        ((0, 1, 2), (0, 2, 1))
    }

    size = [s for s in PQTreeDup.from_perms(perms2)]
    print(size)


def test_merge_chars():
    MC = MergedChar.from_occurrences
    Test = namedtuple('Test', ['perms', 'translations', "should_translate"])

    mc12 = MC((1, 2), (1, 2))
    mc121 = MC((1, 2), (2, 1))
    mc31 = MC((1, 3), (3, 1))

    tests = map(lambda d: Test(**d), [
        # ---------------------------- Test 1 ----------------------------
        {
            "perms": (
                (1, 2, 3, 1),
                (1, 2, 1, 3)
            ),

            "translations": [
                (
                    (mc12, 3, 1),
                    (mc12, 1, 3)
                ),
                (

                    (1, 2, mc31),
                    (1, 2, mc31)
                ),
                (
                    (mc121, 3, 1),
                    (1, mc121, 3)
                ),
            ],
            "should_translate": True
        },
        # ---------------------------- Test 2 ----------------------------
        {
            "perms": (
                (1, 2, 3, 4, 1, 5),
                (2, 1, 1, 3, 5, 4)
            ),

            "translations": [
                (
                    (mc121, 3, 4, 1, 5),
                    (mc121, 1, 3, 5, 4)
                )
            ],
            "should_translate": True
        },
        # ---------------------------- Test 3 ----------------------------
        {
            "perms": (
                (1, 2),
                (2, 1)
            ),

            "translations": [
                (
                    (1, 2),
                    (2, 1)
                )
            ],
            "should_translate": False
        },
        {
            "perms": (
                (1, 2, 3, 4, 5, 1),
                (1, 3, 2, 5, 4, 1)
            ),
            "translations": (),
            "should_translate": False

        }
    ])

    for perms, expects, should_translate in tests:
        res, translation = PQTreeDup.merge_multi_chars(perms)

        assert (res or not expects) and (translation or not should_translate)
        assert any(res == expect for expect in expects) or not expects


def test_pqtree_with_merges():
    perms = [
        (
            (1, 2, 3, 1),
            (1, 2, 1, 3)
        ),

        (
            (1, 2, 3, 5, 1),
            (1, 2, 1, 3, 5)
        ),

        (
            (1, 2, 3, 5, 1),
            (1, 2, 3, 1, 5),
            (1, 2, 5, 3, 1)
        ),

        # had some bug with 0 node
        # (
        #     (1, 2, 3, 0, 1),
        #     (1, 2, 1, 3, 0)
        # ),

        # random generated failed test
        (
            (1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10),
            (1, 2, 6, 4, 4, 3, 5, 7, 8, 9, 10),
            (1, 2, 3, 4, 4, 8, 7, 10, 9, 5, 6)
        )
    ]

    for ps in perms:
        pq = PQTreeDup.from_perms_with_merge(ps)
        all_possibilities = list(PQTreeDup.from_perms(ps))
        best_size = min(t.approx_frontier_size() for t in all_possibilities)
        only_best_sized = lfilter(lambda t: t.approx_frontier_size() == best_size, all_possibilities)

        try:
            assert pq.approx_frontier_size() == best_size
            assert len(list(pq.frontier())) == best_size
            assert any(pq.to_parens() == t.to_parens() for t in only_best_sized)

        except:
            print(f"best no opt: {best_size}, best with merge: {pq.approx_frontier_size()}")
            print(f"Merged: {pq.to_parens()}")
            print("any same front: ", [t.approx_frontier_size() for t in only_best_sized])
            print(ps)
            print([t.to_parens() for t in only_best_sized])
            # PQTreeVisualizer.show_all(pq, *only_best_sized)
            raise


def test_pqtree_with_merges_rand():
    ITERATIONS = 10

    merged = 0

    for i in range(ITERATIONS):
        id_perm = list(range(1, 10))
        duplication_mutations(id_perm, 1)

        other_perms = [list(id_perm), list(id_perm)]
        for p in other_perms:
            mutate_collection(p, 2)

        ps = tmap(tuple, (id_perm, *other_perms))

        pq = PQTreeDup.from_perms_with_merge(ps)

        if not pq:
            continue
        else:
            merged += 1

        all_possibilities = list(PQTreeDup.from_perms(ps))
        best_size = all_possibilities[0].approx_frontier_size()
        only_best_sized = lfilter(lambda t: t.approx_frontier_size() == best_size, all_possibilities)

        try:
            assert pq.approx_frontier_size() == best_size
            assert any(pq.to_parens() == t.to_parens() for t in only_best_sized)
        except:
            print(ps)
            # PQTreeVisualizer.show_all(pq, *only_best_sized)
            raise

    print(f"merged {merged}")


if __name__ == '__main__':
    # test_perm_space()
    # test_reduce_perms()
    # test_pqtree_after_reduce_chars()
    # test_context_char_conversion()
    # test_merge_chars()
    test_pqtree_with_merges()
    # test_pqtree_after_reduce_chars_rand_examples()
    # test_pqtree_with_merges_rand()
