from collections import namedtuple
from random import shuffle

from pqtrees.common_intervals.generalized_letters import MultipleOccurrenceChar as MultiChar, ContextChar, MergedChar
from pqtrees.common_intervals.pqtree_duplications import PQTreeDup


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

    ps1 = set(PQTreeDup.traverse_perm_space(perms1))
    ps2 = set(PQTreeDup.traverse_perm_space(perms2))
    ps3 = set(PQTreeDup.traverse_perm_space(perms3))

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
    ]

    for ps in perms:
        PQTreeDup.from_perms_with_merge(ps)

if __name__ == '__main__':
    test_perm_space()
    test_reduce_perms()
    test_context_char_conversion()
    test_merge_chars()
    test_pqtree_with_merges()
