from collections import Counter
from random import shuffle, randrange, random

import matplotlib.pyplot as plt
from funcy import lmap, flatten

from pqtrees.common_intervals.pqtree_duplications import PQTreeDup

from pqtrees.common_intervals.pqtree import PQTreeBuilder, PQTree
import cProfile


def rand_perms(length: int, amount: int, repeating_letters: int, showup_count: int):
    repeating_letters_count = (showup_count - 1) * repeating_letters
    unique_letters_len = length - repeating_letters_count
    base_id_str = list(range(unique_letters_len))

    for c in range(repeating_letters):
        for _ in range(showup_count - 1):
            base_id_str.append(c)

    base_str = list(base_id_str)
    shuffle(base_str)
    perms = [list(base_str) for _ in range(amount)]

    for i in range(1, amount):
        shuffle(perms[i])

    return perms


def rand_index_with_len_bigger_than(col_len, bigger_than):
    while True:
        x = randrange(col_len)
        len_till_end = col_len - x + 1
        if len_till_end > bigger_than:
            return x, len_till_end


def rand_mutation_size(max_len):
    rand = random()
    mut_size = 2
    acc = 0.5

    while acc <= rand and mut_size < max_len:
        mut_size += 1
        acc += 1 / (2 ** mut_size)

    return mut_size


def mutate(col):
    REVERESE_PROBA = 2 / 3  # if bigger than reverse -> shuffle
    rand = random()

    if rand < REVERESE_PROBA:
        return list(reversed(col))
    else:
        cp = list(col)
        shuffle(cp)
        return cp


def put_sub_col_at(col, mutated_sub_col, mut_start):
    for i in range(len(mutated_sub_col)):
        index = i + mut_start
        col[index] = mutated_sub_col[i]


def mutate_collection(col, num_mutations):
    for _ in range(num_mutations):
        mut_start, len_till_end = rand_index_with_len_bigger_than(len(col), 2)
        mut_size = rand_mutation_size(len_till_end)
        sub_col = col[mut_start: mut_start + mut_size]
        mutated_sub_col = mutate(sub_col)
        put_sub_col_at(col, mutated_sub_col, mut_start)


def duplication_mutations(col: list, duplication_amount):
    for _ in range(duplication_amount):
        i = randrange(len(col))
        col.insert(i, col[i])


def is_list_consecutive(lst):
    sorted_list = list(sorted(lst))
    consecutive_list = list(range(sorted_list[0], sorted_list[-1] + 1))
    return consecutive_list == sorted_list


def can_reduce_chars(id_perm, others):
    count = Counter(id_perm)
    more_than_once = {k: v for k, v in count.items() if v > 1}

    compactable_chars = []
    for char in more_than_once:
        for perm in others:
            indices = [i for i, x in enumerate(perm) if x == char]
            if not is_list_consecutive(indices):
                break
        else:
            compactable_chars.append(char)

    return compactable_chars


def neighbour_set(perm, char):
    indices = [i for i, x in enumerate(perm) if x == char]
    neighbour_indeces = {
        x - 1 for x in indices if x > 0
    }.union({
        x + 1 for x in indices if x < len(perm) - 1
    })

    neighbours = frozenset(perm[idx] for idx in neighbour_indeces)
    return neighbours


def can_merge_double_chars(id_perm, others):
    count = Counter(id_perm)
    more_than_once = {k: v for k, v in count.items() if v > 1}

    mergable_chars = []
    for char in more_than_once:
        neighbours = neighbour_set(id_perm, char)
        for other in others:
            neighbours &= neighbour_set(other, char)

        if neighbours:
            mergable_chars.append(char)
        else:
            print(char, Counter(flatten([
                list(neighbour_set(other, char)) for other in others
            ])))

    return mergable_chars


def build_trees(perms):
    perms_cp = tuple(tuple(p) for p in perms)
    trees = [t for t in PQTreeDup.from_perms(perms_cp)]
    front_sizes = (sorted([t.approx_frontier_size() for t in trees]))
    return {
        "min": min(*front_sizes),
        "max": max(*front_sizes),
        "avg": sum(front_sizes) / len(front_sizes),
        "ratio": max(*front_sizes) / min(*front_sizes),
        "like_min": front_sizes.count(min(*front_sizes)),
        "count": len(front_sizes)
    }


def main():
    LEN = 10
    AMOUNT = 5
    DUP_NUM = 2
    NUM_MUT = 3

    ratios = []
    can_compact_nums = []
    can_merge_nums = []

    for _ in range(100):
        id_perm = list(range(LEN))
        duplication_mutations(id_perm, DUP_NUM)

        perm_copies = [list(id_perm) for _ in range(AMOUNT)]
        for i in range(1, AMOUNT):
            mutate_collection(perm_copies[i], NUM_MUT)

        # print(perm_copies)
        res = build_trees(perm_copies)
        ratios.append(res)
        can_compact_nums.append(len(can_reduce_chars(id_perm, perm_copies)))

        if not can_reduce_chars(id_perm, perm_copies):
            can_merge_nums.append(len(can_merge_double_chars(id_perm, perm_copies)))

    # plt.hist([res["ratio"] for res in ratios], bins=20)
    # plt.show()
    # plt.hist([res["like_min"] for res in ratios], bins=20)
    # plt.show()
    # plt.hist([(res["count"] - res["like_min"]) / res["like_min"] for res in ratios], bins=10)
    # plt.show()
    # plt.hist(can_compact_nums, bins=3)
    # plt.show()
    # plt.hist(can_merge_nums, bins=3)
    # plt.show()


if __name__ == '__main__':
    # cProfile.run("main()", sort="cumtime")
    main()
