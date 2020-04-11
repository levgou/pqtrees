from random import randrange, random, shuffle


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

