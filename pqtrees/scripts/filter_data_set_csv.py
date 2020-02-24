import csv
from collections import defaultdict
from pprint import pprint

from funcy import project, lmap

filename = "/Users/levgour/skool/proj/dataset.csv"
words = defaultdict(list)

# bigger than 5 families
FAMILY1 = "8749"  # 12
FAMILY2 = "8939"  # 8
FAMILY3 = "8849"  # 8
FAMILY4 = "8766"  # 10

# same len after removing duplication, family size = 3, words longer than 7
FAMILY_5 = "2851"  # 9
# FAMILY_6 = "1473"  # 11
FAMILY_7 = "4545"  # 7
# FAMILY_8 = "4062"  # 7
FAMILY_9 = "790"  # 14

with open(filename) as csvfile:
    reader = csv.DictReader(csvfile)
    for i, row_d in enumerate(reader):
        row_csb_list = {**row_d, 'CSB': row_d['CSB'].split(',')}
        words[row_d['Family_ID']].append(row_csb_list)


def rem_duplicates(lst):
    s = set(lst)
    l_uniq = []
    for x in lst:
        if x in s:
            l_uniq.append(x)
            s.remove(x)
    return l_uniq


#
# for k, v in words.items():
#     if any(len(d['CSB']) > 7 for d in v) and len(v) > 2:
#         if len({len(set(d['CSB'])) for d in v}) == 1:
#             if len({tuple(sorted(rem_duplicates(w['CSB']))) for w in v}) == 1:
#                 print(k, len(v), len(rem_duplicates(v[1]['CSB'])))
#
bigger_families = project(words, [FAMILY1, FAMILY2, FAMILY3, FAMILY4])
longer_families = project(words, [FAMILY_5, FAMILY_7, FAMILY_9])
longer_families_no_dups = {
    name: lmap(lambda fam: {**fam, 'CSB': rem_duplicates(fam['CSB'])}, fams)
    for name, fams in longer_families.items()
}

# for fam_ws in longer_families_no_dups.values():
#     for w in fam_ws:
#         print(w['CSB'])
#     print("#" * 50)

fams_ingestable_oren_exe = {}
for name, fam in longer_families_no_dups.items():
    ws = [x['CSB'] for x in fam]
    fams_ingestable_oren_exe[name] = ws

for name, ws in fams_ingestable_oren_exe.items():
    index = dict(zip(ws[0], range(1, len(ws[0]) + 1)))

    print("=" * 50, name, "=" * 50)
    for i, w in enumerate(ws, 1):
        # print(f"{i})",  ",".join([str(index[c]) for c in w]))
        print(f"{i})", w)

    print("=" * 120, '\n')
