import random

p1 = range(10)
perms1 = [list(p1) for _ in range(5)]
for l in perms1[1:]:
    random.shuffle(l)

p2 = range(15)
perms2 = [list(p2) for _ in range(5)]
for l in perms2[1:]:
    random.shuffle(l)

p3 = range(20)
perms3 = [list(p3) for _ in range(7)]
for l in perms3[1:]:
    random.shuffle(l)

print(perms1)
print(perms2)
print(perms3)

for i, ps in enumerate([perms1, perms2, perms3], 1):
    print("\n", "#" * 10, f"perms{i}")

    for i, p in enumerate(ps, 1):
        print(f"{i})", ",".join(map(lambda x: str(x+1), p)))

    # print("$" * 50)

    # print("[")
    # for p in ps:
    #     print(f"{tuple(p)},")
    # print("]")
