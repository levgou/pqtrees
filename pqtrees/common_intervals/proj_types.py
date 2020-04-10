from typing import Callable, Sequence, Tuple, Hashable

Index = int
Val = int
Interval = Tuple[Index, Index]

SigmaFunc = Callable[[Index], Val]
SigmaInvFunc = Callable[[Val], Index]
PiFunc = Callable[[Index], Index]

F_X_Y = Callable[[Index, Index], Val]
U_L_Func = Callable[[Index, Index], Index]

IntSeq = Sequence[int]
Permutation = Sequence[Hashable]
Permutations = Sequence[Permutation]
