from typing import Callable, Sequence

Index = int
Val = int

SigmaFunc = Callable[[Index], Val]
SigmaInvFunc = Callable[[Val], Index]

F_X_Y = Callable[[Index, Index], Val]
U_L_Func = Callable[[Index, Index], Index]

IntSeq = Sequence[int]
