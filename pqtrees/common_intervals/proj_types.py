from typing import Callable, Sequence, Tuple

Index = int
Val = int
Interval = Tuple[Index, Index]

SigmaFunc = Callable[[Index], Val]
SigmaInvFunc = Callable[[Val], Index]
PiFunc = Callable[[Index], Index]

F_X_Y = Callable[[Index, Index], Val]
U_L_Func = Callable[[Index, Index], Index]

IntSeq = Sequence[int]
