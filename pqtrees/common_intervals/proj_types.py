from typing import Callable

Index = int
Val = int

SigmaFunc = Callable[[Index], Val]
SigmaInvFunc = Callable[[Val], Index]
