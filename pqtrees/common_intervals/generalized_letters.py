from collections import Counter
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List


@dataclass(frozen=True)
class MultipleOccurrenceChar:
    char: object
    count: int

    def __str__(self):
        return str(self.char) * self.count

    def __repr__(self):
        return f"MC({self.char},{self.count})"


@dataclass(frozen=True)
class MergedChar:
    char_orders: Tuple[Tuple[object, object], ...]
    counts: Dict[object, int]

    def __str__(self):
        return f"{self.char_orders[0]}{self.char_orders[1]}"

    def __repr__(self):
        return f"Merged{self.char_orders}"

    @classmethod
    def from_occurrences(cls, *occurrences: Tuple[object, object]):
        counts = Counter(occurrences)
        return MergedChar(tuple(sorted(count[0] for count in counts.most_common())), counts)


@dataclass(frozen=True)
class ContextChar:
    left_char: Optional[object]
    char: object
    right_char: Optional[object]
    index: Optional[int] = None
    perm: Optional[object] = None

    @classmethod
    def from_perm_index(cls, perm, index):
        left_char = perm[index - 1] if index > 0 else None
        right_char = perm[index + 1] if index < len(perm) - 1 else None
        char = perm[index]
        return ContextChar(left_char, char, right_char, index, perm)

    def __repr__(self):
        return f"CC[{self.left_char}, <{self.char}>, {self.right_char}]"



ContextPerm = List[ContextChar]