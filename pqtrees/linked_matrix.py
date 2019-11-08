from dataclasses import dataclass
from typing import Optional


@dataclass
class Node:
    h_next: 'Node'
    h_prev: 'Node'
    v_next: 'Node'
    v_prev: 'Node'


class LinkedMatrix:

    h_head: Optional[Node]
    h_tail: Optional[Node]
    height: int

    def __init__(self) -> None:
        super().__init__()

        self.h_head = self.h_tail = None
        self.height = 0
