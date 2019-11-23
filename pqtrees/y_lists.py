from dataclasses import dataclass
from typing import Optional


@dataclass
class YNode:
    u_node: 'UNode'
    l_node: 'LNode'
    next: Optional['YNode'] = None
    prev: Optional['YNode'] = None


@dataclass
class UNode:
    max_y: YNode
    next: Optional['UNode'] = None
    prev: Optional['UNode'] = None


@dataclass
class LNode:
    max_y: YNode
    next: Optional['LNode'] = None
    prev: Optional['LNode'] = None


class YLists:
    ulist: UNode
    llist: LNode
    ylist: YNode
