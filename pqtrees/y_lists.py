from dataclasses import dataclass
from functools import partial
from typing import Optional, Union, Iterable
from copy import deepcopy

from funcy import compose

from pqtrees.common_intervals.proj_types import SigmaFunc, SigmaInvFunc, PiFunc, Index


@dataclass
class YNode:
    val: int
    u_node: Optional['UNode'] = None
    l_node: Optional['LNode'] = None
    next: Optional['YNode'] = None
    prev: Optional['YNode'] = None


@dataclass
class UNode:
    val: int
    y_range: range
    max_y: YNode
    next: Optional['UNode'] = None
    prev: Optional['UNode'] = None


@dataclass
class LNode:
    val: int
    y_range: range
    max_y: YNode
    next: Optional['LNode'] = None
    prev: Optional['LNode'] = None


YUL = Union[YNode, UNode, LNode]


class YLists:
    ulist: UNode
    llist: LNode
    ylist: YNode
    pi_ab: PiFunc


    def __init__(self, sig_a: SigmaFunc, sig_b_inv: SigmaInvFunc, n: int) -> None:
        super().__init__()

        self.pi_ab = compose(sig_b_inv, sig_a)
        pi_ab_last = self.pi_ab(n - 1)

        self.ylist = YNode(val=n - 1)
        self.ulist = UNode(pi_ab_last, range(pi_ab_last, pi_ab_last + 1), self.ylist)
        self.llist = LNode(pi_ab_last, range(pi_ab_last, pi_ab_last + 1), self.ylist)

        self.ylist.l_node = self.llist
        self.ylist.u_node = self.ulist


    def update_for_x(self, x: int) -> 'YLists':

        # in order to save memory after testing should reuse current ylists
        new_lists = deepcopy(self)
        self._add_y_node(new_lists, x)
        self._add_l_node(new_lists, x)
        self._rem_u_nodes(new_lists, x)

        return new_lists


    @classmethod
    def _find_y_star_node(cls, ylists: 'YLists', pi_x: Index) -> UNode:

        def u_x_greater_than_u_x_1(u_node: UNode): return pi_x > u_node.val

        y_star_node = ylists.ulist
        for u_node in cls._iter_list(ylists.ulist):
            if u_x_greater_than_u_x_1(u_node):
                y_star_node = u_node
            else:
                break

        return y_star_node


    @classmethod
    def _rem_u_y_nodes(cls, ylists: 'YLists', y_star_node: UNode) -> None:

        first_y_node_rm = y_star_node.prev.max_y
        first_y_node_keep = first_y_node_rm.next

        first_y_node_keep.prev = ylists.ylist
        ylists.ylist.next = first_y_node_keep

        ylists.ylist = y_star_node
        y_star_node.prev = None


    @classmethod
    def _rem_u_nodes(cls, ylists: 'YLists', x: Index) -> None:
        pi_x = ylists.pi_ab(x)
        pi_x_1 = ylists.pi_ab(x + 1)

        if pi_x <= pi_x_1:
            ylists.ulist.y_range = range(x, ylists.ulist.y_range[-1])
            ylists.ylist.u_node = ylists.ulist
            return

        y_star_node = cls._find_y_star_node(ylists, pi_x)

        if y_star_node.prev:
            cls._rem_u_y_nodes(ylists, y_star_node)

        y_star_node.y_range = range(x, y_star_node.y_range[-1])
        y_star_node.val = pi_x


    @staticmethod
    def _add_l_node(ylists: 'YLists', x: int) -> None:
        pi_x = ylists.pi_ab(x)
        pi_x_1 = ylists.pi_ab(x + 1)

        if pi_x > pi_x_1:
            ylists.llist = LNode(pi_x, range(pi_x, pi_x + 1), max_y=ylists.ylist, next=ylists.llist)
            ylists.llist.next.prev = ylists.llist

        else:
            ylists.llist.y_range = range(x, ylists.llist.y_range[-1])
            ylists.llist.val = pi_x

        ylists.ylist.l_node = ylists.llist


    @staticmethod
    def _add_y_node(ylists: 'YLists', x: int) -> None:
        ylists.ylist = YNode(x, next=ylists.ylist)
        ylists.ylist.next.prev = ylists.ylist


    def _construct_ylist(self, n: int) -> YNode:
        assert n
        cur_node = None
        n -= 1

        while n >= 0:
            new_node = YNode(n, next=cur_node)
            cur_node = new_node

            if cur_node.next:
                cur_node.next.prev = cur_node

            n -= 1

        return cur_node


    def _construct_ulist(self, ylist: YNode) -> UNode:

        for y_node in self._iter_list(ylist):
            pass


    @staticmethod
    def _iter_list(l: YUL) -> Iterable[YUL]:
        cur = l
        while cur:
            yield cur
            cur = cur.next
