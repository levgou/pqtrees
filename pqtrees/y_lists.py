import matplotlib.pyplot as plt
import networkx as nx

from dataclasses import dataclass
from itertools import chain
from typing import Optional, Union, Iterable
from copy import deepcopy
from unittest import TestCase
from random import shuffle
from networkx import DiGraph

from funcy import compose, complement

from pqtrees.common_intervals.proj_types import SigmaFunc, SigmaInvFunc, PiFunc, Index


def l_u_str(l_u_node) -> str:
    return f'[{l_u_node.val}; [{l_u_node.y_range[0]}, {l_u_node.y_range[-1]}]]'


def l_len(l_node) -> int:
    len_ = 0
    while l_node is not None:
        len_ += 1
        l_node = l_node.next

    return len_


@dataclass
class YNode:
    val: int
    u_node: Optional['UNode'] = None
    l_node: Optional['LNode'] = None
    next: Optional['YNode'] = None
    prev: Optional['YNode'] = None


    def __str__(self) -> str:
        return f'Y[{self.val}]'


    def __len__(self) -> int: return l_len(self)


@dataclass
class UNode:
    val: int
    y_range: range
    max_y: YNode
    next: Optional['UNode'] = None
    prev: Optional['UNode'] = None


    def __str__(self) -> str:
        return "U" + l_u_str(self)


    def __len__(self) -> int: return l_len(self)


@dataclass
class LNode:
    val: int
    y_range: range
    max_y: YNode
    next: Optional['LNode'] = None
    prev: Optional['LNode'] = None


    def __str__(self) -> str:
        return "L" + l_u_str(self)


    def __len__(self) -> int: return l_len(self)


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
        self.ulist = UNode(pi_ab_last, range(n - 1, n), self.ylist)
        self.llist = LNode(pi_ab_last, range(n - 1, n), self.ylist)

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

        def u_x_greater_than_u_x_1(u_node: UNode):
            return pi_x > u_node.val


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
            ylists.ulist.y_range = range(pi_x, ylists.ulist.y_range.stop)
            ylists.ylist.u_node = ylists.ulist
            return

        y_star_node = cls._find_y_star_node(ylists, pi_x)

        if y_star_node.prev:
            cls._rem_u_y_nodes(ylists, y_star_node)

        y_star_node.y_range = range(x, y_star_node.y_range.stop)
        y_star_node.val = pi_x
        ylists.ylist.u_node = y_star_node


    @staticmethod
    def _add_l_node(ylists: 'YLists', x: int) -> None:
        pi_x = ylists.pi_ab(x)
        pi_x_1 = ylists.pi_ab(x + 1)

        if pi_x > pi_x_1:
            ylists.llist = LNode(pi_x, range(x, x + 1), max_y=ylists.ylist, next=ylists.llist)
            ylists.llist.next.prev = ylists.llist

        else:
            ylists.llist.y_range = range(x, ylists.llist.y_range.stop)
            ylists.llist.val = pi_x

        ylists.ylist.l_node = ylists.llist


    @staticmethod
    def _add_y_node(ylists: 'YLists', x: int) -> None:
        ylists.ylist = YNode(x, next=ylists.ylist)
        ylists.ylist.next.prev = ylists.ylist


    @staticmethod
    def _iter_list(l: YUL) -> Iterable[YUL]:
        cur = l
        while cur:
            yield cur
            cur = cur.next


    def pretty_print(self) -> None:
        def l_to_str(l): return '-> '.join(map(str, self._iter_list(l)))


        u_list_str = l_to_str(self.ulist)
        y_list_str = l_to_str(self.ylist)
        l_list_str = l_to_str(self.llist)

        print(f"u: {u_list_str}")
        print(f"y: {y_list_str}")
        print(f"l: {l_list_str}")


    def gen_graph(self) -> DiGraph:
        g = DiGraph()
        node_lists = (self.ulist, self.ylist, self.llist)

        for node in chain(*[self._iter_list(l) for l in node_lists]):
            if node.next:
                g.add_edge(str(node), str(node.next))
            if node.prev:
                g.add_edge(str(node), str(node.prev))
            if getattr(node, 'max_y', None):
                g.add_edge(str(node), str(node.max_y))
            if getattr(node, 'u_node', None):
                g.add_edge(str(node), str(node.u_node))
            if getattr(node, 'l_node', None):
                g.add_edge(str(node), str(node.l_node))

        for l in node_lists:
            g.add_edge('ylists', str(l))

        return g


    def _gen_node_positions(self, g: DiGraph) -> dict:
        STEP_SIZE = 50
        Y_POS = 100
        L_POS = 200
        U_POS = 0

        positions = {'ylists': (0, Y_POS)}

        y_count = 50
        l_count = 50
        u_count = 50

        def sort_key(node):
            if node[0] == 'Y':
                return int(node.split('[')[-1].split(']')[0])
            elif node[0] == 'U':
                return int(node.split('[')[2].split(',')[0]) * 100
            elif node[0] == 'L':
                return int(node.split('[')[2].split(',')[0]) * 1000
            else:
                return 0

        for node in filter(complement(r'ylists'), sorted(g.nodes, key=sort_key)):
            if node[0] == 'Y':
                positions[node] = (y_count, Y_POS)
                y_count += STEP_SIZE
            if node[0] == 'L':
                positions[node] = (l_count, L_POS)
                l_count += STEP_SIZE
            if node[0] == 'U':
                positions[node] = (u_count, U_POS)
                u_count += STEP_SIZE

        return  positions

    def display_plot(self) -> None:
        NODE_SIZE = 5000
        FIG_EDGE_SIZE = 8

        g = self.gen_graph()

        def color_chooser(node_name):
            return {
                'y': 0,
                'U': 0.25,
                'L': 0.5,
                'Y': 0.75,
            }[node_name[0]]


        colors = list(map(color_chooser, g.nodes))
        positions = self._gen_node_positions(g)

        plt.figure(1, figsize=(FIG_EDGE_SIZE, FIG_EDGE_SIZE))
        nx.draw_networkx(g, pos=positions, node_size=NODE_SIZE,
                         node_shape='8', node_color=colors)
        plt.show()


class YListsTests(TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.a = [1, 2, 3, 4, 5, 6, 7]
        self.b = [7, 3, 1, 4, 5, 2, 6]
        self.b_index = dict(zip(self.b, range(len(self.b))))
        self.ylists = YLists(sig_a=self.a.__getitem__,
                             sig_b_inv=self.b_index.__getitem__,
                             n=len(self.a))


    def test_01_initialization(self):
        for n in range(1, 101):
            a = list(range(n))

            b = list(a)
            shuffle(b)

            b_index = dict(zip(b, range(len(b))))
            ylists = YLists(sig_a=a.__getitem__, sig_b_inv=b_index.__getitem__, n=len(a))

            u_l = b.index(a[-1])

            self.assertEqual(n - 1, ylists.ylist.val)
            self.assertEqual(u_l, ylists.ulist.val)
            self.assertEqual(range(n - 1, n), ylists.ulist.y_range)
            self.assertEqual(u_l, ylists.llist.val)
            self.assertEqual(range(n - 1, n), ylists.llist.y_range)


    def test_02_update_x_pi_x_smaller_pi_x_1(self):
        after_one = self.ylists.update_for_x(5)

        self.assertEqual(5, after_one.ylist.val)
        self.assertEqual(6, after_one.ylist.next.val)
        self.assertEqual(2, len(after_one.ylist))

        self.assertEqual(6, after_one.llist.val)
        self.assertEqual(0, after_one.llist.next.val)
        self.assertEqual(2, len(after_one.llist))
        self.assertEqual(range(5, 6), after_one.llist.y_range)

        self.assertEqual(6, after_one.ulist.val)
        self.assertIsNone(after_one.ulist.next)
        self.assertEqual(range(5, 7), after_one.ulist.y_range)


if __name__ == '__main__':
    a = [1, 2, 3, 4, 5, 6, 7]
    b = [7, 3, 1, 4, 5, 2, 6]

    b_index = dict(zip(b, range(len(b))))

    ylists = YLists(sig_a=a.__getitem__, sig_b_inv=b_index.__getitem__, n=len(a))
    ylists.pretty_print()

    print("#" * 120)
    after_one = ylists.update_for_x(5)
    after_one.pretty_print()

    after_one.display_plot()
