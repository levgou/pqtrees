import math
import operator
from collections import defaultdict
from functools import reduce
from itertools import permutations, product, tee
from pprint import pprint
from typing import List, Union, Callable

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

from funcy import pairwise, lmap

from pqtrees.common_intervals.common_interval import CommonInterval
from pqtrees.common_intervals.generate_s import IntervalHierarchy
from pqtrees.common_intervals.proj_types import Interval
from pqtrees.common_intervals.reduce_intervals import ReduceIntervals
from pqtrees.common_intervals.trivial import trivial_common_k, trivial_common_k_with_singletons


def interval_in_interval(small: Interval, big: Interval):
    return small[0] >= big[0] and small[1] <= big[1]


def iter3(seq):
    """Yields all triplets of neighboring items in seq."""
    a, b = tee(seq)
    c = iter(seq)
    next(b, None)
    next(c, None)
    next(c, None)

    return zip(a, b, c)


def tree_sort_key(node: Union['PQNode', 'LeafNode']):
    if isinstance(node, LeafNode):
        return node.ci.first_start
    else:
        return node.interval[0]


class PQNode:
    def __init__(self, interval: Interval) -> None:
        self.children = []
        self.interval = interval

    def __str__(self) -> str:
        return f"{self.interval}"

    def __contains__(self, other: Union['QNode', 'PNode', 'LeafNode']):
        if isinstance(other, LeafNode):
            return interval_in_interval(other.ci.to_tuple(), self.interval)
        return interval_in_interval(other.interval, self.interval)

    def to_parens(self):
        return " ".join(map(lambda x: x.to_parens(), self.children))

    def with_children(self, children):
        self.children = children
        return self

    def add_child(self, c):
        if c in self.children:
            raise Exception(f"{c} already in {self}")

        print(f"Adding {c} to {self}")
        self.children.append(c)

    def sort(self):
        for c in self.children:
            c.sort()
        self.children.sort(key=tree_sort_key)

    def immute(self):
        for c in self.children:
            c.immute()
        self.children = tuple(self.children)

    def __hash__(self):
        return hash(self.interval)

    def approx_frontier_size(self):
        node_multiply_factor = 2 if isinstance(self, QNode) else math.factorial((len(self.children)))
        children_frontier_sizes = map(lambda c: c.approx_frontier_size(), self.children)
        f_size = reduce(operator.mul, children_frontier_sizes, node_multiply_factor)
        return f_size


class QNode(PQNode):

    def __str__(self) -> str:
        return f"Q{self.interval}"

    @classmethod
    def from_chain(cls, chain: List[CommonInterval]) -> 'QNode':
        return cls((chain[0].first_start, chain[-1].first_end))

    def to_parens(self):
        return "[" + super().to_parens() + "]"

    def frontier(self):
        for p in product(*[c.frontier() for c in self.children]):
            yield str("".join(p))
            yield str("".join(reversed(p)))

    def __eq__(self, other):
        if not isinstance(other, QNode):
            return False
        else:
            return self.interval == other.interval

    def __hash__(self):
        return super().__hash__()


class PNode(PQNode):

    def __str__(self) -> str:
        return f"P{self.interval}"

    @classmethod
    def from_interval(cls, ci: CommonInterval) -> 'PNode':
        return cls((ci.first_start, ci.first_end))

    def to_parens(self):
        return "(" + super().to_parens() + ")"

    def to_qnode_if_needed(self):
        if len(self.children) > 2:
            return self
        return QNode(self.interval).with_children(self.children)

    def frontier(self):
        for p in product(*[c.frontier() for c in self.children]):
            front = "".join(p)
            for perm in permutations(front):
                yield "".join(perm)

    def __eq__(self, other):
        if not isinstance(other, PNode):
            return False
        else:
            return self.interval == other.interval

    def __hash__(self):
        return super().__hash__()


class LeafNode:
    def __init__(self, ci: CommonInterval) -> None:
        self.ci = ci

    def __str__(self) -> str:
        return "L-" + str(self.ci)

    def to_parens(self):
        return str(self.ci.first_end)

    def immute(self):
        return

    def __contains__(self, _):
        return False

    def frontier(self):
        yield str(self.ci.first_end)

    def __eq__(self, other):
        if not isinstance(other, LeafNode):
            return False
        else:
            return self.ci == other.ci

    def sort(self):
        return

    def __hash__(self):
        return hash(self.ci)

    def frontier_size(self):
        return 1

    approx_frontier_size = frontier_size


class PQTree:
    def __init__(self, root) -> None:
        super().__init__()
        root.sort()
        root.immute()
        self.root = root

    def to_parens(self):
        return self.root.to_parens()

    def frontier(self):
        return set(self.root.frontier())

    def approx_frontier_size(self):
        """
        Calculates approximate frontier size -
        in case there are same strings - they'll counted for each occurrence
        """
        return self.root.approx_frontier_size()


class PQTreeBuilder:

    @classmethod
    def from_perms(cls, perms):

        # todo: note denormalize_dict
        # todo: note denormalize_dict
        # todo: note denormalize_dict
        # todo: note denormalize_dict

        normalized_perms, denormalize_dict = cls.normalize_perms(perms)
        common_intervals = trivial_common_k_with_singletons(*normalized_perms)
        ir_intervals = ReduceIntervals.reduce(common_intervals)
        s = IntervalHierarchy.from_irreducible_intervals(ir_intervals)
        return cls.from_s(s)

    @classmethod
    def normalize_perms(cls, perms):
        l_perms0 = list(perms[0])

        if l_perms0 == list(range(len(l_perms0))):
            return perms, dict(zip(l_perms0, l_perms0))

        normalizer = dict(zip(l_perms0, range(len(l_perms0))))
        norm_perms = tuple(tuple(normalizer[c] for c in perm) for perm in perms)
        denormalizer = {v: k for k, v in normalizer.items()}

        return norm_perms, denormalizer

    @classmethod
    def from_s(cls, s_intervals: IntervalHierarchy) -> 'PQTree':
        construction_lut = {}
        nodes_by_level = defaultdict(list)
        processed_intervals = set()

        for ci in s_intervals.iter_bottom_up():
            print(f"Processing: {ci}")

            if ci in processed_intervals:
                continue
            processed_intervals.add(ci)

            if ci.is_trivial():
                cls._leaf_case(ci, construction_lut, nodes_by_level, s_intervals)
                continue

            if chain := s_intervals.chain_starting_with(ci):
                cls._start_of_chain_case(ci, chain, construction_lut, processed_intervals, nodes_by_level, s_intervals)

            else:
                cls._not_start_of_chain_case(ci, construction_lut, nodes_by_level, s_intervals)

        return PQTree(construction_lut[s_intervals.boundaries])

    @classmethod
    def _leaf_case(cls, ci, construction_lut, nodes_by_level, s_intervals):
        print("Adding trivial", ci.first_start, ci.first_end)
        leaf = LeafNode(ci)
        construction_lut[ci.first_start, ci.first_end] = leaf
        nodes_by_level[s_intervals.reverse_index[ci]].append(leaf)

    @classmethod
    def _start_of_chain_case(cls, ci, chain, construction_lut, processed_intervals, nodes_by_level, s_intervals):
        print(">>> Chain starting with", ci)
        qnode = QNode.from_chain(chain)

        for ci1, ci2 in pairwise(chain):
            # only_in_1 = (ci1.first_start, ci2.first_start - 1)
            intersection = (ci2.first_start, ci1.first_end)
            # print(ci1, ci2, only_in_1, intersection)
            # qnode.add_child(construction_lut[only_in_1])
            qnode.add_child(construction_lut[intersection])
            processed_intervals.add(ci2)

        for ci1, ci2, ci3 in iter3(chain):
            if ci3.first_start - 1 == ci1.first_end:
                continue
            only_in_2 = (ci1.first_end + 1, ci3.first_start - 1)
            qnode.add_child(construction_lut[only_in_2])
            print(ci1, ci2, ci3)

        first, second = chain[:2]
        qnode.add_child(construction_lut[(first.first_start, second.first_start - 1)])

        prelast, last = chain[-2:]
        qnode.add_child(construction_lut[(prelast.first_end + 1, last.first_end)])

        # else:
        #     only_in_2 = (ci1.first_end + 1, ci2.first_end)
        #     qnode.add_child(construction_lut[only_in_2])

        print("Adding QNode", chain[0].first_start, chain[-1].first_end)
        print(" |- And ", chain[0].first_start, chain[0].first_end)
        construction_lut[chain[0].first_start, chain[-1].first_end] = qnode
        construction_lut[chain[0].first_start, chain[0].first_end] = qnode
        nodes_by_level[s_intervals.reverse_index[ci]].append(qnode)

    @classmethod
    def _not_start_of_chain_case(cls, ci, construction_lut, nodes_by_level, s_intervals):
        print("@@@ Not Chain with", ci)
        pnode = PNode.from_interval(ci)

        for lower_node in nodes_by_level[s_intervals.reverse_index[ci] + 1]:
            if lower_node in pnode:
                pnode.add_child(lower_node)

        print("Adding Pnode", ci.first_start, ci.first_end)
        pnode = pnode.to_qnode_if_needed()
        construction_lut[ci.first_start, ci.first_end] = pnode
        nodes_by_level[s_intervals.reverse_index[ci]].append(pnode)


class PQTreeVisualizer:
    NODE_COLORS = {
        QNode: '#32a852',
        PNode: '#9032a8',
        LeafNode: '#c7b17d'
    }

    @classmethod
    def show(cls, pqtree: PQTree):
        g = nx.DiGraph()

        def rec_construct_graph(node):
            children = getattr(node, 'children', [])
            [g.add_edge(node, child) for child in children]
            [rec_construct_graph(child) for child in children]

        rec_construct_graph(pqtree.root)

        NODE_SIZE = 5000
        FIG_EDGE_SIZE = 8

        plt.figure(1, figsize=(FIG_EDGE_SIZE, FIG_EDGE_SIZE))
        plt.title('PQTree')
        pos = graphviz_layout(g, prog='dot')

        node_colors = lmap(lambda n: cls.NODE_COLORS[n.__class__], g.nodes)

        nx.draw(g, pos, with_labels=True, node_size=NODE_SIZE, node_shape='8', node_color=node_colors)
        plt.show()


if __name__ == '__main__':
    # known_example()
    # known_e2e()
    # known_e2e_2()

    1
