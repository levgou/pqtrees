import json
import math
import operator
from collections import defaultdict
from functools import reduce
from itertools import permutations, product, tee
from typing import List, Union, Callable, Dict, Optional, Iterable, Mapping

import matplotlib.pyplot as plt
import networkx as nx
from frozendict import frozendict
from networkx.drawing.nx_agraph import graphviz_layout

from funcy import pairwise, lmap

from pqtrees.common_intervals.common_interval import CommonInterval
from pqtrees.pqtree_helpers.generate_s import IntervalHierarchy
from pqtrees.common_intervals.preprocess_find import common_k_indexed_with_singletons
from pqtrees.proj_types import Interval, Index
from pqtrees.pqtree_helpers.reduce_intervals import ReduceIntervals
from pqtrees.utilities.iterator_product import IterProduct


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


TreeNode = Union['PQNode', 'LeafNode']


def gen_denormalize(denormalizer: Dict[int, object]):
    def denormalize(char: int):
        denorm = denormalizer[char]
        return denorm.org if hasattr(denorm, 'org') else denorm

    return denormalize


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

    def dict_repr(self, ommit_multi_info: bool):
        return {
            "type": self.__class__.__name__,
            "children": [c.dict_repr(ommit_multi_info) for c in self.children]
        }

    def with_children(self, children):
        self.children = children
        return self

    def add_child(self, c):
        if c in self.children:
            raise Exception(f"{c} already in {self}")

        # print(f"Adding {c} to {self}")
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

    def translate(self, denormalizer: Callable[[int], object]):
        self.interval = tuple(map(denormalizer, self.interval))
        for c in self.children:
            c.translate(denormalizer)

    def iter(self, parent: Optional['PQNode'] = None) -> Iterable[TreeNode]:
        yield self, parent
        for c in self.children:
            yield from c.iter(self)

    def replace_child(self, child: TreeNode, *with_nodes: TreeNode):
        child_index = self.children.index(child)
        self.children = (
            *self.children[:child_index],
            *with_nodes,
            *self.children[child_index + 1:]
        )


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

    def no_reverse_frontier(self):
        for p in product(*[c.frontier() for c in self.children]):
            yield str("".join(p))

    def reverse_frontier(self):
        for s in self.no_reverse_frontier():
            yield s[::-1]

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

        children_fronts = [c.frontier() for c in self.children]
        for p in IterProduct.iproduct(*children_fronts):
            for perm in permutations(p):
                yield "".join(perm)

    def __eq__(self, other):
        if not isinstance(other, PNode):
            return False
        else:
            return self.interval == other.interval

    def __hash__(self):
        return super().__hash__()


class LeafNode:
    ci: CommonInterval
    multi_occurrences: Mapping[int, int]

    def __init__(self, ci: CommonInterval) -> None:
        self.ci = ci
        self.multi_occurrences = frozendict({1: 1})

    def __str__(self) -> str:
        return "L-" + str(self.ci)

    def to_parens(self):
        return self.ci.sign

    def dict_repr(self, ommit_multi_info: bool = False):
        times_char_occured = sum(self.multi_occurrences.values())
        mutli_stats = {
            num_occur: f"{times_occured}:{times_char_occured}" for num_occur, times_occured in
            self.multi_occurrences.items()
        }

        return {
            "type": "LEAF",
            "char": self.ci.sign,

            **({
                   "multi": self.multi,
                   "multi_stats": mutli_stats
               } if not ommit_multi_info else {})
        }

    def immute(self):
        return

    def __contains__(self, _):
        return False

    def frontier(self):
        for multiplier in self.multi_occurrences:
            yield str(self.ci.sign) * multiplier

    def __eq__(self, other):
        if not isinstance(other, LeafNode):
            return False
        else:
            return self.ci == other.ci

    def sort(self):
        return

    def __hash__(self):
        return hash(self.ci)

    def approx_frontier_size(self):
        return len(self.multi_occurrences)

    def translate(self, denormalizer: Callable[[int], object]):
        self.ci.sign = denormalizer(self.ci.first_start)

    def iter(self, parent: Optional['PQNode']):
        yield self, parent

    @property
    def multi(self):
        return set(self.multi_occurrences) != {1}


class PQTree:
    def __init__(self, root) -> None:
        super().__init__()
        self.root = root

    def to_parens(self):
        return self.root.to_parens()

    def frontier(self):
        return self.root.frontier()

    def approx_frontier_size(self):
        """
        Calculates approximate frontier size -
        in case there are same strings - they'll be counted for each occurrence
        """
        return self.root.approx_frontier_size()

    def dict_repr(self, ommit_multi_info: bool) -> dict:
        has_multi_chars = any(l.multi for l in self.iter_leafs())
        return {
            "approx_front_size": self.approx_frontier_size(),
            "root": self.root.dict_repr(ommit_multi_info),

            **({"has_multi_chars": has_multi_chars} if not ommit_multi_info else {})
        }

    def to_json(self, pretty=False, ommit_multi_info=False) -> str:
        kwargs = {"indent": 2} if pretty else {}
        return json.dumps(self.dict_repr(ommit_multi_info), **kwargs)

    def __iter__(self):
        return self.root.iter()

    def iter_leafs(self) -> Iterable[LeafNode]:
        for node, _ in iter(self):
            if isinstance(node, LeafNode):
                yield node

    def parent_of(self, node: TreeNode) -> Optional[PQNode]:
        for n, parent in self:
            if n == node:
                return parent
        return None

    def __str__(self):
        return f"PQ<{self.to_parens()}>"

    __repr__ = __str__


class PQTreeBuilder:
    @classmethod
    def from_perms(cls, perms) -> PQTree:
        normalized_perms, denormalize_dict = cls._normalize_perms(perms)
        # common_intervals = trivial_common_k_with_singletons(*normalized_perms)
        common_intervals = common_k_indexed_with_singletons(*normalized_perms)
        ir_intervals = ReduceIntervals.reduce(common_intervals)
        s = IntervalHierarchy.from_irreducible_intervals(ir_intervals)
        return cls._from_s(s, denormalize_dict)

    @classmethod
    def _normalize_perms(cls, perms):
        l_perms0 = list(perms[0])

        if l_perms0 == list(range(len(l_perms0))):
            return perms, dict(zip(l_perms0, l_perms0))

        normalizer = dict(zip(l_perms0, range(len(l_perms0))))
        norm_perms = tuple(tuple(normalizer[c] for c in perm) for perm in perms)
        denormalizer = {v: k for k, v in normalizer.items()}

        return norm_perms, denormalizer

    @classmethod
    def _from_s(cls, s_intervals: IntervalHierarchy, denormalizer: Optional[Dict[Index, object]] = None) -> 'PQTree':
        construction_lut = {}
        nodes_by_level = defaultdict(list)
        processed_intervals = set()

        for ci in s_intervals.iter_bottom_up():
            # print(f"Processing: {ci}")

            if ci in processed_intervals:
                continue
            else:
                processed_intervals.add(ci)

            if ci.is_trivial():
                cls._leaf_case(ci, construction_lut, nodes_by_level, s_intervals)
                continue

            if chain := s_intervals.chain_starting_with(ci):
                cls._start_of_chain_case(ci, chain, construction_lut, processed_intervals, nodes_by_level, s_intervals)

            else:
                cls._not_start_of_chain_case(ci, construction_lut, nodes_by_level, s_intervals)

        root = cls.post_process_tree(construction_lut[s_intervals.boundaries], denormalizer)
        return PQTree(root)

    @classmethod
    def post_process_tree(cls, root: PQNode, denormalizer_dict: Dict[Index, object]):
        root.sort()
        root.immute()

        if denormalizer_dict:
            root.translate(gen_denormalize(denormalizer_dict))

        return root

    @classmethod
    def _leaf_case(cls, ci, construction_lut, nodes_by_level, s_intervals):
        # print("Adding trivial", ci.first_start, ci.first_end)
        leaf = LeafNode(ci)
        construction_lut[ci.first_start, ci.first_end] = leaf
        nodes_by_level[s_intervals.reverse_index[ci]].append(leaf)

    @classmethod
    def _start_of_chain_case(cls, ci, chain, construction_lut, processed_intervals, nodes_by_level, s_intervals):
        # print(">>> Chain starting with", ci)
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
            # print(ci1, ci2, ci3)

        first, second = chain[:2]
        qnode.add_child(construction_lut[(first.first_start, second.first_start - 1)])

        prelast, last = chain[-2:]
        qnode.add_child(construction_lut[(prelast.first_end + 1, last.first_end)])

        # else:
        #     only_in_2 = (ci1.first_end + 1, ci2.first_end)
        #     qnode.add_child(construction_lut[only_in_2])

        # print("Adding QNode", chain[0].first_start, chain[-1].first_end)
        # print(" |- And ", chain[0].first_start, chain[0].first_end)
        construction_lut[chain[0].first_start, chain[-1].first_end] = qnode
        construction_lut[chain[0].first_start, chain[0].first_end] = qnode
        nodes_by_level[s_intervals.reverse_index[ci]].append(qnode)

    @classmethod
    def _not_start_of_chain_case(cls, ci, construction_lut, nodes_by_level, s_intervals):
        # print("@@@ Not Chain with", ci)
        pnode = PNode.from_interval(ci)

        for lower_node in nodes_by_level[s_intervals.reverse_index[ci] + 1]:
            if lower_node in pnode:
                pnode.add_child(lower_node)

        # print("Adding Pnode", ci.first_start, ci.first_end)
        pnode = pnode.to_qnode_if_needed()
        construction_lut[ci.first_start, ci.first_end] = pnode
        nodes_by_level[s_intervals.reverse_index[ci]].append(pnode)


class PQTreeVisualizer:
    FONT_SIZE = 16
    NODE_COLORS = {
        'Q': '#FFD49B',
        'P': '#9BE9FF',
        'L': '#90BB3323'
    }

    NODE_SIZE = 5000
    FIG_EDGE_SIZE = 8
    NODE_SHAPE = '8'  # hexagonal

    SUFFIXES = "*+%$#@&~^?<"

    @classmethod
    def gen_leaf_strs(cls, pqtree: PQTree):
        leaf_str_reprs = lmap(str, pqtree.iter_leafs())
        leaf_suffixes = {l: cls.SUFFIXES for l in leaf_str_reprs}

        leaf_strs = {}
        for leaf in pqtree.iter_leafs():
            leaf_str = str(leaf)
            if leaf_str_reprs.count(leaf_str) == 1:
                leaf_strs[leaf] = leaf_str
            else:
                leaf_strs[leaf] = leaf_str + leaf_suffixes[leaf_str][0]
                leaf_suffixes[leaf_str] = leaf_suffixes[leaf_str][1:]
        return leaf_strs

    @classmethod
    def show(cls, pqtree: PQTree, figure_index=1, skip_show=False, title='PQTree'):
        g = nx.DiGraph()
        leaf_strs = cls.gen_leaf_strs(pqtree)

        def child_str(child):
            if isinstance(child, LeafNode):
                return leaf_strs[child]
            return str(child)

        def rec_construct_graph(node):
            children = getattr(node, 'children', [])
            children_strs = map(child_str, children)
            [g.add_edge(str(node), child) for child in children_strs]
            [rec_construct_graph(child) for child in children]

        rec_construct_graph(pqtree.root)

        plt.figure(figure_index, figsize=(cls.FIG_EDGE_SIZE, cls.FIG_EDGE_SIZE))
        plt.title(title)
        pos = graphviz_layout(g, prog='dot')

        node_colors = lmap(lambda n: cls.NODE_COLORS[n[0]], g.nodes)

        nx.draw(g, pos, with_labels=True, font_size=cls.FONT_SIZE, font_weight='bold', node_size=cls.NODE_SIZE,
                node_shape=cls.NODE_SHAPE, node_color=node_colors)

        if not skip_show:
            plt.show()

    @classmethod
    def show_all(cls, *pqtrees: PQTree):
        for i, tree in enumerate(pqtrees):
            cls.show(tree, i, True)
        plt.show()


if __name__ == '__main__':
    pqtree = PQTreeBuilder.from_perms(
        [(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), (14, 10, 12, 7, 1, 3, 5, 13, 8, 2, 6, 11, 9, 4, 15)])
    print(pqtree.to_parens())
    print(pqtree.to_json(pretty=True))
    PQTreeVisualizer.show(pqtree)
