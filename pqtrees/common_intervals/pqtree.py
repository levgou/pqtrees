from pprint import pprint

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

from funcy import pairwise

from pqtrees.common_intervals.common_interval import CommonInterval
from pqtrees.common_intervals.generate_s import IntervalHierarchy
from pqtrees.common_intervals.reduce_intervals import ReduceIntervals


class QNode:
    def __init__(self, ci) -> None:
        self.children = []
        self.ci = ci

    def __str__(self) -> str:
        return f"Q{self.ci}"



class PNode:
    def __init__(self, ci) -> None:
        self.children = []
        self.ci = ci

    def __str__(self) -> str:
        return f"P{self.ci}"


class LeafNode:
    def __init__(self, ci) -> None:
        self.ci = ci

    def __str__(self) -> str:
        return f"L{self.ci}"


class PQTree:

    def __init__(self, root) -> None:
        super().__init__()
        self.root = root


    def show(self):
        g = nx.DiGraph()

        def rec_construct_graph(node):
            children = getattr(node, 'children', [])
            [g.add_edge(node, child) for child in children]
            [rec_construct_graph(child) for child in children]

        rec_construct_graph(self.root)

        NODE_SIZE = 5000
        FIG_EDGE_SIZE = 8

        plt.figure(1, figsize=(FIG_EDGE_SIZE, FIG_EDGE_SIZE))
        plt.title('draw_networkx')
        pos = graphviz_layout(g, prog='dot')
        nx.draw(g, pos, with_labels=True, node_size=NODE_SIZE, node_shape='8')
        plt.show()

    @classmethod
    def from_s(cls, s_intervals: IntervalHierarchy) -> 'PQTree':
        construction_lut = {}

        for ci in s_intervals.iter_bottom_up():
            if ci in construction_lut:
                continue

            if ci.is_trivial():
                print("Adding ", ci.first_start, ci.first_end)
                construction_lut[ci.first_start, ci.first_end] = LeafNode(ci)
                continue

            l_up, l_down, r_up, r_down = s_intervals.s_arrows_of(ci)
            print(ci, "-->", l_up, l_down, r_up, r_down)

            if chain := s_intervals.chain_starting_with(ci):
                print(">>> Chain starting with", ci)
                qnode = QNode(ci)

                for ci1, ci2 in pairwise(chain):
                    only_in_1 = (ci1.first_start, ci2.first_start - 1)
                    intersection = (ci2.first_start, ci1.first_end)

                    qnode.children.append(construction_lut[only_in_1])
                    qnode.children.append(construction_lut[intersection])

                else:
                    only_in_2 = (ci1.first_end + 1, ci2.first_end)
                    qnode.children.append(only_in_2)

                print("Adding ", chain[0].first_start, chain[-1].first_end)
                print(" |- And ", chain[0].first_start, chain[0].first_end)
                construction_lut[chain[0].first_start, chain[-1].first_end] = qnode
                construction_lut[chain[0].first_start, chain[0].first_end] = qnode

            else:
                print("@@@ Not Chain with", ci)
                pnode = PNode(ci)
                if l_down is not None:
                    pnode.children.append(construction_lut[l_down.first_start, l_down.first_end])
                if r_down is not None:
                    pnode.children.append(construction_lut[r_down.first_start, r_down.first_end])

                print("Adding ", ci.first_start, ci.first_end)
                construction_lut[ci.first_start, ci.first_end] = pnode

        return PQTree(construction_lut[s_intervals.boundaries])


def known_example():
    commons1 = [

        # len 1
        *[CommonInterval((i, i)) for i in range(9)],

        # len 2
        CommonInterval((0, 1), (6, 7), (0, 1)),
        CommonInterval((1, 2), (7, 8), (1, 2)),
        CommonInterval((3, 4), (2, 3), (5, 6)),
        CommonInterval((4, 5), (3, 4), (6, 7)),

        # len 3
        CommonInterval((0, 2), (6, 8), (0, 2)),
        CommonInterval((3, 5), (2, 4), (5, 7)),

        # len 4
        CommonInterval((3, 6), (2, 5), (4, 7)),

        # len 5
        CommonInterval((3, 7), (1, 5), (3, 7)),

        # len 6
        CommonInterval((3, 8), (0, 5), (3, 8)),

        # len 8
        CommonInterval((0, 7), (1, 8), (0, 7)),

        # len 9
        CommonInterval((0, 8), (0, 8), (0, 8)),
    ]

    ir_intervals = ReduceIntervals.reduce(commons1)
    s = IntervalHierarchy.from_irreducible_intervals(ir_intervals)
    pprint(s.nesting_levels)

    PQTree.from_s(s).show()


if __name__ == '__main__':
    known_example()
