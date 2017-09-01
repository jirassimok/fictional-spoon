"""
Project: SearchAlgorithms
File: Main
Created on: 8/28/2017

Project Description:
CS4341 - Project 1
Search algorithms

A project in which several search algorithms are implemented.
"""
from abc import ABC, abstractmethod, ABCMeta
from collections import OrderedDict
from functools import reduce
from itertools import starmap
import re
from typing import NamedTuple, List, Dict, Set, FrozenSet, Callable, Iterable, MutableMapping, Union, Tuple
import sys

Path = Union[Tuple[str], List[str]]

class GraphABC(ABC):
    @abstractmethod
    def heuristic(node: str) -> float:
        """Get the heuristic value for a node, or 0 if the node has no heuristic value.
        """
    @abstractmethod
    def neighbors(node: str) -> Set[str]:
        """Get a safe-to-modify set of e the neighbors of the given node.
        """
    @abstractmethod
    def distance(a: str, b: str) -> float:
        """Get the distance between two nodes.

        If the nodes are not connected, KeyError.
        """
    @abstractmethod
    def path_cost(path: Path) -> float:
        """Get the total distance of the edges connecting the listed nodes.

        If any two adjacent nodes in the path are not connected, KeyError.
        """

class Graph(GraphABC):
    def __init__(self,
                 edge_lengths: Dict[FrozenSet[str], float],
                 heuristics: Dict[str, float]):
        self._edges = frozenset(edge_lengths.items())
        self._heuristics = frozenset(heuristics.items())
    @property
    def edge_lengths(self) -> Dict[FrozenSet[str], float]:
        """A mapping from pairs of adjacent nodes to the distance between them.
        """
        return dict(self._edges)
    @property
    def edges(self) -> Set[FrozenSet[str]]:
        """A set of all pairs of adjacent nodes.
        """
        return {edge[0] for edge in self._edges}
    @property
    def heuristics(self) -> Dict[str, float]:
        """A mapping from nodes to heuristics
        """
        return dict(self._heuristics)

    def heuristic(self, node: str) -> float:
        """Get the heuristic value for a node, or 0 if the node has no heuristic value.
        """
        return self.heuristics.get(node, 0)

    def neighbors(self, node: str) -> Set[str]:
        """Get a safe-to-modify set of e the neighbors of the given node.
        """
        return {n for edge in self.edges
                for n in edge if node in edge} - {node}

    def distance(self, a: str, b: str) -> float:
        """Get the distance between two nodes.

        If the nodes are not connected, KeyError.
        """
        return self.edge_lengths[frozenset({a, b})]

    def path_cost(self, path: Path) -> float:
        """Get the total distance of the edges connecting the listed nodes.

        If any two adjacent nodes in the path are not connected, KeyError.
        """
        return sum(starmap(self.distance, zip(path, path[1:])))

    @classmethod
    def from_lines(cls, lines: Iterable[str]) -> 'Graph':
        edges, edge_regex = [], r'([A-Z])\s+([A-Z])\s+(\S+)\s+'
        heuristics, heuristic_regex = [], r'([A-Z])\s+(\S+)\s+'

        section, line_regex = edges, edge_regex
        for line in lines:
            # Allow comments and blank lines, for convenience
            if line.startswith("//") or re.fullmatch(r'\s*', line) is not None:
                continue
            elif re.fullmatch(r'#####\s+', line) is not None:
                if section is edges:
                    section = heuristics
                    line_regex = heuristic_regex
                    continue
                elif section is heuristics:
                    raise ValueError("Already in heuristic section")
                assert False, "Unreachable state"

            match = re.fullmatch(line_regex, line)
            if match is None:
                raise ValueError("Badly-formatted line: " + line)
            else:
                section.append(match.groups())

        # Check for illegal duplicates

        heuristic_nodes = set(map(lambda x: x[0], heuristics))
        if len(heuristics) != len(heuristic_nodes):
            raise ValueError("Multiple heuristics for same node")

        edge_pairs = set(map(lambda x: frozenset(x[:2]), edges))
        if len(edges) != len(edge_pairs):
            raise ValueError("Multiple edges between same nodes")

        # Convert for graph creation
        edges = dict(map(lambda x: (frozenset(x[:2]), float(x[2])), edges))
        heuristics = {node: float(h) for node, h in heuristics}

        # Validate that node names are good
        edge_nodes = {name for edge in edges for name in edge}
        all_nodes = edge_nodes.union(heuristics.keys())
        no_heuristic_nodes = edge_nodes.difference(heuristics.keys())
        if len(no_heuristic_nodes) != 1:
            raise ValueError("More than one node has no heuristic value")

        return cls(edges, heuristics)


class TreeNode:
    def __init__(self, graph: Graph, state: str, parent: 'TreeNode' = None):
        self.graph = graph
        self.state = state
        self.parent = parent

    def trace_path(self, accumulated_path=None) -> List['TreeNode']:
        accumulated_path = accumulated_path if accumulated_path else []
        accumulated_path.append(self)
        return self.parent.trace_path(accumulated_path) if self.parent is not None else accumulated_path

    @staticmethod
    def _print_path(path: List['TreeNode']):
        return '<{}>'.format(', '.join(node.state for node in path))

    def get_trace_path(self) -> str:
        return self._print_path(self.trace_path())

    def expand(self) -> List['TreeNode']:
        return list(self.__class__(self.graph, state, self) for state in self.graph.neighbors(self.state))


class Problem(NamedTuple):
    graph: Graph
    initial_state: str
    solution_state: str


SearchMethod = Callable[[List[TreeNode], List[TreeNode]], List[TreeNode]]

def General_Search(problem: Problem, search_method: SearchMethod):
    print('   Expanded  Queue')

    queue = [TreeNode(problem.graph, problem.initial_state, parent=None)]
    while queue:
        node = queue[0]
        print('      {}      [{}]'.format(node.state, ' '.join(node.get_trace_path() for node in queue)))
        node = queue.pop(0)
        if node.state == problem.solution_state:
            print('      goal reached!')
            return node.state
        opened_nodes = node.expand()
        opened_nodes = [opened_node for opened_node in opened_nodes
                        if opened_node.state not in [ancestor_node.state for ancestor_node in node.trace_path()]]
        queue = search_method(queue, opened_nodes)
    print(f'   failure to find path between {problem.initial_state} and {problem.solution_state}')
    return None


def main(search_methods: Dict[str, SearchMethod], argv):
    args = argv[1:]
    assert(len(args) == 1), f'Must have exactly 1 argument. Arguments detected: {args}'

    with open(args[0], 'r') as f:
        graph = Graph.from_lines(f)

    first_problem = Problem(graph, 'S', 'G')

    for name, function in search_methods.items():
        print(name)
        General_Search(first_problem, function)
        print()

"""
SEARCH METHOD DEFINITIONS
"""
search_methods: MutableMapping[str, SearchMethod] = OrderedDict()
def searchmethod(name: str):
    """Register a search method with a name
    """
    def decorator(function):
        global search_methods
        search_methods[name] = function
        return function
    return decorator

@searchmethod('Depth 1st search')
def depth_first_search(queue: List[TreeNode], new_nodes_list: List[TreeNode]) -> List[TreeNode]:
    return sorted(new_nodes_list, key=lambda treenode: treenode.state) + queue


@searchmethod('Breadth 1st search')
def breadth_first_search(queue: List[TreeNode], new_nodes_list: List[TreeNode]) -> List[TreeNode]:
    return queue + sorted(new_nodes_list, key=lambda treenode: treenode.state)


@searchmethod("Depth-limited search (depth limit 2)")
def depth_limited_2(open_nodes, new_nodes):
    return depth_limited(2, open_nodes, new_nodes)
def depth_limited(n, open_nodes, new_nodes):
    print("   Not Implemented")
    ...


@searchmethod("Iterative deepening search")
def iterative_deepening(open_nodes, new_nodes):
    print("   Not Implemented")
    ...


@searchmethod("Uniform cost search")
def uniform_cost(open_nodes, new_nodes):
    print("   Not Implemented")
    ...


@searchmethod("Greedy search")
def greedy(open_nodes, new_nodes):
    print("   Not Implemented")
    ...


@searchmethod("A*")
def astar(open_nodes, new_nodes):
    print("   Not Implemented")
    ...


@searchmethod("Hill-climbing search")
def hill_climbing(open_nodes, new_nodes):
    print("   Not Implemented")
    ...


@searchmethod("Beam search (w=2)")
def beam_2(open_nodes, new_nodes):
    return beam(2, open_nodes, new_nodes)
def beam(n, open_nodes, new_nodes):
    print("   Not Implemented")
    ...


# Will run at script execution
if __name__ == '__main__':
    main(search_methods, sys.argv)
