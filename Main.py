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
import re
from typing import NamedTuple, List, Dict, Set, FrozenSet, Callable, Iterable, MutableMapping, Union, Tuple
import sys

Path = Union[Tuple[str], List[str]]
SearchMethod = Callable[['Graph', List[Path], List[Path]], List[Path]]

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
        return sum(map(self.distance, path, path[1:]))

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

        heuristic_nodes = {x[0] for x in heuristics}
        if len(heuristics) != len(heuristic_nodes):
            raise ValueError("Multiple heuristics for same node")

        edge_pairs = {frozenset(edge[:2]) for edge in edges}
        if len(edges) != len(edge_pairs):
            raise ValueError("Multiple edges between same nodes")

        # Convert for graph creation
        edges = {frozenset(edge[:2]): float(edge[2]) for edge in edges}
        heuristics = {node: float(h) for node, h in heuristics}

        # Validate that node names are good
        edge_nodes = {name for edge in edges for name in edge}
        all_nodes = edge_nodes.union(heuristics.keys())
        no_heuristic_nodes = edge_nodes.difference(heuristics.keys())
        if len(no_heuristic_nodes) != 1:
            raise ValueError("More than one node has no heuristic value")

        return cls(edges, heuristics)

class Problem(NamedTuple):
    graph: Graph
    start: str
    goal: str

def general_search(problem: Problem, search_method: SearchMethod):
    graph = problem.graph
    start = problem.start
    goal = problem.goal
    paths = [(start,)]

    print('   Expanded  Queue')
    while paths:
        expanded = paths[0][0]
        print(f'      {expanded}      ',
              '[<', '> <'.join(','.join(path) for path in paths), '>]',
              sep='')
        path = paths.pop(0)
        if expanded == goal:
            print('      goal reached!')
            return path
        opened_paths = {(node,)+path for node in graph.neighbors(expanded) if node not in path}
        paths = search_method(graph, paths, opened_paths)
    print(f'   failure to find path between {start} and {goal}')
    return None

def main(search_methods: Dict[str, SearchMethod], argv):
    args = argv[1:]
    assert(len(args) == 1), f'Must have exactly 1 argument. Arguments detected: {args}'

    with open(args[0], 'r') as f:
        graph = Graph.from_lines(f)

    first_problem = Problem(graph, 'S', 'G')

    for name, function in search_methods.items():
        print(name)
        general_search(first_problem, function)
        print()

"""
SEARCH METHOD HELPERS
"""
# Mapping of algorithm names to functions
search_methods: MutableMapping[str, SearchMethod] = OrderedDict()
def searchmethod(name: str):
    """Register a search method with a name
    """
    def decorator(function):
        global search_methods
        search_methods[name] = function
        return function
    return decorator

def print_paths(paths: List[Path],
                cost_calculator: Callable[[List[Path]], float]) -> str:
    """Stringify the paths with costs as formatted by the given function.
    """
    pass



"""
SEARCH METHOD FUNCTIONS
"""

@searchmethod('Depth 1st search')
def depth_first(graph: Graph, open_paths: List[Path], new_paths: List[Path]) -> List[Path]:
    return sorted(new_paths) + open_paths


@searchmethod('Breadth 1st search')
def breadth_first(graph: Graph, open_paths: List[Path], new_paths: List[Path]) -> List[Path]:
    return open_paths + sorted(new_paths)


@searchmethod("Depth-limited search (depth limit 2)")
def depth_limited_2(graph: Graph, open_paths, new_paths):
    return depth_limited(2, open_paths, new_paths)
def depth_limited(n, open_paths, new_paths):
    print("   Not Implemented")
    ...


@searchmethod("Iterative deepening search")
def iterative_deepening(graph: Graph, open_paths, new_paths):
    print("   Not Implemented")
    ...


@searchmethod("Uniform cost search")
def uniform_cost(graph: Graph, open_paths, new_paths):
    print("   Not Implemented")
    ...


@searchmethod("Greedy search")
def greedy(graph: Graph, open_paths, new_paths):
    print("   Not Implemented")
    ...


@searchmethod("A*")
def astar(graph: Graph, open_paths, new_paths):
    print("   Not Implemented")
    ...


@searchmethod("Hill-climbing search")
def hill_climbing(graph: Graph, open_paths, new_paths):
    print("   Not Implemented")
    ...


@searchmethod("Beam search (w=2)")
def beam_2(graph: Graph, open_paths, new_paths):
    return beam(2, open_paths, new_paths)
def beam(n, open_paths, new_paths):
    print("   Not Implemented")
    ...


# Will run at script execution
if __name__ == '__main__':
    main(search_methods, sys.argv)
