"""
Project: SearchAlgorithms
File: Main
Created on: 8/28/2017

Project Description:
CS4341 - Project 1
Search algorithms

A project in which several search algorithms are implemented.
"""
from abc import ABC, ABCMeta, abstractmethod
from functools import reduce
import re
import itertools
from typing import NamedTuple, List, Dict, Set, FrozenSet, Callable, Iterable, MutableMapping, Union, Tuple
import warnings
import sys

Path = Union[Tuple[str], List[str]]

# Forward references for as-yet-undefined types
SearchMethod, Problem, Graph = 'SearchMethod', 'Problem', 'Graph'


def main(search_methods: List[SearchMethod], argv):
    args = argv[1:]
    assert(len(args) == 1), f'Must have exactly 1 argument. Arguments detected: {args}'

    with open(args[0], 'r') as f:
        graph = Graph.from_lines(f)

    first_problem = Problem(graph, 'S', 'G')

    print(search_methods)
    for method in search_methods:
        print(method.name)
        general_search(first_problem, method)
        print()


def general_search(problem: Problem, search_method: SearchMethod):
    graph = problem.graph
    start = problem.start
    goal = problem.goal
    paths = [(start,)]

    print('   Expanded  Queue')
    while paths:
        expanded = paths[0][0]
        print(f'      {expanded}      ',
              search_method.paths_to_str(graph, paths), sep='')
              # '[<', '> <'.join(','.join(path) for path in paths), '>]',
              # sep='')
        path = paths.pop(0)
        if expanded == goal:
            print('      goal reached!')
            return path
        opened_paths = {(node,)+path for node in graph.neighbors(expanded) if node not in path}
        paths = search_method(graph, paths, opened_paths)
    print(f'   failure to find path between {start} and {goal}')
    return None



class Problem(NamedTuple):
    graph: Graph
    start: str
    goal: str

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
    def path_length(path: Path) -> float:
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

    def path_length(self, path: Path) -> float:
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


class SearchMethodMeta(type):
    """Metaclass for SearchMethod

    Allows calling SearchMethods like functions.
    Prevents modification of name and search method.
    Provides convenient string representation.
    Automatically makes 'search' and 'paths_to_str' static methods.
      (If 'search' is already static, this will break it.)
    """
    def __call__(cls, graph: Graph, open_paths: List[Path], new_paths: List[Path], **kwargs) -> List[Path]:
        return cls.search(graph, open_paths, new_paths, **kwargs)
    def __setattr__(cls, key, value):
        if key == "name" or key == "search":
            raise AttributeError("can't set SearchMethod name or search method")
        else:
            ABCMeta.__setattr__(cls, key, value)
    def __str__(cls):
        return f"SearchMethod({cls.name})"
    def __new__(cls, name, bases, attrs):
        # Make search and path_to_str static
        SearchMethodMeta._staticize(attrs, "search")
        SearchMethodMeta._staticize(attrs, "paths_to_str")
        return type.__new__(cls, name, bases, attrs)
    @staticmethod
    def _staticize(attrs, attr):
        if attr in attrs:
            attrs[attr] = staticmethod(attrs[attr])

# These abstract methods will not be enforced, because the subclasses are never instantiated
class SearchMethod(object, metaclass=SearchMethodMeta):

    def search(graph: Graph, open_paths: List[Path], new_paths: Set[Path], **kwargs) -> List[Path]:
        raise TypeError("Can not call abstract search method")

    def paths_to_str(graph: Graph, paths: Iterable[Path]) -> None:
        """Print a path in a nice format.

        The default implementation does not print costs.
        """
        return ''.join(('[<', '> <'.join(','.join(path) for path in paths), '>]'))



"""
HELPER METHODS
"""
def paths_to_str_cost(paths: List[Path], cost_calculator: Callable[[Path], float]) -> str:
    output = ['[']
    for path in paths:
        cost = cost_calculator(path)
        output.extend((
            str(cost),
            '<',
            ','.join(path),
            '> '))
    output[-1] = '>]'
    return ''.join(output)

def groupby(iterable, key=None):
    """Version of itertools.groupby that sorts by the key first
    """
    if key is None:
        key = lambda x: x
    return itertools.groupby(sorted(iterable, key=key), key=key)

"""
SEARCH METHOD IMPLEMENTATIONS
"""

class depth_first(SearchMethod):
    name = 'Depth 1st search'
    def search(graph: Graph, open_paths: List[Path], new_paths: Set[Path]) -> List[Path]:
        return sorted(new_paths) + open_paths


class breadth_first(SearchMethod):
    name = 'Breadth 1st search'
    def search(graph: Graph, open_paths: List[Path], new_paths: Set[Path]) -> List[Path]:
        return open_paths + sorted(new_paths)


class depth_limited_2(SearchMethod):
    name = "Depth-limited search (depth limit 2)"
    def search(graph, open_paths, new_paths):
        return DepthLimited(open_paths, new_paths, 2)

class depth_limited(SearchMethod):
    name = "Depth-limited search"
    def search(open_paths, new_paths, n):
        print("   Not Implemented")
        ...


class iterative_deepening(SearchMethod):
    name = "Iterative deepening search"
    def search(graph, open_paths, new_paths):
        print("   Not Implemented")
        ...


class uniform_cost(SearchMethod):
    name = "Uniform cost search"
    def search(graph, open_paths, new_paths):
        print("   Not Implemented")
        ...


class greedy(SearchMethod):
    name = "Greedy search"
    def search(graph, open_paths, new_paths):
        print("   Not Implemented")
        ...


warnings.warn("A* does not break ties according to the instructions")
class astar(SearchMethod):
    name = "A*"
    def search(graph, open_paths, new_paths):
        all_paths = open_paths + list(new_paths)
        groups = [list(path) for _, path in
                  groupby(all_paths, key=lambda path: path[0])]

        orderby = astar._cost(graph)
        return sorted((min(group, key=orderby) for group in groups), key=orderby)

    @staticmethod
    def _cost(graph):
        def cost(path):
            return graph.heuristic(path[0]) + graph.path_length(path)
        return cost

    def paths_to_str(graph, paths):
        return paths_to_str_cost(paths, astar._cost(graph))

warnings.warn("Hill-climbing does not break ties according to the instructions\n"
              + "\tand we don't know the exact output format they want for hill-climbing")
class hill_climbing(SearchMethod):
    name = "Hill-climbing search"

    def search(graph, open_paths, new_paths):
        if len(new_paths) == 0:
            return []
        else:
            return [max(new_paths, key=graph.heuristic)]

    def paths_to_str(graph, paths):
        return paths_to_str_cost(paths, lambda path: graph.heuristic(path[0]))


class beam_2(SearchMethod):
    name = "Beam search (w=2)"
    def search(graph, open_paths, new_paths):
        return beam(open_paths, new_paths, 2)

class beam(SearchMethod):
    name = "Beam search"
    def search(open_paths, new_paths, n):
        print("   Not Implemented")
        ...


# Mapping of algorithm names to functions
search_methods: List[SearchMethod]
search_methods = [
    depth_first,
    breadth_first,
    astar,
    hill_climbing,
]


# Will run at script execution
if __name__ == '__main__':
    main(search_methods, sys.argv)

