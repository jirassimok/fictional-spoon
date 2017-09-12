"""
Project: SearchAlgorithms
File: Main
Created on: 8/28/2017

Project Description:
CS4341 - Project 1
Search algorithms

A project in which several search algorithms are implemented.
"""
from abc import ABC, abstractmethod
import re
import itertools
from typing import NamedTuple, List, Dict, Set, FrozenSet, Callable, Iterable, MutableMapping, Union, Tuple, Optional
from types import FunctionType
import sys

Path = Union[Tuple[str], List[str]]

# Forward references for as-yet-undefined types
SearchMethod, SearchAlgorithm = 'SearchMethod', 'SearchAlgorithm'
Problem, Graph = 'Problem', 'Graph'


def main(search_methods: List[Union[SearchMethod, SearchAlgorithm]], argv):
    args = argv[1:]
    if len(args) != 1:
        print(f'Must have exactly 1 argument. Arguments detected: {args}')
        sys.exit(1)

    with open(args[0], 'r') as f:
        graph = Graph.from_lines(f)

    problem = Problem(graph, 'S', 'G')

    success = '      goal reached!'
    failure = f'   failure to find path between {problem.start} and {problem.goal}'

    for method in search_methods:
        print()
        print(method.name, '   Expanded  Queue', sep='\n\n')
        if isinstance(method, SearchMethod):
            result = General_Search(problem, method)
        elif isinstance(method, SearchAlgorithm):
            result = method(problem)
        print(failure if result is None else success, end='\n\n')

def General_Search(problem: Problem, search_method: SearchMethod) -> Optional[Path]:
    graph = problem.graph
    start = problem.start
    goal = problem.goal
    paths = [(start,)]

    while paths:
        expanded = paths[0][0]
        print(f'      {expanded}      ', search_method.paths_to_str(graph, paths), sep='')
        path = paths.pop(0)

        if expanded == goal:
            return path

        opened_paths = {(n,)+path for n in graph.neighbors(expanded) if n not in path}

        paths = search_method(graph, paths, opened_paths)

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
        return {n for edge in self.edges for n in edge if node in edge} - {node}

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


class SearchMethod(type):
    """Metaclass for SearchMethod

    Allows calling SearchMethods like functions.
    Prevents modification of name and search method.
    Provides convenient string representation.
    Automatically makes all methods static, except already-static methods.
    """
    def __call__(cls,
                 graph: Graph,
                 open_paths: List[Path],
                 new_paths: List[Path]) -> List[Path]:
        return cls.search(graph, open_paths, new_paths)

    def __setattr__(cls, key, value):
        if key == "name" or key == "search":
            raise AttributeError("can't set SearchMethod name or search method")
        else:
            object.__setattr__(cls, key, value)

    def __str__(cls):
        return f"SearchMethod({cls.name})"

    def __new__(cls, name, bases, attrs):
        copy = ((k, v) for k, v in attrs.items())
        for name, value in copy:
            if not isinstance(value, staticmethod) and isinstance(value, FunctionType):
                attrs[name] = staticmethod(value)
        return type.__new__(cls, name, bases, attrs)


class SearchMethodBase(object, metaclass=SearchMethod):
    """Base class that other SearchMethods should inherit from and use as a template"""

    name = None

    def search(graph: Graph, open_paths: List[Path], new_paths: Set[Path]) -> List[Path]:
        raise TypeError("Can not call abstract search method")

    def paths_to_str(graph: Graph, paths: Iterable[Path]) -> None:
        """Print a path in a nice format.

        The default implementation does not print costs.
        """
        return ''.join(('[<', '> <'.join(','.join(path) for path in paths), '>]'))


class SearchAlgorithm(object):
    def __init__(self, name, method):
            self._name = name
            self._method = method

    def __call__(self, *args, **kwargs):
        return self.method(*args, **kwargs)

    @property
    def name(self):
        return self._name

    @property
    def method(self):
        return self._method


def algorithm(name):
    def wrapper(function):
        return SearchAlgorithm(name, function)
    return wrapper


"""
HELPER METHODS
"""


def paths_to_str_cost(paths: List[Path], cost_calculator: Callable[[Path], float]) -> str:
    """Convert a list of paths to a string, using the given cost calculation
    function for each path.
    """
    output = ['[']
    for path in paths:
        cost = cost_calculator(path)
        output.extend((str(cost), '<', ','.join(path), '> '))
    output[-1] = '>]'
    return ''.join(output)


def groupby(iterable, key=None):
    """Version of itertools.groupby that sorts by the key first
    """
    if key is None:
        key = lambda x: x
    return itertools.groupby(sorted(iterable, key=key), key=key)


class costs(object):
    """Utility methods for estimating costs
    """
    @staticmethod
    def heuristic(graph: GraphABC, path: Path=None):
        def cost(path: Path):
            return graph.heuristic(path[0])
        if path is None:
            return cost
        else:
            return cost(path)

    @staticmethod
    def path_cost(graph: GraphABC, path: Path=None):
        def cost(path: Path):
            return graph.path_length(path)
        if path is None:
            return cost
        else:
            return cost(path)


class paths(object):
    """Utility methods for sorting paths
    """
    @staticmethod
    def _key(base_key):
        if base_key is None:
            def realkey(path):
                return (path[0], len(path), tuple(path))
        else:
            def realkey(path):
                return (base_key(path), path[0], len(path), tuple(path))
        return realkey
    @staticmethod
    def sorted(paths_, *, key=None):
        return sorted(paths_, key=paths._key(key))
    @staticmethod
    def min(paths_, *, key=None):
        return min(paths_, key=paths._key(key))


def paths_to_str_heuristic(graph: Graph, paths: List[Path]):
    return paths_to_str_cost(paths, costs.heuristic(graph))

def paths_to_str_path_cost(graph: Graph, paths: List[Path]):
    return paths_to_str_cost(paths, costs.path_cost(graph))

"""
SEARCH METHOD IMPLEMENTATIONS
"""


class depth_first(SearchMethodBase):
    name = 'Depth 1st search'

    def search(graph: Graph, open_paths: List[Path], new_paths: Set[Path]) -> List[Path]:
        return sorted(new_paths) + open_paths


class breadth_first(SearchMethodBase):
    name = 'Breadth 1st search'

    def search(graph: Graph, open_paths: List[Path], new_paths: Set[Path]) -> List[Path]:
        return open_paths + sorted(new_paths)


def depth_limited(limit: int) -> SearchMethod:
    """Method to construct depth-limited search instances of SearchMethod"""

    class depth_limited_n(SearchMethodBase):
        name = f"Depth-limited search (depth-limit = {limit})"

        def search(graph, open_paths, new_paths):
            valid = lambda path: len(path)-1 <= limit
            return sorted(filter(valid, new_paths)) + open_paths
    return depth_limited_n


@algorithm("Iterative deepening search")
def iterative_deepening_search(problem: Problem) -> Optional[Path]:
    graph = problem.graph
    start = problem.start
    goal = problem.goal

    for i in itertools.count(): # for i in range(0, infinity)
        print(f"L={i}")
        result = General_Search(problem, depth_limited(i))
        if result is not None:
            return result


class uniform_cost(SearchMethodBase):
    name = "Uniform Search (Branch-and-bound)"
    paths_to_str = paths_to_str_path_cost

    def search(graph, open_paths, new_paths):
        all_paths = paths.sorted(new_paths.union(open_paths), key=costs.path_cost(graph))
        return all_paths


class greedy(SearchMethodBase):
    name = "Greedy search"
    paths_to_str = paths_to_str_heuristic

    def search(graph, open_paths, new_paths):
        all_paths = paths.sorted(new_paths.union(open_paths), key=costs.heuristic(graph))
        return all_paths

class astar(SearchMethodBase):
    name = "A*"

    def search(graph, open_paths, new_paths):
        all_paths = open_paths + list(new_paths)
        groups = [list(path) for _, path in
                  groupby(all_paths, key=lambda path: path[0])]

        orderby = astar.cost(graph)
        return paths.sorted((paths.min(group, key=orderby) for group in groups), key=orderby)

        # # Instead of group on unexplored node then take shortest,
        # # First filter out new_paths that have unexplored node that unexplored node in open_paths, then do union
        # all_paths = open_paths + [
        #     new_path for new_path in new_paths if new_path[0] not in (open_path[0] for open_path in open_paths)]
        # return paths.sorted(all_paths, key=astar.cost(graph))
        # # Long form:
        # news = []
        # frontier = [open_path[0] for open_path in open_paths]
        # for new in new_paths:
        #     if new[0] not in frontier:
        #         news.append(new)
        # return paths.sorted(open_paths + news, key=astar.cost(graph))


    def cost(graph):
        def cost(path):
            return graph.heuristic(path[0]) + graph.path_length(path)
        return cost

    def paths_to_str(graph, paths):
        return paths_to_str_cost(paths, astar.cost(graph))


class hill_climbing(SearchMethodBase):
    name = "Hill-climbing search"
    paths_to_str = paths_to_str_heuristic

    def search(graph, open_paths, new_paths):
        if len(new_paths) == 0:
            return []
        else:
            return [paths.min(new_paths, key=costs.heuristic(graph))]


def beam(limit: int) -> SearchMethod:
    """Method to construct beam search instances of SearchMethod"""
    class beam_k(SearchMethodBase):
        name = f"Beam search (w = {limit})"
        paths_to_str = paths_to_str_heuristic

        def search(graph, open_paths, new_paths):
            paths_by_length = [list(group) for _, group in groupby(open_paths + list(new_paths), key=len)]
            return list(itertools.chain.from_iterable(
                sorted(sorted(paths, key=costs.heuristic(graph))[:limit]) for paths in paths_by_length)
            )

    return beam_k


# Mapping of algorithm names to functions
search_methods: List[Union[SearchMethod, SearchAlgorithm]]
search_methods = [
    depth_first,
    breadth_first,
    depth_limited(2),
    iterative_deepening_search,
    uniform_cost,
    greedy,
    astar,
    hill_climbing,
    beam(2),
]


# Will run at script execution
if __name__ == '__main__':
    main(search_methods, sys.argv)


"""
Testing methods
"""


def get_graph(filename):
    with open(filename, 'r') as f:
        return Graph.from_lines(f)


def make_problem(filename, start='S', goal='G'):
    with open(filename, 'r') as f:
        graph = Graph.from_lines(f)
    return Problem(graph, start, goal)
