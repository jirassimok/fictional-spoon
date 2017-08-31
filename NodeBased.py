"""
File: NodeBased.py
Date: 2017-08-30

Search algorithms with a list history
"""
from functools import partial
from itertools import chain
from operator import eq
from typing import NamedTuple, Any, Generic, TypeVar, Callable, Iterable
import re

T = TypeVar('T')

class Node(Generic[T]):
    """A node's name must be a unique identifier within its graph
    """
    def __init__(self,
                 name: T,
                 heuristic:float,
                 edges: {'Node': float} = None):
        if edges == None:
            edges = dict()

        for node in edges.keys():
            if not isinstance(node, self.__class__):
                raise TypeError("Node neighbors must be Nodes")

        self._name = name
        self._heuristic = heuristic
        self._edges = dict(edges)

        def _add_edge(name, value):
            self._edges[name] = value
        self.add_edge = _add_edge

    def freeze(self):
        """Make this node immmutable
        """
        del self.add_edge

    @property
    def name(self):
        return self._name
    @property
    def heuristic(self):
        return self._heuristic
    @property
    def edges(self):
        return dict(self._edges)
    @property
    def neighbors(self):
        return frozenset(self._edges.keys())

    def __eq__(self, other):
        return type(other) == self.__class__ and self.name == other.name

    def __hash__(self):
        return hash((self.__class__, self.name))

    def __str__(self):
        return f"Node({self.name!r})"

    def __name_repr(self):
        return repr(self.name)

    def __repr__(self):
        neighbors = ", ".join(":".join((str(k.name), str(v)))
                              for k, v in self._edges.items())
        return f'Node({self.name!r}, {self.heuristic!r}, {{{neighbors}}})'


class Graph(Generic[T]):
    """An undirected graph
    """
    def __init__(self, nodes: {Node}, edges: {(frozenset({Node, Node}), float)}, heuristics: {Node: float}):
        self._nodes = frozenset(nodes)
        self._edges = frozenset(edges)
        self._heuristics = frozenset(heuristics.items())
        print(heuristics)
        print(self._heuristics)

    @property
    def nodes(self):
        return self._nodes
    @property
    def edges(self):
        return self._edges
    @property
    def heuristics(self):
        return dict(self._heuristics)

    # TODO: Using slices for searching ALMOST makes sense, but should be removed.
    def __getitem__(self, key):
        return {node.name: node for node in self._nodes}[key]

    @classmethod
    def from_edges(cls, edges: {frozenset({T, T}): float}, heuristics: {T: float}):
        edge_names = {name for edge in edges for name in edge}
        all_names = edge_names.union(heuristics.keys())
        no_heuristic_names = edge_names.difference(heuristics.keys())

        if len(no_heuristic_names) != 1:
            raise ValueError("More than one node has no heuristic value")

        # TODO: Do we want this to be 0, None, or something else?
        nodes = {name: Node(name, heuristics.get(name, 0)) for name in all_names}
        new_heuristics = {nodes[name]: h for name, h in heuristics.items()}

        new_edges = set()
        for ends, length in edges.items():
            a, b = ends
            new_ends = frozenset((nodes[a], nodes[b]))
            new_edges.add((new_ends, length))

        for node in nodes.values():
            for ends, length in new_edges:
                a, b = ends
                if a == node:
                    node.add_edge(b, length)
                elif b == node:
                    node.add_edge(a, length)
            node.freeze()

        return cls(nodes.values(), new_edges, new_heuristics)

def read_lines(lines: Iterable[str]) -> Graph:
    edges, edge_regex = [], r'([A-Z])\s+([A-Z])\s+(\S+)\s+'
    heuristics, heuristic_regex = [], r'([A-Z])\s+(\S+)\s+'

    section, line_regex = edges, edge_regex
    for line in lines:
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
    edges = dict(map(lambda x: (frozenset(x[:2]), x[2]), edges))
    heuristics = dict(heuristics)
    return Graph.from_edges(edges, heuristics)

class Problem(NamedTuple):
    graph: Graph
    start: Node
    goal: Node

def general_search(problem: Problem, search_method):
    print("   Expanded  Queue")
    paths = [(problem.start,)]
    while paths:
        state = paths.pop()
        current = state[-1]
#        print(paths, state, current, current.neighbors)
        if current == problem.goal:
            print("   goal reached!")
            return state
        else:
            print(f"{current.name}      ", end='')
            found = {state+(new,) for new in current.neighbors if new not in state}
            print(found)
            paths = search_method(paths, found)
    print(f"   failure to find a path between {problem.start.name}"
           " and {problem.goal.name}", sep='')
    return None

def dfs(paths, state, new_nodes=1):
    print(paths, flush=True)
    inner = '> <'.join([','.join([str(node.name) for node in path])
                        for path in chain([state], paths)])
    print(f'[<{inner}>]')
    #paths.extend(new_nodes)
    paths.extend(state)
    return paths

def main(filename, algorithm):
    with open(filename, 'r') as f:
        for line in f:
            ...
