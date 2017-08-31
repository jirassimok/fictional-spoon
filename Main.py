"""
Project: SearchAlgorithms
File: Main
Created on: 8/28/2017
  
Project Description:
CS4341 - Project 1
Search algorithms

A project in which several search algorithms are implemented.
"""
from collections import OrderedDict
from functools import update_wrapper, wraps
import re
from typing import NamedTuple, List, Dict, FrozenSet, Callable, Iterable, MutableMapping
import sys

class Graph:
    def __init__(self,
                 edges: Dict[FrozenSet[str], float],
                 heuristics: Dict[FrozenSet[str], float]):
        self.edges = edges
        self.heuristics = heuristics

    @classmethod
    def from_lines(cls, lines: Iterable[str]) -> 'Graph':
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
        return cls(edges, heuristics)

    def get_adjacent_states(self, state: str):
        return [(set(key) - {state}).pop() for key in self.edges.keys() if state in key]

    def get_heuristic(self, state1: str, state2: str):
        return self.heuristics[frozenset({state1, state2})]

    def get_weight(self, state1: str, state2: str):
        return self.edges[frozenset({state1, state2})]

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
        return list(self.__class__(self.graph, state, self) for state in self.graph.get_adjacent_states(self.state))

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

def main(search_methods: Dict[str, SearchMethod]):
    args = sys.argv[1:]
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
def register(name: str):
    """Register a search method with a name
    """
    def decorator(function):
        global search_methods
        search_methods[name] = function
        return function
    return decorator

@register('Breadth 1st search')
def breadth_first_search(queue: List[TreeNode], new_nodes_list: List[TreeNode]) -> List[TreeNode]:
    return queue + new_nodes_list

@register('Depth 1st search')
def depth_first_search(queue: List[TreeNode], new_nodes_list: List[TreeNode]) -> List[TreeNode]:
    return new_nodes_list + queue


@register("Depth-limited search (depth limit 2)")
def depth_limited_2(open_nodes, new_nodes):
    return depth_limited(2, open_nodes, new_nodes)
def depth_limited(n, open_nodes, new_nodes):
    print("   Not Implemented")
    ...

@register("Iterative deepening search")
def iterative_deepening(open_nodes, new_nodes):
    print("   Not Implemented")
    ...

@register("Uniform cost search")
def uniform_cost(open_nodes, new_nodes):
    print("   Not Implemented")
    ...

@register("Greedy search")
def greedy(open_nodes, new_nodes):
    print("   Not Implemented")
    ...

@register("A*")
def astar(open_nodes, new_nodes):
    print("   Not Implemented")
    ...

@register("Hill-climbing search")
def hill_climbing(open_nodes, new_nodes):
    print("   Not Implemented")
    ...

@register("Beam search (w=2)")
def beam_2(open_nodes, new_nodes):
    return beam(2, open_nodes, new_nodes)
def beam(n, open_nodes, new_nodes):
    print("   Not Implemented")
    ...

# Will run at script execution
if __name__ == '__main__':
    main(search_methods)
