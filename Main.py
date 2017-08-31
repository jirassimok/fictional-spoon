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
from collections.abc import Callable as abcCallable
from functools import wraps
from typing import List, Dict, FrozenSet, Callable
import sys


class Graph:
    def __init__(self,
                 edges: Dict[FrozenSet[str], float],
                 heuristics: Dict[FrozenSet[str], float]):
        self.edges = edges
        self.heuristics = heuristics

    @classmethod
    def fromFile(cls, graph_input_txt_path: str) -> 'Graph':
        edges = {}  # type
        heuristics = {}  # type
        with open(graph_input_txt_path) as graph_input_txt_fp:
            stage_heuristic = False
            for line in graph_input_txt_fp:
                line = line[:-1]
                if line == '#####':
                    stage_heuristic = True
                    continue
                elif not line:
                    break
                if stage_heuristic is False:
                    state1, state2, weight = line.split(' ')
                    weight = float(weight)
                    assert weight > 0
                    assert frozenset({state1, state2}) not in edges
                    edges[frozenset({state1, state2})] = weight
                else:
                    state1, heuristic = line.split(' ')
                    heuristic = float(heuristic)
                    state2 = 'G'
                    assert frozenset({state1, state2}) not in heuristics
                    assert heuristic >= 0
                    heuristics[frozenset({state1, state2})] = heuristic
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

    def print_trace_path(self) -> str:
        return self._print_path(self.trace_path())

    def expand(self) -> List['TreeNode']:
        return list(self.__class__(self.graph, state, self) for state in self.graph.get_adjacent_states(self.state))


class Problem:
    def __init__(self, graph_input_txt_path: str, initial_state: str, solution_state: str):
        self.graph = Graph.fromFile(graph_input_txt_path)
        self.solution_state = solution_state
        self.initial_state = initial_state


def General_Search(problem: Problem, search_method: Callable[[List[TreeNode]], List[TreeNode]]):
    print('Expanded\tQueue')

    queue = [TreeNode(problem.graph, problem.initial_state, parent=None)]
    while queue:
        node = queue[0]
        print('{}\t[{}]'.format(node.state, ', '.join(node.print_trace_path() for node in queue)))
        node = queue.pop(0)
        if node.state == problem.solution_state:
            print('goal reached!')
            return node.state
        opened_nodes = node.expand()
        opened_nodes = [opened_node for opened_node in opened_nodes
                        if opened_node.state not in [ancestor_node.state for ancestor_node in node.trace_path()]]
        queue = search_method(queue, opened_nodes)
    print(f'failure to find path between {problem.initial_state} and {problem.solution_state}')
    return None


def breadth_first_search(queue: List[TreeNode], new_nodes_list: List[TreeNode]) -> List[TreeNode]:
    return queue + new_nodes_list

def depth_first_search(queue: List[TreeNode], new_nodes_list: List[TreeNode]) -> List[TreeNode]:
    return new_nodes_list + queue


search_method_name_mapping = {
    'Depth 1st search': depth_first_search,
    'Breadth 1st search': breadth_first_search,
}


def main():
    args = sys.argv[1:]
    assert(len(args) == 1), f'Must have exactly 1 argument. Arguments detected: {args}'
    name = args[0]
    assert name in search_method_name_mapping, f'Invalid method name "{name}"'
    first_problem = Problem(name, 'S', 'G')
    print(name, end='\n')
    General_Search(first_problem, 'Depth 1st search')


# Will run at script execution
if __name__ == '__main__':
    main()
