# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions
from typing import List

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()




def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    start_state = problem.getStartState()
    frontier = util.Stack() # Frontier being a Stack
    frontier.push((start_state, [])) # Frontier consisting of start state and path (which is an empty list for start)
    reached_states = set() # An empty set to which reached states are added
    while not frontier.isEmpty():
        state, path = frontier.pop() # Select and remove the most recently pushed state and path from the frontier
        if problem.isGoalState(state):
            return path
        if state not in reached_states:
            reached_states.add(state) # Add state to reached states to prevent loops
            for child_state, move, _ in problem.getSuccessors(state):
                child_path = path + [move] # Add the next move to reach the child state to the 'child path'
                frontier.push((child_state, child_path)) # Push child state and child path onto the stack of frontier
    return [] # Return an empty path if no solution is found

def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
    start_state = problem.getStartState()
    frontier = util.Queue() # Frontier being a Queue
    frontier.push((start_state, [])) # Frontier consisting of start state and path (which is an empty list for start)
    reached_states = set() # An empty set to which reached states are added
    while not frontier.isEmpty():
        state, path = frontier.pop() # Select and remove the earliest enqueued state and path still in the frontier
        if problem.isGoalState(state):
            return path
        if state not in reached_states:
            reached_states.add(state) # Add state to reached states to prevent loops
            for child_state, move, _ in problem.getSuccessors(state):
                child_path = path + [move] # Add the next move to reach the child state to the 'child path'
                frontier.push((child_state, child_path)) # Enqueue child state and child path into the queue of frontier
    return [] # Return an empty path if no solution is found

def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    start_state = problem.getStartState()
    frontier = util.PriorityQueue() # Frontier being a Priority Queue
    # Frontier consisting of start state and path (which is an empty list for start) and priority value of 0
    frontier.push((start_state, []),0)
    reached_states = set() # An empty set to which reached states are added
    while not frontier.isEmpty():
        # Select and remove state and path with the highest priority (with the smallest path cost) in the frontier
        state, path = frontier.pop()
        if problem.isGoalState(state):
            return path
        if state not in reached_states:
            reached_states.add(state) # Add state to reached states to prevent loops
            for child_state, move, _ in problem.getSuccessors(state):
                child_path = path + [move] # Add the next move to reach the child state to the 'child path'
                child_cost = problem.getCostOfActions(child_path) # Child cost is child path cost, child backward cost
                # A priority value is path cost for UCS. Higher priority is for lower value.
                frontier.update((child_state, child_path), child_cost)
    return [] # Return an empty path if no solution is found

def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    start_state = problem.getStartState()
    frontier = util.PriorityQueue() # Frontier being a Priority Queue
    # Frontier consisting of start state and path (which is an empty list for start) and priority value of 0
    frontier.push((start_state, []),0)
    reached_states_with_costs = {} # An empty dictionary to which reached states with their costs are added
    while not frontier.isEmpty():
        # Select and remove state and path with the highest priority (the smallest sum of backward and forward costs)
        state, path = frontier.pop()
        if problem.isGoalState(state):
            return path
        # If we reach the state already reached but with a lower path cost than the same state previously reached,
        # it should be revisited and explored further by updating it
        if state not in reached_states_with_costs or problem.getCostOfActions(path)<reached_states_with_costs[state]:
            # Add the state with its path cost to the dictionary to prevent loops and track costs
            reached_states_with_costs[state] = problem.getCostOfActions(path)
            for child_state, move, _ in problem.getSuccessors(state):
                child_path = path + [move] # Add the next move to reach the child state to the 'child path'
                child_backward_cost = problem.getCostOfActions(child_path) # Child path cost, child backward cost
                child_forward_cost = heuristic(child_state, problem) # Goal proximity of child state, child forward cost
                child_cost = child_backward_cost + child_forward_cost # Child cost is sum of backward and forward costs
                # A priority value is the sum of backward and forward costs for A*. Higher priority is for lower value.
                frontier.update((child_state, child_path), child_cost)
    return [] # Return an empty path if no solution is found

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
