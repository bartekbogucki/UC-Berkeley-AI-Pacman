# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        remainingFoods = newFood.asList()
        newGhostPos = successorGameState.getGhostPositions()

        # Get distance to the closest ghost
        ghost_dis = {}
        close_ghost = 0
        if newGhostPos:
            for newGhost in newGhostPos:
                ghost_dis[newGhost] = abs(newPos[0] - newGhost[0]) + abs(newPos[1] - newGhost[1])
            closest_ghost = min(ghost_dis, key=ghost_dis.get)
            # A minimum of ghost distance and 4 is taken as we have to worry about the closest ghost,
            # which is really close. So if it is further away, we take 4. So if it is further away,
            # we always just take 4 without differentiate it, even if we are indeed closer.
            close_ghost += min(ghost_dis[closest_ghost], 4)

        # Get distance to the closest food
        food_dis = {}
        close_food = 0
        if remainingFoods:
            for remainingFood in remainingFoods:
                food_dis[remainingFood] = abs(newPos[0] - remainingFood[0]) + abs(newPos[1] - remainingFood[1])
            closest_food = min(food_dis, key=food_dis.get)
            # A minimum of 9 points and food distance is taken as it should be small enough that it is better to
            # eat a dot and score 10 points than not eating it, as other dots are further away than 9.
            # It can result in a random walk or stop that ends when we get closer to the dot or closer to the ghost,
            # when it starts to differentiate.
            close_food = min(food_dis[closest_food], 9)

        # Score function is different depending on whether it is during the scared time or not to chase ghosts if it is.
        # A minimum of scared times is taken, so if we have already eaten one ghost, we will stop chasing others -
        # so as not to chase un-scared ghosts.
        if not newGhostPos or min(newScaredTimes) == 0:
            # Distance to the food is subtracted, so closer to the food is better (as higher the score).
            # Distance to the ghost is added, so closer to the ghost the worse (as lower the score).
            # It is penalised more for being close to the ghost (11) than for gaining points by eating a dot (10).
            score = successorGameState.getScore() - close_food + 11 * close_ghost
        else:
            # Arbitrarily added 200 points to be better in Scared Timer, also low enough that it is better to eat the
            # ghost and get 500 points or finish a game and get 1000 points than not turn off the timer.
            # To chase ghosts the distance to the ghost is subtracted, so the closer to the ghost the better.
            score = successorGameState.getScore() - 11 * close_ghost + 200

        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimax_value(state, depth, agent):
            # In case of maximum depth or terminal state already reached return utility.
            if self.depth == depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            # Get next agent and its depth:
            nextAgent = (agent + 1) % state.getNumAgents()
            if nextAgent == 0:
                nextDepth = depth + 1
            else:
                nextDepth = depth

            # If agent is Pacman then maximum:
            if agent == 0:
                # Initialize value to be updated
                value = float('-inf')
                for action in state.getLegalActions(agent):
                    # Recursively refer to next agent and its depth within legal successors of the current agent's state
                    value = max(value, minimax_value(state.generateSuccessor(agent, action), nextDepth, nextAgent))
                return value
            # If agent is ghost then minimum:
            else:
                value = float('inf')
                for action in state.getLegalActions(agent):
                    value = min(value, minimax_value(state.generateSuccessor(agent, action), nextDepth, nextAgent))
                return value

        # The algorithm is initialized for agent 0 of Pacman and the depth of 0 (root),
        # for which state the action is actually taken.
        agent = 0
        depth = 0

        # Get next Agent and next depth. It will always be 1 and 0 respectively here in case of any number of ghosts,
        # but in case of no ghost the algorithm would work as 'maxAgent' always using maximum possible value
        # (where 0 and 1 would be next Agent and next depth here).
        nextAgent = (agent + 1) % gameState.getNumAgents()
        if nextAgent == 0:
            nextDepth = depth + 1
        else:
            nextDepth = depth

        # Calculate for Pacman the maximum value of the minimax value of his possible actions at root.
        # Initialize maximum value to be updated
        maxValue = float('-inf')
        for action in gameState.getLegalActions(agent):
            minimaxValue = minimax_value(gameState.generateSuccessor(agent, action), nextDepth, nextAgent)
            if minimaxValue > maxValue:
                # Update a maximum value and corresponding action for Pacman:
                maxAction = action
                maxValue = minimaxValue

        # Return the action of the maximum minimax value
        return maxAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def minimax_value_with_pruning(state, depth, agent, alpha, beta):
            if self.depth == depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            nextAgent = (agent + 1) % state.getNumAgents()
            if nextAgent == 0:
                nextDepth = depth + 1
            else:
                nextDepth = depth

            if agent == 0:
                value = float('-inf')
                for action in state.getLegalActions(agent):
                    value = max(value, minimax_value_with_pruning(state.generateSuccessor(agent, action), nextDepth,
                                                                  nextAgent, alpha, beta))
                    # If the value is higher than beta, no further actions will be explored,
                    # as ghosts wouldn't choose such a state given beta.
                    if value > beta:
                        return value
                    # Update alpha to correspond to the best value the Pacman can guarantee so far (highest value)
                    alpha = max(alpha, value)
                return value
            else:
                value = float('inf')
                for action in state.getLegalActions(agent):
                    value = min(value, minimax_value_with_pruning(state.generateSuccessor(agent, action), nextDepth,
                                                                  nextAgent, alpha, beta))
                    # If value is lower than alpha, no further actions will be explored,
                    # as Pacman wouldn't choose such a state given alpha.
                    if value < alpha:
                        return value
                    # Update beta to correspond to the best value the ghost can guarantee so far (lowest value)
                    beta = min(beta, value)
                return value

        agent = 0
        depth = 0

        nextAgent = (agent + 1) % gameState.getNumAgents()
        if nextAgent == 0:
            nextDepth = depth + 1
        else:
            nextDepth = depth

        maxValue = float('-inf')
        # Initialize alpha and beta to be updated during the algorithm
        alpha = float('-inf')
        beta = float('inf')
        for action in gameState.getLegalActions(agent):
            minimaxValue = minimax_value_with_pruning(gameState.generateSuccessor(agent, action), nextDepth, nextAgent,
                                                      alpha, beta)
            if minimaxValue > maxValue:
                maxAction = action
                maxValue = minimaxValue
                # Update alpha to correspond to the maximum value for Pacman at root
                alpha = minimaxValue

        return maxAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def expectimax_value(state, depth, agent):
            if self.depth == depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            nextAgent = (agent + 1) % state.getNumAgents()
            if nextAgent == 0:
                nextDepth = depth + 1
            else:
                nextDepth = depth

            if agent == 0:
                value = float('-inf')
                for action in state.getLegalActions(agent):
                    value = max(value, expectimax_value(state.generateSuccessor(agent, action), nextDepth, nextAgent))
                return value
            # Instead of minimum, expected value is calculated in expectimax algorithm
            else:
                # Initialize value as 0
                value = 0
                for action in state.getLegalActions(agent):
                    # Probability of given successor is the same for each possible successor
                    p = 1 / len(state.getLegalActions(agent))
                    # Add up expected value of the successors to the value
                    value += p * expectimax_value(state.generateSuccessor(agent, action), nextDepth, nextAgent)
                return value

        agent = 0
        depth = 0

        nextAgent = (agent + 1) % gameState.getNumAgents()
        if nextAgent == 0:
            nextDepth = depth + 1
        else:
            nextDepth = depth

        # Here, it is the same as for minimax
        maxValue = float('-inf')
        for action in gameState.getLegalActions(agent):
            minimaxValue = expectimax_value(gameState.generateSuccessor(agent, action), nextDepth, nextAgent)
            if minimaxValue > maxValue:
                maxAction = action
                maxValue = minimaxValue

        return maxAction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    The betterEvaluationFunction function takes in the current GameStates and returns a number,
    where higher numbers are better.

    Similar to the scoring function from Q1, score function is different depending on whether it is during the scared
    time or not to chase ghosts if it is. Here we differentiate based on the minimum number of scared times taken,
    so if we have already eaten one ghost, we will stop chasing others - so as not to chase un-scared ghosts.
    However, compared to Q1, we use current states rather than successor states generated from current state actions.

    If both ghosts or one ghost is not scared, then the distance to the closest food and the closest ghost are used:
    - Distance to the food is subtracted, so closer to the food is better (as higher the score). The distance is
    calculated using a minimum of food distance and 9 as it should be small enough that it is better to eat a dot and
    score 10 points than not eating it, i.e. the remaining dots are further away than 9. It can result in a random walk
    or stop that ends when we get closer to the dot or closer to the ghost, when it starts to differentiate.
    - Distance to the ghost is added, so closer to the ghost the worse (as lower the score). The distance is calculated
    using a minimum of ghost distance and 4 as we have to worry about the closest ghost, which is really close. So if
    it is further away, we always just take 4 without differentiate it, even if we are indeed closer. In the function is
    also penalised more for being closer to the ghost (11) than for gaining points by eating a dot (10).

    If both ghosts are scared (min(currScaredTimes)!=0) we hunt ghosts:
    - Distance to the closest ghost is subtracted, so the closer to the ghost the better (as higher the score). It is
    also rewarded more for being closer to the ghost (11) than for gaining points by eating a dot (10).
    - Distance to the food is not used.
    - Also, arbitrarily, 200 points is added to make it better to be in the Scared Timer than not, as we can gain a lot
    by being in it and eating a scared ghost. It is also low enough that it is better to eat the ghost and get 500
    points, or finish the game and get 1000 points, than not to do it to keep the timer from going off and losing
    arbitrarily added 200 points.
    """
    "*** YOUR CODE HERE ***"
    currPos = currentGameState.getPacmanPosition()
    currFood = currentGameState.getFood()
    currGhostStates = currentGameState.getGhostStates()
    currScaredTimes = [ghostState.scaredTimer for ghostState in currGhostStates]
    remainingFoods = currFood.asList()
    currGhostPos = currentGameState.getGhostPositions()

    ghost_dis = {}
    close_ghost = 0
    if currGhostPos:
        for currGhost in currGhostPos:
            ghost_dis[currGhost] = abs(currPos[0] - currGhost[0]) + abs(currPos[1] - currGhost[1])
        closest_ghost = min(ghost_dis, key=ghost_dis.get)
        close_ghost += min(ghost_dis[closest_ghost], 4)

    food_dis = {}
    close_food = 0
    if remainingFoods:
        for remainingFood in remainingFoods:
            food_dis[remainingFood] = abs(currPos[0] - remainingFood[0]) + abs(currPos[1] - remainingFood[1])
        closest_food = min(food_dis, key=food_dis.get)
        close_food = min(food_dis[closest_food], 9)

    if not currGhostPos or min(currScaredTimes) == 0:
        score = currentGameState.getScore() - close_food + 11 * close_ghost
    else:
        score = currentGameState.getScore() - 11 * close_ghost + 200

    return score

# Abbreviation
better = betterEvaluationFunction
