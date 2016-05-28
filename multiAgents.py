
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
import logging
#logging.basicConfig(level=logging.DEBUG)

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """

        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]


    def evaluationFunction(self, currentGameState, action):
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
        # what happening next, t=1
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        
        # dont choose a losing successor
        if successorGameState.isLose():
          return float('-inf')

        # always choose a winning successor
        if successorGameState.isWin():
          return float('inf')

        # if this box is a power capsule, eat it
        if len(currentGameState.getCapsules()) > len(successorGameState.getCapsules()):
          return float('inf')

        #
        #  Compute the distance to closest ghost unafraid ghost
        #
        closestGhostDistance = getClosestGhostDistance(newPos, newGhostStates)
        
        # dont get close to ghosts
        if closestGhostDistance <= 2:
          return float('-inf')

        # if this successor is a pellet, eat away
        if len(currentGameState.getFood().asList()) > len(successorGameState.getFood().asList()):
          return float('inf')

        # dont linger (this jitters, but hey we're looking at successors only and just a few features)
        if successorGameState.getPacmanPosition() == currentGameState.getPacmanPosition():
          return -1

        #
        # Compute the distance to the closest food
        #
        closestFoodDistance = getClosestFoodDistance(newPos, newFood.asList())

        # this could be much better
        score = (closestGhostDistance / closestFoodDistance)

        return score

def getClosestGhostDistance(pos, ghostStates):
  d = getClosestDistance(pos, [ghostState.getPosition() for ghostState in ghostStates if ghostState.scaredTimer < 2])
  if d == 0:
    return 999999
  elif d >= 6:
    return 6
  return d

def getClosestFoodDistance(pos, foodPositions):
  return getClosestDistance(pos, foodPositions)

def getClosestDistance(fromPos, dests):
  distances = [util.manhattanDistance(fromPos, dest) for dest in dests]
  if len(distances) == 0:
    return 0
  closest = distances[0]
  for distance in distances:
      if distance < closest:
        closest = distance
  return closest

def scoreEvaluationFunction(currentGameState):
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
    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.
        """
        x,action = self.minimax(0, gameState, 0, 0)

        logging.debug( 'minimax best action is %s', action )
        logging.debug( '' )
        logging.debug( '' )
        return action

    def minimax(self, agent, state, depth, rDepth):
        """
          Computes the minimax score using DFS
            agent  - the index of the current agent to represent
                     index 0 is always a maximizer, the rest are adversaries
            state  - the current game state to evaluate
            depth  - the current depth of the expansion
                     this depth only increments after all agents have been considered
            rDepth - the recursion depth, to format debug messages

            call this with 0,gameState, 0, 0 to kick things off
        """
        padding = '   ' * rDepth

        # if we are at our depth limit or there are no moves, we are at a leaf node
        # compute and return the score of this state
        if( (depth == self.depth) or (len(state.getLegalActions(agent)) == 0)):
          score = self.evaluationFunction(state)
          logging.debug('%s LEAF: returning score %.1f', padding, score )
          return (score,0)

        optimalScore = float('-inf') if agent == 0 else float('inf')
        optimalAction = Directions.STOP

        for action in state.getLegalActions(agent):
          logging.debug( '%s minimax agent: %d depth: %d action: %s', padding, agent, depth, action )
          successor = state.generateSuccessor(agent, action)
          nextAgent = (agent+1) % state.getNumAgents()
          nextDepth = depth+1 if nextAgent == 0 else depth
          score,x = self.minimax(nextAgent, successor, nextDepth, rDepth+1)
          newOptimalScore = max(score, optimalScore) if agent == 0 else min(score, optimalScore)
          optimalAction = optimalAction if newOptimalScore == optimalScore else action
          optimalScore = newOptimalScore

        logging.debug( '%s #### agent: %d optimalAction: %s optimalScore: %d', padding, agent, optimalAction, optimalScore )
        return (optimalScore,optimalAction)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        x, action = self.alphaBeta(0, gameState, 0, 0, float('-inf'), float('inf'))
        logging.debug( 'alphabeta best action is %s', action )
        logging.debug( '' )
        logging.debug( '' )
        return action

    def alphaBeta(self, agent, state, depth, rDepth, alpha, beta):
        """
          Computes the minimax score using DFS
            agent  - the index of the current agent to represent
                     index 0 is always a maximizer, the rest are adversaries
            state  - the current game state to evaluate
            depth  - the current depth of the expansion
                     this depth only increments after all agents have been considered
            rDepth - the recursion depth, to format debug messages
            alpha  - pacmans best score on path to root
            beta   - ghosts best score on path to root
        """
        padding = '   ' * rDepth

        # if we are at our depth limit or there are no moves, we are at a leaf node
        # compute and return the score of this state
        if( (depth == self.depth) or (len(state.getLegalActions(agent)) == 0)):
          score = self.evaluationFunction(state)
          logging.debug('%s LEAF: returning score %.1f', padding, score )
          return (score,0)

        optimalScore = float('-inf') if agent == 0 else float('inf')
        optimalAction = Directions.STOP

        for action in state.getLegalActions(agent):
          logging.debug( '%s minimax agent: %d depth: %d action: %s', padding, agent, depth, action )
          successor = state.generateSuccessor(agent, action)
          nextAgent = (agent+1) % state.getNumAgents()
          nextDepth = depth+1 if nextAgent == 0 else depth
          score,x = self.alphaBeta(nextAgent, successor, nextDepth, rDepth+1, alpha, beta)
          newOptimalScore = max(score, optimalScore) if agent == 0 else min(score, optimalScore)
          optimalAction = optimalAction if newOptimalScore == optimalScore else action
          optimalScore = newOptimalScore
          if agent == 0:
            if score > beta:
              break
            alpha = max(alpha, score)
          else:
            if score < alpha:
              break
            beta = min(beta, score)

        logging.debug( '%s #### agent: %d optimalAction: %s optimalScore: %d', padding, agent, optimalAction, optimalScore )
        return (optimalScore,optimalAction)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction