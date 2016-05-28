
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
from game import Actions
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
          logging.debug( '%s alphabeta agent: %d depth: %d action: %s', padding, agent, depth, action )
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
        x,action = self.expectimax(0, gameState, 0, 0)

        logging.debug( 'expectimax best action is %s', action )
        logging.debug( '' )
        logging.debug( '' )
        return action

    def expectimax(self, agent, state, depth, rDepth):
        """
          Computes the expectimax score using DFS
            agent  - the index of the current agent to represent
                     index 0 is always a maximizer, the rest are adversaries
            state  - the current game state to evaluate
            depth  - the current depth of the expansion
                     this depth only increments after all agents have been considered
            rDepth - the recursion depth, to format debug messages

            call this with 0,gameState, 0, 0 to kick things off
        """
        padding = '   ' * rDepth
        numActions = len(state.getLegalActions(agent))

        # if we are at our depth limit or there are no moves, we are at a leaf node
        # compute and return the score of this state
        if( (depth == self.depth) or (numActions == 0)):
          score = self.evaluationFunction(state)
          logging.debug('%s LEAF: returning score %.1f', padding, score )
          return (score,0)

        optimalScore = float('-inf') if agent == 0 else float('inf')
        optimalAction = Directions.STOP
        p = 1.0 / numActions
        expectValue = 0

        for action in state.getLegalActions(agent):
          logging.debug( '%s expectimax agent: %d depth: %d action: %s', padding, agent, depth, action )
          successor = state.generateSuccessor(agent, action)
          nextAgent = (agent+1) % state.getNumAgents()
          nextDepth = depth+1 if nextAgent == 0 else depth
          score,x = self.expectimax(nextAgent, successor, nextDepth, rDepth+1)
          if agent == 0:
            newOptimalScore = max(score, optimalScore)  
            optimalAction = optimalAction if newOptimalScore == optimalScore else action
            optimalScore = newOptimalScore
          else:
            expectValue += ( p * score )

        if agent == 0:
          logging.debug( '%s #### agent: %d optimalAction: %s optimalScore: %d', padding, agent, optimalAction, optimalScore )
          return (optimalScore,optimalAction)
        else:
          logging.debug( '%s #### agent: %d expectValue: %.1f', padding, agent, expectValue )
          return (expectValue, 0)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    score = currentGameState.getScore()
    capsules = currentGameState.getCapsules()

    ghostDistance = 0
    for ghost in newGhostStates:
      ghostDistance += manhattanDistance(newPos, ghost.getPosition())

    capsuleDistance = 0
    if len(capsules) > 0 and max(newScaredTimes) > 25:
      if ghostDistance < 2:
        return -1000000000
      else:
        closestCapsule = 10000
        for capsule in capsules:
          capsuleDistance += mazeDistance(capsule, newPos, currentGameState)
          if capsuleDistance < closestCapsule:
            closestCapsule = capsuleDistance
    else:
      capsuleDistance = 10000000000000000

    foodDistance = 0
    closestFood = (1234, 5678)
    for x in range(newFood.width):
      for y in range(newFood.height):
        if newFood[x][y]:
          distance = manhattanDistance(newPos, (x, y))
          foodDistance += distance
          if distance < manhattanDistance(closestFood, newPos):
            closestFood = (x, y)
    if closestFood != (1234, 5678):
      closestFood = mazeDistance(closestFood, newPos, currentGameState)
      
    if ghostDistance < 2:
      return -100000000000
    elif foodDistance == 0:
      return 100000000 * score
    if foodDistance == 2:
      return 1000000 * score
    elif foodDistance == 1:
      return 10000000 * score

    value = 0
    value += - foodDistance
    value += - 10*closestFood**2
    value += - 10/ghostDistance**2
    value += score**3
    value += 100000000 / (1 + capsuleDistance)
    return value

# Abbreviation
better = betterEvaluationFunction
class Node():
  """
  A container storing the current state of a node, the list 
  of  directions that need to be followed from the start state to
  get to the current state and the specific problem in which the
  node will be used.
  """
  def __init__(self, state, path, cost=0, heuristic=0, problem=None):
    self.state = state
    self.path = path
    self.cost = cost
    self.heuristic = heuristic
    self.problem = problem

  def __str__(self):
    string = "Current State: "
    string += __str__(self.state)
    string += "\n"
    string == "Path: " + self.path + "\n"
    return string

  def getSuccessors(self, heuristicFunction=None):
    children = [] 
    for successor in self.problem.getSuccessors(self.state):
      state = successor[0]
      path = list(self.path)
      path.append(successor[1])
      cost = self.cost + successor[2]
      if heuristicFunction:
        heuristic = heuristicFunction(state, self.problem)
      else:
        heuristic = 0
      node = Node(state, path, cost, heuristic, self.problem)
      children.append(node)
    return children
def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    
    closed = set()
    fringe = util.Queue()

    startNode = Node(problem.getStartState(), [], 0, 0, problem)
    fringe.push(startNode)

    while True:
      if fringe.isEmpty():
        return False
      node = fringe.pop()
      if problem.isGoalState(node.state):
        return node.path
      if node.state not in closed:
        closed.add(node.state)
        for childNode in node.getSuccessors():
          fringe.push(childNode)
class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).
    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state
        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state
        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take
        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()
class PositionSearchProblem(SearchProblem):
    """
    A search problem defines the state space, start state, goal test,
    successor function and cost function.  This search problem can be
    used to find paths to a particular point on the pacman board.
    The state space consists of (x,y) positions in a pacman game.
    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.
        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print 'Warning: this does not look like a regular search maze'

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.
         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost



def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built.  The gameState can be any game state -- Pacman's position
    in that state is ignored.
    Example usage: mazeDistance( (2,4), (5,6), gameState)
    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + point1
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(breadthFirstSearch(prob))


