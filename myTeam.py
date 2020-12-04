# myTeam.py
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


import game
import random, time, util
from game import Directions
from game import Actions
from util import PriorityQueue
from util import nearestPoint
from captureAgents import CaptureAgent


#################
# Team creation #
#################

### Mahmood Shilleh TAMU

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveReflexAgent', second='DefensiveReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):

    actions = gameState.getLegalActions(self.index)
    values = [self.evaluate(gameState, a) for a in actions]

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start, pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 1.0}

  def avoidGhostHeuristic(self, gameState, state):
    enemyGhosts = [gameState.getAgentState(opponent) for opponent in self.getOpponents(gameState) if not gameState.getAgentState(opponent).isPacman and gameState.getAgentState(opponent).scaredTimer < 1 and gameState.getAgentState(opponent).getPosition() is not None]
    heuristic = 0
    ghostPositionListDistance = []
    for ghost in enemyGhosts:
      ghostPositionListDistance.append(self.getMazeDistance(state, ghost.getPosition()))
    if ghostPositionListDistance != []:
      closestGhostDis = min(ghostPositionListDistance)
      if closestGhostDis < 3:
        heuristic = (4 - closestGhostDis) ** 4
    return heuristic

  def secureFood(self,gameState):
    foodList = self.getFood(gameState).asList()
    secureList = []
    for food in foodList:
      counter = 0
      if not gameState.hasWall(food[0] + 1, food[1]):
        counter += 1
      if not gameState.hasWall(food[0] - 1, food[1]):
        counter += 1
      if not gameState.hasWall(food[0], food[1] + 1):
        counter += 1
      if not gameState.hasWall(food[0], food[1] - 1):
        counter += 1
      if counter > 2:
        secureList.append(food)
    return secureList

  ## This function finds the closest ghost distance if the agent is a Pacman and the ghost is not scared

  def closestGhostDistanceifPacman(self, gameState):
    if gameState.getAgentState(self.index).isPacman:
      enemyGhosts = [gameState.getAgentState(opponent) for opponent in self.getOpponents(gameState) if not gameState.getAgentState(opponent).isPacman and gameState.getAgentState(opponent).scaredTimer < 1 and gameState.getAgentState(opponent).getPosition() is not None]
      myAgentPosition = gameState.getAgentState(self.index).getPosition()
      ghostPositionListDistance = []
      for ghost in enemyGhosts:
        ghostPositionListDistance.append(self.getMazeDistance(myAgentPosition, ghost.getPosition()))
      if ghostPositionListDistance != []:
        return min(ghostPositionListDistance)
      else:
        return 0
    else:
      return 0



##########################################


              # Offender #


##########################################


class OffensiveReflexAgent(ReflexCaptureAgent):

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()
    features['successorScore'] = -len(foodList)  # self.getScore(successor)

    # Compute distance to the nearest food

    if len(foodList) > 0:  # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1}

  def scaredGhostTimer(self, gameState):
    for ghost in self.getOpponents(gameState):
      if gameState.getAgentState(ghost).scaredTimer > 0:
        return gameState.getAgentState(ghost).scaredTimer
    return 0

  def aStarSearch(self, problem, gameState, heuristic):
    a = [problem.getStartState()]
    fringe = util.PriorityQueue()
    fringe.push((a[0], [], 0), heuristic(gameState, a[0]))
    visited_nodes = []
    expanded_nodes = [problem.getStartState()]
    while fringe.isEmpty() == False:
      a = fringe.pop()
      visited_nodes.append(a[0])
      if problem.isGoalState(a[0]) == True:
        return a[1][0]
      b = problem.getSuccessors(a[0])
      for (new_node, new_direction, new_cost) in b:
        if new_node not in visited_nodes and new_node not in expanded_nodes:
          # Do not want to expand twice
          if problem.isGoalState(new_node) != True:
            expanded_nodes.append(new_node)
          new_direction = a[1] + [new_direction]
          new_cost = a[2] + new_cost
          fringe.push((new_node, new_direction, new_cost), new_cost + heuristic(gameState, new_node))
    return []

  def chooseAction(self, gameState):

    ## The attacker can do many things, we want problems that define getting food, returning food, getting capsules and so on... all shown below
    if gameState.getAgentState(self.index).numCarrying > 2:
      problem = ReturnFoodtoBoundary(gameState, self, self.index)
      if self.scaredGhostTimer(gameState) > 5:
        problem = FoodSearchProblem(gameState, self, self.index)
      elif self.closestGhostDistanceifPacman(gameState) > 8:
        if len(self.secureFood(gameState)) > 0:
          problem = SecureSearchProblem(gameState, self, self.index)
      if (gameState.data.timeleft) < 75:
        problem = ReturnFoodtoBoundary(gameState, self, self.index)

    elif len(self.getCapsules(gameState)) > 0 and self.scaredGhostTimer(gameState) < 6:
      problem = CapsuleSearchProblem(gameState, self, self.index)

    elif len(self.secureFood(gameState)) > 0:

      if self.closestGhostDistanceifPacman(gameState) < 3:
        actions = gameState.getLegalActions(self.index)

        values = [self.evaluate(gameState, a) for a in actions]

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 2:
          bestDist = 9999
          for action in actions:
            successor = self.getSuccessor(gameState, action)
            pos2 = successor.getAgentPosition(self.index)
            dist = self.getMazeDistance(self.start, pos2)
            if dist < bestDist:
              bestAction = action
              bestDist = dist
          return bestAction
        return random.choice(bestActions)

      else:
        problem = SecureSearchProblem(gameState, self, self.index)

    else:
      problem = FoodSearchProblem(gameState, self, self.index)

    action = self.aStarSearch(problem, gameState, self.avoidGhostHeuristic)

    ### Sometimes astar cannot find a solution given the problem

    if action != []:
      return action
    else:
      actions = gameState.getLegalActions(self.index)

      values = [self.evaluate(gameState, a) for a in actions]

      maxValue = max(values)
      bestActions = [a for a, v in zip(actions, values) if v == maxValue]

      foodLeft = len(self.getFood(gameState).asList())

      if foodLeft <= 2:
        bestDist = 9999
        for action in actions:
          successor = self.getSuccessor(gameState, action)
          pos2 = successor.getAgentPosition(self.index)
          dist = self.getMazeDistance(self.start, pos2)
          if dist < bestDist:
            bestAction = action
            bestDist = dist
        return bestAction
      return random.choice(bestActions)


############################################


    # All attacker Search Problems


############################################

### This search problem is the father of all the other search problems I defined, this was taken from Assignment 1

class SearchProblem:
  def __init__(self, gameState, agentIndex, costFn=lambda x: 1):
    self.startState = gameState.getAgentState(agentIndex).getPosition()
    self.walls = gameState.getWalls()
    self.costFn = costFn

  def getSuccessors(self, state):
    successors = []
    for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
      x, y = state
      dx, dy = Actions.directionToVector(action)
      nextx, nexty = int(x + dx), int(y + dy)
      if not self.walls[nextx][nexty]:
        nextState = (nextx, nexty)
        cost = self.costFn(nextState)
        successors.append((nextState, action, cost))

    self._expanded += 1
    if state not in self._visited:
      self._visited[state] = True
      self._visitedlist.append(state)

    return successors

  def getCostOfActions(self, actions):
    if actions == None: return 999999
    x, y = self.getStartState()
    cost = 0
    for action in actions:
      dx, dy = Actions.directionToVector(action)
      x, y = int(x + dx), int(y + dy)
      if self.walls[x][y]: return 999999
      cost += self.costFn((x, y))
    return cost


class FoodSearchProblem(SearchProblem):

  def __init__(self, gameState, agent, agentIndex = 0):
    "Stores information from the gameState.  You don't need to change this."
    # Store the food for later reference
    self.gameState = gameState
    self.agent = agent
    self.food = agent.getFood(gameState)
    self.startState = gameState.getAgentState(agentIndex).getPosition()
    self.walls = gameState.getWalls()
    self.costFn = lambda x: 1
    self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE

  def getStartState(self):
    return self.startState

  def isGoalState(self, state):
    return state in self.food.asList()


class SecureSearchProblem(SearchProblem):

  def __init__(self, gameState, agent, agentIndex = 0):
    self.secureFood = agent.secureFood(gameState)
    self.startState = gameState.getAgentState(agentIndex).getPosition()
    self.walls = gameState.getWalls()
    self.costFn = lambda x: 1
    self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE

  def getStartState(self):
    return self.startState

  def isGoalState(self, state):
    return state in self.secureFood


class CapsuleSearchProblem(SearchProblem):
  def __init__(self, gameState, agent, agentIndex = 0):
    self.startState = gameState.getAgentState(agentIndex).getPosition()
    self.walls = gameState.getWalls()
    self.costFn = lambda x: 1
    self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE
    self.capsules = agent.getCapsules(gameState)

  def getStartState(self):
    return self.startState

  def isGoalState(self, state):
    return state in self.capsules


class ReturnFoodtoBoundary(SearchProblem):
  def __init__(self, gameState, agent, agentIndex=0):
    self.gameState = gameState
    self.agent = agent
    self.startState = gameState.getAgentState(agentIndex).getPosition()
    self.walls = gameState.getWalls()
    self.costFn = lambda x: 1
    self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE

  def getStartState(self):
    return self.startState

  def isGoalState(self, state):
    width = self.gameState.data.layout.width
    if self.agent.red:
      return state[0] < width / 2
    else:
      return state[0] > width / 2


##########################################


              # Defender #


##########################################

class DefensiveReflexAgent(ReflexCaptureAgent):

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}

  def aStarSearch(self, problem, gameState, heuristic):
    a = [problem.getStartState()]
    fringe = util.PriorityQueue()
    fringe.push((a[0], [], 0), heuristic(gameState, a[0]))
    visited_nodes = []
    expanded_nodes = [problem.getStartState()]
    while fringe.isEmpty() == False:
      a = fringe.pop()
      visited_nodes.append(a[0])
      if problem.isGoalState(a[0]) == True:
        return a[1][0]
      b = problem.getSuccessors(a[0])
      for (new_node, new_direction, new_cost) in b:
        if new_node not in visited_nodes and new_node not in expanded_nodes:
          if problem.isGoalState(new_node) != True:
            expanded_nodes.append(new_node)
          new_direction = a[1] + [new_direction]
          new_cost = a[2] + new_cost
          fringe.push((new_node, new_direction, new_cost), new_cost + heuristic(gameState, new_node))
    return []

  def chooseAction(self, gameState):
    # This gets the opponents who are currently invading, if there are none, I have the chance to help my offender and go get food
    pacmanOpponents = [gameState.getAgentState(opponent) for opponent in self.getOpponents(gameState) if gameState.getAgentState(opponent).isPacman]
    if len(pacmanOpponents) < 1:
      if gameState.getAgentState(self.index).numCarrying > 2:
        problem = ReturnFoodtoBoundary(gameState, self, self.index)
      else:
        problem = FoodSearchProblem(gameState, self, self.index)

      action = self.aStarSearch(problem, gameState, self.avoidGhostHeuristic)
      if action != []:
        return action

    actions = gameState.getLegalActions(self.index)

    values = [self.evaluate(gameState, a) for a in actions]

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start, pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

## This is it