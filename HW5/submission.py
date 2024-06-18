from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def __init__(self):
        self.lastPositions = []
        self.dc = None

    def getAction(self, gameState):
        """
        getAction chooses among the best options according to the evaluation function.

        getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
        ------------------------------------------------------------------------------
        Description of GameState and helper functions:

        A GameState specifies the full game state, including the food, capsules,
        agent configurations and score changes. In this function, the |gameState| argument
        is an object of GameState class. Following are a few of the helper methods that you
        can use to query a GameState object to gather information about the present state
        of Pac-Man, the ghosts and the maze.

        gameState.getLegalActions():
            Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

        gameState.generateSuccessor(agentIndex, action):
            Returns the successor state after the specified agent takes the action.
            Pac-Man is always agent 0.

        gameState.getPacmanState():
            Returns an AgentState object for pacman (in game.py)
            state.configuration.pos gives the current position
            state.direction gives the travel vector

        gameState.getGhostStates():
            Returns list of AgentState objects for the ghosts

        gameState.getNumAgents():
            Returns the total number of agents in the game

        gameState.getScore():
            Returns the score corresponding to the current state of the game
            It corresponds to Utility(s)


        The GameState class is defined in pacman.py and you might want to look into that for
        other helper methods, though you don't need to.
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (oldFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        return successorGameState.getScore()


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

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


######################################################################################
# Problem 1a: implementing minimax


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (problem 1)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction. Terminal states can be found by one of the following:
        pacman won, pacman lost or there are no legal moves.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
          Returns a list of legal actions for an agent
          agentIndex=0 means Pacman, ghosts are >= 1

        Directions.STOP:
          The stop direction, which is always legal

        gameState.generateSuccessor(agentIndex, action):
          Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
          Returns the total number of agents in the game

        gameState.getScore():
          Returns the score corresponding to the current state of the game
          It corresponds to Utility(s)

        gameState.isWin():
          Returns True if it's a winning state

        gameState.isLose():
          Returns True if it's a losing state

        self.depth:
          The depth to which search should continue
        """

        # BEGIN_YOUR_ANSWER
        def minimax_explore(gameState, search_depth, playing_agent):
            if gameState.isWin() or gameState.isLose() or search_depth == 0:
                return (None, self.evaluationFunction(gameState))

            (current_agent, next_agent) = (
                playing_agent,
                (playing_agent + 1) % gameState.getNumAgents(),
            )
            next_depth = search_depth - int(next_agent == 0)

            A = [action for action in gameState.getLegalActions(current_agent)]
            q = [
                minimax_explore(
                    gameState.generateSuccessor(current_agent, action),
                    next_depth,
                    next_agent,
                )[1]
                for action in A
            ]

            qopt = max(q) if current_agent == 0 else min(q)
            Aopt = A[q.index(qopt)]

            return Aopt, qopt

        return minimax_explore(gameState, self.depth, 0)[0]
        # END_YOUR_ANSWER

    def getQ(self, gameState, action):
        """
        Returns the minimax Q-Value from the current gameState and given action
        using self.depth and self.evaluationFunction.
        Terminal states can be found by one of the following:
        pacman won, pacman lost or there are no legal moves.
        """

        # BEGIN_YOUR_ANSWER
        def minimax_explore(gameState, search_depth, playing_agent):
            if gameState.isWin() or gameState.isLose() or search_depth == 0:
                return (None, self.evaluationFunction(gameState))

            (current_agent, next_agent) = (
                playing_agent,
                (playing_agent + 1) % gameState.getNumAgents(),
            )
            next_depth = search_depth - int(next_agent == 0)

            A = [action for action in gameState.getLegalActions(current_agent)]
            q = [
                minimax_explore(
                    gameState.generateSuccessor(current_agent, action),
                    next_depth,
                    next_agent,
                )[1]
                for action in A
            ]

            qopt = max(q) if current_agent == 0 else min(q)
            Aopt = A[q.index(qopt)]

            return Aopt, qopt

        return minimax_explore(gameState.generateSuccessor(0, action), self.depth, 1)[1]
        # END_YOUR_ANSWER


######################################################################################
# Problem 2a: implementing expectimax


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (problem 2)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """

        # BEGIN_YOUR_ANSWER
        def expectimax_explore(gameState, search_depth, playing_agent):
            if gameState.isWin() or gameState.isLose() or search_depth == 0:
                return (None, self.evaluationFunction(gameState))

            (current_agent, next_agent) = (
                playing_agent,
                (playing_agent + 1) % gameState.getNumAgents(),
            )
            next_depth = search_depth - int(next_agent == 0)

            A = [action for action in gameState.getLegalActions(current_agent)]
            q = [
                expectimax_explore(
                    gameState.generateSuccessor(current_agent, action),
                    next_depth,
                    next_agent,
                )[1]
                for action in A
            ]

            qopt = (
                max(q)
                if current_agent == 0
                else sum([q_ * (1 / len(A)) for q_, action in zip(q, A)])
            )
            Aopt = A[q.index(qopt)] if current_agent == 0 else None

            return Aopt, qopt

        return expectimax_explore(gameState, self.depth, 0)[0]
        # END_YOUR_ANSWER

    def getQ(self, gameState, action):
        """
        Returns the expectimax Q-Value using self.depth and self.evaluationFunction.
        """

        # BEGIN_YOUR_ANSWER
        def expectimax_explore(gameState, search_depth, playing_agent):
            if gameState.isWin() or gameState.isLose() or search_depth == 0:
                return (None, self.evaluationFunction(gameState))

            (current_agent, next_agent) = (
                playing_agent,
                (playing_agent + 1) % gameState.getNumAgents(),
            )
            next_depth = search_depth - int(next_agent == 0)

            A = [action for action in gameState.getLegalActions(current_agent)]
            q = [
                expectimax_explore(
                    gameState.generateSuccessor(current_agent, action),
                    next_depth,
                    next_agent,
                )[1]
                for action in A
            ]

            qopt = (
                max(q)
                if current_agent == 0
                else sum([q_ * (1 / len(A)) for q_, action in zip(q, A)])
            )
            Aopt = A[q.index(qopt)] if current_agent == 0 else None

            return Aopt, qopt

        return expectimax_explore(
            gameState.generateSuccessor(0, action), self.depth, 1
        )[1]
        # END_YOUR_ANSWER


######################################################################################
# Problem 3a: implementing biased-expectimax


class BiasedExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your biased-expectimax agent (problem 3)
    """

    def getAction(self, gameState):
        """
        Returns the biased-expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing stop-biasedly from their
        legal moves.
        """

        # BEGIN_YOUR_ANSWER
        def p_A(agent, action):
            if action == Directions.STOP:
                return 0.5 + (0.5 / len(gameState.getLegalActions(agent)))
            else:
                return 0.5 / len(gameState.getLegalActions(agent))

        def biased_expectimax_explore(gameState, search_depth, playing_agent):
            if gameState.isWin() or gameState.isLose() or search_depth == 0:
                return (None, self.evaluationFunction(gameState))

            (current_agent, next_agent) = (
                playing_agent,
                (playing_agent + 1) % gameState.getNumAgents(),
            )
            next_depth = search_depth - int(next_agent == 0)

            A = [action for action in gameState.getLegalActions(current_agent)]
            q = [
                biased_expectimax_explore(
                    gameState.generateSuccessor(current_agent, action),
                    next_depth,
                    next_agent,
                )[1]
                for action in A
            ]

            qopt = (
                max(q)
                if current_agent == 0
                else sum([q_ * p_A(current_agent, action) for q_, action in zip(q, A)])
            )
            Aopt = A[q.index(qopt)] if current_agent == 0 else None

            return Aopt, qopt

        return biased_expectimax_explore(gameState, self.depth, 0)[0]
        # END_YOUR_ANSWER

    def getQ(self, gameState, action):
        """
        Returns the biased-expectimax Q-Value using self.depth and self.evaluationFunction.
        """

        # BEGIN_YOUR_ANSWER
        def p_A(agent, action):
            if action == Directions.STOP:
                return 0.5 + (0.5 / len(gameState.getLegalActions(agent)))
            else:
                return 0.5 / len(gameState.getLegalActions(agent))

        def biased_expectimax_explore(gameState, search_depth, playing_agent):
            if gameState.isWin() or gameState.isLose() or search_depth == 0:
                return (None, self.evaluationFunction(gameState))

            (current_agent, next_agent) = (
                playing_agent,
                (playing_agent + 1) % gameState.getNumAgents(),
            )
            next_depth = search_depth - int(next_agent == 0)

            A = [action for action in gameState.getLegalActions(current_agent)]
            q = [
                biased_expectimax_explore(
                    gameState.generateSuccessor(current_agent, action),
                    next_depth,
                    next_agent,
                )[1]
                for action in A
            ]

            qopt = (
                max(q)
                if current_agent == 0
                else sum([q_ * p_A(current_agent, action) for q_, action in zip(q, A)])
            )
            Aopt = A[q.index(qopt)] if current_agent == 0 else None

            return Aopt, qopt

        return biased_expectimax_explore(
            gameState.generateSuccessor(0, action), self.depth, 1
        )[1]
        # END_YOUR_ANSWER


######################################################################################
# Problem 4a: implementing expectiminimax


class ExpectiminimaxAgent(MultiAgentSearchAgent):
    """
    Your expectiminimax agent (problem 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectiminimax action using self.depth and self.evaluationFunction

        The even-numbered ghost should be modeled as choosing uniformly at random from their
        legal moves.
        """

        # BEGIN_YOUR_ANSWER
        def expecti_minimax_explore(gameState, search_depth, playing_agent):
            if gameState.isWin() or gameState.isLose() or search_depth == 0:
                return (None, self.evaluationFunction(gameState))

            (current_agent, next_agent) = (
                playing_agent,
                (playing_agent + 1) % gameState.getNumAgents(),
            )
            next_depth = search_depth - int(next_agent == 0)

            if playing_agent == 0:
                A = [action for action in gameState.getLegalActions(current_agent)]
                q = [
                    expecti_minimax_explore(
                        gameState.generateSuccessor(current_agent, action),
                        next_depth,
                        next_agent,
                    )[1]
                    for action in A
                ]

                qopt = max(q)
                Aopt = A[q.index(qopt)]

                return Aopt, qopt

            elif playing_agent % 2 == 1:
                A = [action for action in gameState.getLegalActions(current_agent)]
                q = [
                    expecti_minimax_explore(
                        gameState.generateSuccessor(current_agent, action),
                        next_depth,
                        next_agent,
                    )[1]
                    for action in A
                ]

                qopt = min(q)
                Aopt = A[q.index(qopt)]

                return Aopt, qopt

            elif playing_agent % 2 == 0:
                A = [action for action in gameState.getLegalActions(current_agent)]
                q = [
                    expecti_minimax_explore(
                        gameState.generateSuccessor(current_agent, action),
                        next_depth,
                        next_agent,
                    )[1]
                    for action in A
                ]

                qopt = sum([q_ * (1 / len(A)) for q_, action in zip(q, A)])
                Aopt = None

                return Aopt, qopt

        return expecti_minimax_explore(gameState, self.depth, 0)[0]
        # END_YOUR_ANSWER

    def getQ(self, gameState, action):
        """
        Returns the expectiminimax Q-Value using self.depth and self.evaluationFunction.
        """

        # BEGIN_YOUR_ANSWER
        def expecti_minimax_explore(gameState, search_depth, playing_agent):
            if gameState.isWin() or gameState.isLose() or search_depth == 0:
                return (None, self.evaluationFunction(gameState))

            (current_agent, next_agent) = (
                playing_agent,
                (playing_agent + 1) % gameState.getNumAgents(),
            )
            next_depth = search_depth - int(next_agent == 0)

            if playing_agent == 0:
                A = [action for action in gameState.getLegalActions(current_agent)]
                q = [
                    expecti_minimax_explore(
                        gameState.generateSuccessor(current_agent, action),
                        next_depth,
                        next_agent,
                    )[1]
                    for action in A
                ]

                qopt = max(q)
                Aopt = A[q.index(qopt)]

                return Aopt, qopt

            elif playing_agent % 2 == 1:
                A = [action for action in gameState.getLegalActions(current_agent)]
                q = [
                    expecti_minimax_explore(
                        gameState.generateSuccessor(current_agent, action),
                        next_depth,
                        next_agent,
                    )[1]
                    for action in A
                ]

                qopt = min(q)
                Aopt = A[q.index(qopt)]

                return Aopt, qopt

            elif playing_agent % 2 == 0:
                A = [action for action in gameState.getLegalActions(current_agent)]
                q = [
                    expecti_minimax_explore(
                        gameState.generateSuccessor(current_agent, action),
                        next_depth,
                        next_agent,
                    )[1]
                    for action in A
                ]

                qopt = sum([q_ * (1 / len(A)) for q_, action in zip(q, A)])
                Aopt = None

                return Aopt, qopt

        return expecti_minimax_explore(
            gameState.generateSuccessor(0, action), self.depth, 1
        )[1]
        # END_YOUR_ANSWER


######################################################################################
# Problem 5a: implementing alpha-beta


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your expectiminimax agent with alpha-beta pruning (problem 5)
    """

    def getAction(self, gameState):
        """
        Returns the expectiminimax action using self.depth and self.evaluationFunction
        """

        # BEGIN_YOUR_ANSWER
        def alpha_beta_explore(gameState, search_depth, playing_agent, alpha, beta):
            if gameState.isWin() or gameState.isLose() or search_depth == 0:
                return (None, self.evaluationFunction(gameState))

            current_agent, next_agent = (
                playing_agent,
                (playing_agent + 1) % gameState.getNumAgents(),
            )
            next_depth = search_depth - int(next_agent == 0)

            if current_agent == 0:
                Aopt, qopt = None, float("-inf")
                for nowA in gameState.getLegalActions(current_agent):
                    A, q = alpha_beta_explore(
                        gameState.generateSuccessor(current_agent, nowA),
                        next_depth,
                        next_agent,
                        alpha,
                        beta,
                    )

                    Aopt, qopt = (nowA, q) if q >= qopt else (Aopt, qopt)
                    alpha = max(alpha, qopt)

                    if beta <= alpha:
                        break

            elif current_agent % 2 == 1:
                Aopt, qopt = None, float("inf")
                for nowA in gameState.getLegalActions(current_agent):
                    A, q = alpha_beta_explore(
                        gameState.generateSuccessor(current_agent, nowA),
                        next_depth,
                        next_agent,
                        alpha,
                        beta,
                    )

                    Aopt, qopt = (nowA, q) if q < qopt else (Aopt, qopt)
                    beta = min(beta, qopt)

                    if beta <= alpha:
                        break

            elif current_agent % 2 == 0:
                A = [action for action in gameState.getLegalActions(current_agent)]
                q = [
                    alpha_beta_explore(
                        gameState.generateSuccessor(current_agent, action),
                        next_depth,
                        next_agent,
                        alpha,
                        beta,
                    )[1]
                    for action in gameState.getLegalActions(current_agent)
                ]

                Aopt, qopt = None, sum([q_ * (1 / (len(A))) for q_, act in zip(q, A)])

            return (Aopt, qopt)

        return alpha_beta_explore(
            gameState, self.depth, 0, float("-inf"), float("inf")
        )[0]
        # END_YOUR_ANSWER

    def getQ(self, gameState, action):
        """
        Returns the expectiminimax Q-Value using self.depth and self.evaluationFunction.
        """

        # BEGIN_YOUR_ANSWER
        def alpha_beta_explore(gameState, search_depth, playing_agent, alpha, beta):
            if gameState.isWin() or gameState.isLose() or search_depth == 0:
                return (None, self.evaluationFunction(gameState))

            current_agent, next_agent = (
                playing_agent,
                (playing_agent + 1) % gameState.getNumAgents(),
            )
            next_depth = search_depth - int(next_agent == 0)

            if current_agent == 0:
                Aopt, qopt = None, float("-inf")
                for nowA in gameState.getLegalActions(current_agent):
                    A, q = alpha_beta_explore(
                        gameState.generateSuccessor(current_agent, nowA),
                        next_depth,
                        next_agent,
                        alpha,
                        beta,
                    )

                    Aopt, qopt = (nowA, q) if q >= qopt else (Aopt, qopt)
                    alpha = max(alpha, qopt)

                    if beta <= alpha:
                        break

            elif current_agent % 2 == 1:
                Aopt, qopt = None, float("inf")
                for nowA in gameState.getLegalActions(current_agent):
                    A, q = alpha_beta_explore(
                        gameState.generateSuccessor(current_agent, nowA),
                        next_depth,
                        next_agent,
                        alpha,
                        beta,
                    )

                    Aopt, qopt = (nowA, q) if q < qopt else (Aopt, qopt)
                    beta = min(beta, qopt)

                    if beta <= alpha:
                        break

            elif current_agent % 2 == 0:
                A = [action for action in gameState.getLegalActions(current_agent)]
                q = [
                    alpha_beta_explore(
                        gameState.generateSuccessor(current_agent, action),
                        next_depth,
                        next_agent,
                        alpha,
                        beta,
                    )[1]
                    for action in gameState.getLegalActions(current_agent)
                ]

                Aopt, qopt = None, sum([q_ * (1 / (len(A))) for q_, act in zip(q, A)])

            return (Aopt, qopt)

        return alpha_beta_explore(
            gameState.generateSuccessor(0, action),
            self.depth,
            1,
            float("-inf"),
            float("inf"),
        )[1]
        # END_YOUR_ANSWER


######################################################################################
# Problem 6a: creating a better evaluation function


def betterEvaluationFunction(currentGameState):
    """
    Your extreme, unstoppable evaluation function (problem 6).
    """

    # BEGIN_YOUR_ANSWER
    raise NotImplemented
    # END_YOUR_ANSWER


def choiceAgent():
    """
    Choose the pacman agent model you want for problem 6.
    You can choose among the agents above or design your own agent model.
    You should return the name of class of pacman agent.
    (e.g. 'MinimaxAgent', 'BiasedExpectimaxAgent', 'MyOwnAgent', ...)
    """
    # BEGIN_YOUR_ANSWER
    option = 5 - 1  # Index of the agent will be choosed

    list_of_agent_names = [
        "MinimaxAgent",
        "ExpectimaxAgent",
        "BiasedExpectimaxAgent",
        "ExpectiminimaxAgent",
        "AlphaBetaAgent",
    ]

    return list_of_agent_names[option]
    # END_YOUR_ANSWER


# Abbreviation
better = betterEvaluationFunction
