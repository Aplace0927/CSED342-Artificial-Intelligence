from typing import Callable, List, Set

import shell
import util
import wordsegUtil


############################################################
# Problem 1a: Solve the segmentation problem under a unigram model


class WordSegmentationProblem(util.SearchProblem):
    def __init__(self, query: str, unigramCost: Callable[[str], float]):
        self.query = query
        self.unigramCost = unigramCost

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return self.query
        # END_YOUR_CODE

    def isEnd(self, state) -> bool:
        # BEGIN_YOUR_CODE (our solution is 1 lines of code, but don't worry if you deviate from this)
        return len(state) == 0
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        return [
            (state[:x], state[x:], self.unigramCost(state[:x]))
            for x in range(len(state), 0, -1)
        ]
        # END_YOUR_CODE


def segmentWords(query: str, unigramCost: Callable[[str], float]) -> str:
    if len(query) == 0:
        return ""

    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(WordSegmentationProblem(query, unigramCost))

    # BEGIN_YOUR_CODE (our solution is 1 lines of code, but don't worry if you deviate from this)
    return " ".join(ucs.actions)
    # END_YOUR_CODE


############################################################
# Problem 1b: Solve the vowel insertion problem under a bigram cost


class VowelInsertionProblem(util.SearchProblem):
    def __init__(
        self,
        queryWords: List[str],
        bigramCost: Callable[[str, str], float],
        possibleFills: Callable[[str], Set[str]],
    ):
        self.queryWords = queryWords
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return (0, self.queryWords[0])
        # END_YOUR_CODE

    def isEnd(self, state) -> bool:
        # BEGIN_YOUR_CODE (our solution is 1 lines of code, but don't worry if you deviate from this)
        return state[0] == len(self.queryWords) - 1
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
        fills = self.possibleFills(self.queryWords[state[0] + 1])
        words = fills if len(fills) != 0 else {self.queryWords[state[0] + 1]}
        return [(x, (state[0] + 1, x), self.bigramCost(state[1], x)) for x in words]
        # END_YOUR_CODE


def insertVowels(
    queryWords: List[str],
    bigramCost: Callable[[str, str], float],
    possibleFills: Callable[[str], Set[str]],
) -> str:
    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
    if len(queryWords) == 0:
        return ""

    ucs = util.UniformCostSearch()
    ucs.solve(
        VowelInsertionProblem(
            [wordsegUtil.SENTENCE_BEGIN] + queryWords, bigramCost, possibleFills
        )
    )
    return " ".join(ucs.actions)
    # END_YOUR_CODE


############################################################
# Problem 1c: Solve the joint segmentation-and-insertion problem


class JointSegmentationInsertionProblem(util.SearchProblem):
    def __init__(
        self,
        query: str,
        bigramCost: Callable[[str, str], float],
        possibleFills: Callable[[str], Set[str]],
    ):
        self.query = query
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return (self.query, wordsegUtil.SENTENCE_BEGIN)
        # END_YOUR_CODE

    def isEnd(self, state) -> bool:
        # BEGIN_YOUR_CODE (our solution is 1 lines of code, but don't worry if you deviate from this)
        return len(state[0]) == 0
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
        res = []
        for prev, succ in [
            (state[0][:x], state[0][x:]) for x in range(len(state[0]), 0, -1)
        ]:
            for wd in self.possibleFills(prev):
                res.append((wd, (succ, wd), self.bigramCost(state[1], wd)))
        return res
        # END_YOUR_CODE


def segmentAndInsert(
    query: str,
    bigramCost: Callable[[str, str], float],
    possibleFills: Callable[[str], Set[str]],
) -> str:
    if len(query) == 0:
        return ""

    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
    ucs = util.UniformCostSearch()
    ucs.solve(JointSegmentationInsertionProblem(query, bigramCost, possibleFills))
    return " ".join(ucs.actions)
    # END_YOUR_CODE


############################################################
# Problem 2a: Solve the maze search problem with uniform cost search


class MazeProblem(util.SearchProblem):
    def __init__(
        self,
        start: tuple,
        goal: tuple,
        moveCost: Callable[[tuple, str], float],
        possibleMoves: Callable[[tuple], Set[tuple]],
    ) -> float:
        self.start = start
        self.goal = goal
        self.moveCost = moveCost
        self.possibleMoves = possibleMoves

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return self.start
        # END_YOUR_CODE

    def isEnd(self, state) -> bool:
        # BEGIN_YOUR_CODE (our solution is 1 lines of code, but don't worry if you deviate from this)
        return state == self.goal
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        return [
            (
                direct,
                (
                    state[0] + util.directions[direct][0],
                    state[1] + util.directions[direct][1],
                ),
                self.moveCost(state, direct),
            )
            for direct in util.directions.keys()
        ]
        # END_YOUR_CODE


def UCSMazeSearch(
    start: tuple,
    goal: tuple,
    moveCost: Callable[[tuple, str], float],
    possibleMoves: Callable[[tuple], Set[tuple]],
) -> float:
    ucs = util.UniformCostSearch()
    ucs.solve(MazeProblem(start, goal, moveCost, possibleMoves))

    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    return ucs.totalCost
    # END_YOUR_CODE


############################################################
# Problem 2b: Solve the maze search problem with A* search


def consistentHeuristic(goal: tuple):
    def _consistentHeuristic(state: tuple) -> float:
        # BEGIN_YOUR_CODE (our solution is 1 lines of code, but don't worry if you deviate from this)
        return abs(state[0] - goal[0]) + abs(state[1] - goal[1])
        # END_YOUR_CODE

    return _consistentHeuristic


def AStarMazeSearch(
    start: tuple,
    goal: tuple,
    moveCost: Callable[[tuple, str], float],
    possibleMoves: Callable[[tuple], Set[tuple]],
) -> float:
    ucs = util.UniformCostSearch()
    ucs.solve(
        MazeProblem(start, goal, moveCost, possibleMoves),
        heuristic=consistentHeuristic(goal),
    )

    # BEGIN_YOUR_CODE (our solution is 13 lines of code, but don't worry if you deviate from this)
    x, y, cost = start[0], start[1], 0
    for action in ucs.actions:
        cost += moveCost((x, y), action)
        x, y = x + util.directions[action][0], y + util.directions[action][1]
    return cost
    # END_YOUR_CODE


############################################################


if __name__ == "__main__":
    shell.main()
