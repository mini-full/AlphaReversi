from PosFirstPlayer import PosFirstPlayer
from MCT import MCT
from game import SimulateGame
from board import Board
from copy import deepcopy
import time

class MCTS:
    """
    Monte Carlo Tree Search
    """

    def __init__(self):
        self.simulate_black = PosFirstPlayer('X')
        self.simulate_white = PosFirstPlayer('O')
        self.begin_time = time.time()

    def select(self, node : MCT, board : Board):
        
        max_UCB = -5636
        choice = None
        if len(node.children) == 0:
            return node
        for i in node.children.keys():
            child = node.children[i]
            if child.N <= 1:
                choice = i
                break
            else:
                temp = child.UCB()
                if temp > max_UCB:
                    max_UCB = temp
                    choice = i

        board._move(choice, node.color)
        return self.select(node.children[choice], board)
            
    def expand(self, node, board):
        children = board.get_legal_actions(node.color)
        for i in children:
            node.children[i] = MCT(node.oppo(), node)

    def simulate_policy(self, node, board : Board):
        """
        Simulate using self defined SimualteGame
        """

        current_player = self.simulate_white if node.color == 'X' else self.simulate_black
        return SimulateGame(self.simulate_black, self.simulate_white, board, current_player).run()

    def backprop(self, node, score):
        """_summary_
        Update the Q and N of the node and its ancestors
        Args:
            node (_type_): _description_
            score (_type_): _description_
        """
        v = node
        while v is not None:
            v.Q += score
            v.N += 1
            v = v.parent
            score = 1 - score

    def UCTSearch(self, root, board):
        while time.time() - self.begin_time < 59: # prevent longer search time
            saved_board = deepcopy(board)
            v_select = self.select(root, saved_board)
            self.expand(v_select, saved_board)
            winner = self.simulate_policy(v_select, saved_board)
            
            if winner == 2: # draw
                score = 0.5
            elif winner == 0: # black win
                score = 1
            else: # white win
                score = 0
            # score is won by black                
            if v_select.color == 'X': # v_select is white
                score = 1 - score
            
            self.backprop(v_select, score)