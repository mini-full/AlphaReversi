from MCT import MCT
from MCTS import MCTS
from PosFirstPlayer import PosFirstPlayer
from RandomPlayer import RandomPlayer
from copy import deepcopy
from board import Board



class AIPlayer:
    """
    AI 玩家
    """

    def __init__(self, color):
        """
        玩家初始化
        :param color: 下棋方，'X' - 黑棋，'O' - 白棋
        """

        self.color = color

    def get_move(self, board):
        """
        根据当前棋盘状态获取最佳落子位置
        :param board: 棋盘
        :return: action 最佳落子位置, e.g. 'A1'
        """
        if self.color == 'X':
            player_name = '黑棋'
        else:
            player_name = '白棋'
        print("请等一会，AI Player{}-{} 正在思考中...".format(player_name, self.color))

        # -----------------请实现你的算法代码--------------------------------------
        mcts = MCTS()
        root = MCT(color=self.color)
        mcts.UCTSearch(root, deepcopy(board))
        choice = root
        max_N = -5636
        for i in root.children.keys():
            child = root.children[i]
            if child.N > max_N:
                max_N = child.N
                choice = i
        action = choice
        # ------------------------------------------------------------------------

        return action


from game import Game

black_player =  AIPlayer("X")

white_player = PosFirstPlayer("O")

# 游戏初始化，第一个玩家是黑棋，第二个玩家是白棋
game = Game(black_player, white_player)

# 开始下棋
game.run()