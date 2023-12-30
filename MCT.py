from math import sqrt, log

class MCT:

    def __init__(self, color, parent = None):
        self.N = 0 # total number of games
        self.Q = 0 # total number of winned games
        self.parent = parent
        self.color = color
        self.children = dict() # key: 落子位置, value: reference to an MCT instance

    def oppo(self):
        """
        Return a opposite color w.r.t. self.color
        """
        return "X" if self.color == "O" else "O"

    def UCB(self):
        """
        Upper Confidence Bound. Details in the report.
        :return: UCT value
        """
        return self.Q / self.N + 2 * sqrt(2 * log(self.parent.N) / self.N)
