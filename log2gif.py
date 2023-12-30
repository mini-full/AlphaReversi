from termios import VMIN
import numpy as np
import re
import matplotlib.pyplot as plt
import gif
import argparse


@gif.frame
def plot_chess_board(board):
    # 创建一个8x8的棋盘
    chess_board = np.zeros((8, 8), dtype=float)

    # 初始化棋盘颜色
    chess_board[::2, ::2] = 0.7
    chess_board[1::2, 1::2] = 0.7
    chess_board[1::2, ::2] = 0.8
    chess_board[::2, 1::2] = 0.8

    # 创建一个图形对象
    fig, ax = plt.subplots()

    # 绘制棋盘
    ax.imshow(
        chess_board, cmap="gray", origin="lower", extent=[0, 8, 0, 8], vmin=0, vmax=1
    )
    ax = plt.gca()  # 获取到当前坐标轴信息
    ax.xaxis.set_ticks_position("top")  # 将X坐标轴移到上面
    ax.invert_yaxis()  # 反转Y坐标轴

    # 在棋盘上添加棋子
    for i, row in enumerate(board.split("\n")):
        for j, cell in enumerate(row.split(" ")):
            if cell == "X":
                draw_piece(ax, j, i, "black")
            elif cell == "O":
                draw_piece(ax, j, i, "white")

    # 隐藏坐标轴
    ax.set_axis_off()

    # 显示图形
    # plt.show()
    # plt.savefig(f"{path}{file_name}{num}.png", dpi=300)


def draw_piece(ax, x, y, color="black" or "white"):
    circle = plt.Circle((x + 0.5, y + 0.5), 0.25, color=color)
    ax.add_patch(circle)


def main(log_path, gif_path, duration=300):
    with open(log_path, "r") as file:
        log = file.read()

    pattern = re.compile(r"A B C D E F G H\n((\d.*?\n){8})")
    matches = pattern.finditer(log)

    frame = []
    for i, match in enumerate(matches):
        board = match.group(1)
        board = re.sub(r"\d ", "", board)
        frame.append(plot_chess_board(board))
    gif.save(frame, gif_path, duration=duration)
    print(f"gif saved to {gif_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path", type=str)
    parser.add_argument("gif_path", type=str)
    args = parser.parse_args()
    main(args.log_path, args.gif_path)
