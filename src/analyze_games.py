# # -*- coding: utf-8 -*-
# """
# Created on Thu Feb  7 09:55:20 2019

# @author: WT
# """

# import os
# import numpy as np
# import pickle
# import encoder_decoder as ed
# from matplotlib.table import Table
# import pandas as pd
# import matplotlib.pyplot as plt


# def view_board(np_data, fmt='{:s}', bkg_colors=['yellow', 'white']):
#     data = pd.DataFrame(np_data, columns=['A','B','C','D','E','F','G','H'])
#     fig, ax = plt.subplots(figsize=[7,7])
#     ax.set_axis_off()
#     tb = Table(ax, bbox=[0,0,1,1])
#     nrows, ncols = data.shape
#     width, height = 1.0 / ncols, 1.0 / nrows

#     for (i,j), val in np.ndenumerate(data):
#         idx = [j % 2, (j + 1) % 2][i % 2]
#         color = bkg_colors[idx]

#         tb.add_cell(i, j, width, height, text=fmt.format(val), 
#                     loc='center', facecolor=color)

#     for i, label in enumerate(data.index):
#         tb.add_cell(i, -1, width, height, text=label, loc='right', 
#                     edgecolor='none', facecolor='none')

#     for j, label in enumerate(data.columns):
#         tb.add_cell(-1, j, width, height/2, text=label, loc='center', 
#                            edgecolor='none', facecolor='none')
#     tb.set_fontsize(24)
#     ax.add_table(tb)
#     return fig


# data_path = "./datasets/iter2/"
# file = "dataset_cpu1_5"
# filename = os.path.join(data_path,file)
# with open(filename, 'rb') as fo:
#     dataset = pickle.load(fo, encoding='bytes')

# last_move = np.argmax(dataset[-1][1])
# b = ed.decode_board(dataset[-1][0])
# act = ed.decode_action(b,last_move)

# b.move_piece(act[0][0],act[1][0],act[2][0])
# for i in range(len(dataset)):
#     board = ed.decode_board(dataset[i][0])
#     fig = view_board(board.current_board)
#     plt.savefig(os.path.join("C:/Users/WT/Desktop/Python_Projects/chess/chess_ai_py35updated/gamesimages/ex4/", \
#                              f"{file}_{i}.png"))
    
# fig = view_board(b.current_board)
# plt.savefig(os.path.join("C:/Users/WT/Desktop/Python_Projects/chess/chess_ai_py35updated/gamesimages/ex4/", \
#                              f"{file}_{i+1}.png"))