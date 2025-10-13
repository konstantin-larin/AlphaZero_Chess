# #!/usr/bin/env python

import os.path
import torch
import numpy as np
from alpha_net import ChessNet as cnet
from chess_board import board as c_board
import encoder_decoder as ed
import copy
from MCTS_chess import UCT_search, do_decode_n_move_pieces, get_policy, save_as_pickle
import pickle
import torch.multiprocessing as mp
from collections import Counter
import datetime


class Arena():
    def __init__(self,current_chessnet,best_chessnet, max_moves, simulation_depth, dataset_path):
        self.current = current_chessnet
        self.best = best_chessnet
        self.max_moves = max_moves
        self.simulation_depth = simulation_depth
        self.dataset_path = dataset_path        
    
    def play_round(self):
        if np.random.uniform(0,1) <= 0.5:
            white = self.current; black = self.best; w = "current"; b = "best"
        else:
            white = self.best; black = self.current; w = "best"; b = "current"
        current_board = c_board()
        checkmate = False
        # states = []; 
        
        states = Counter()        
        value = 0
        game_states = {'s': [], 'p': [], 'v': []}
        while checkmate == False and current_board.move_count <= self.max_moves:
            # draw_counter = 0
            # for s in states:
            #     if np.array_equal(current_board.current_board,s):
            #         draw_counter += 1
            # if draw_counter == 3: # draw by repetition
            #     break


            board_tuple = tuple(current_board.current_board.flatten())
            states[board_tuple] += 1
            if states[board_tuple] == 3:
                # троекратное повторение ходов
                break

            
            board_state = copy.deepcopy(ed.encode_board(current_board))
            
            if current_board.player == 0:
                best_move, root = UCT_search(current_board,self.simulation_depth,white)
            elif current_board.player == 1:
                best_move, root = UCT_search(current_board,self.simulation_depth,black)

            current_board = do_decode_n_move_pieces(current_board,best_move) # decode move and move piece(s)
            policy = get_policy(root)
                   
            
            print(current_board.current_board,current_board.move_count)
            print(" ")

            game_states['s'].append(board_state)
            game_states['p'].append(policy)     

            if current_board.check_status() == True and current_board.in_check_possible_moves() == []: # checkmate
                if current_board.player == 0: # black wins
                    value = -1
                elif current_board.player == 1: # white wins
                    value = 1
                checkmate = True   
                game_states['v'].append(value)  
                break

            
            game_states['v'].append(0) 

        
        if value == -1:
            return b, game_states
        elif value == 1:
            return w, game_states
        else:
            return None, game_states
    
    def evaluate(self, num_games):        
        current_wins = 0
        for i in range(num_games):
            print("Match:",i + 1, '\n')
            winner, dataset = self.play_round(); 

            game_info = {
                'winner': winner,                
                'num_moves': len(dataset['v']),
                'dataset_name': "dataset_%i_%s" % (i, datetime.datetime.today().strftime("%Y-%m-%d")),
                'game_states': dataset
            }
            if winner is None:
                print('Draw!')
            else:
                print("%s wins!" % winner)            
            if winner == "current":
                current_wins += 1
            save_as_pickle(
            os.path.join(self.dataset_path, 
                         "game_info_%i_%s" % (i, datetime.datetime.today().strftime("%Y-%m-%d"))),
            game_info,)

        current_wins_ratio = current_wins/num_games
        print("Current_net wins ratio:", str(current_wins_ratio))
        if current_wins_ratio >= 0.55:
            return self.current
        else:          
            return self.best


