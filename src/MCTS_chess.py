#!/usr/bin/env python
import pickle
import os
from collections import Counter, defaultdict

import numpy as np
import math
import encoder_decoder as ed
from chess_board import board as c_board
import copy
import torch
import torch.multiprocessing as mp
from alpha_net import ChessNet
import datetime
import h5py
import mlflow

class UCTNode():
    def __init__(self, game, move, parent=None):
        self.game = game # state s
        self.move = move # action index
        self.is_expanded = False
        self.parent = parent  
        self.children = {}
        self.child_priors = np.zeros([4672], dtype=np.float32)
        self.child_total_value = np.zeros([4672], dtype=np.float32)
        self.child_number_visits = np.zeros([4672], dtype=np.float32)
        self.action_idxes = []
        
    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.move]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.move] = value
    
    @property
    def total_value(self):
        return self.parent.child_total_value[self.move]
    
    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[self.move] = value
    
    def child_Q(self):
        return self.child_total_value / (1 + self.child_number_visits)
    
    def child_U(self):
        return math.sqrt(self.number_visits) * (
            abs(self.child_priors) / (1 + self.child_number_visits))
    
    def best_child(self):
        if self.action_idxes != []:
            bestmove = self.child_Q() + self.child_U()
            bestmove = self.action_idxes[np.argmax(bestmove[self.action_idxes])]
        else:
            bestmove = np.argmax(self.child_Q() + self.child_U())
        return bestmove
    
    def select_leaf(self):
        current = self
        while current.is_expanded:
          best_move = current.best_child()
          current = current.maybe_add_child(best_move)
        return current
    
    def add_dirichlet_noise(self,action_idxs,child_priors):
        valid_child_priors = child_priors[action_idxs] # select only legal moves entries in child_priors array
        valid_child_priors = 0.75*valid_child_priors + 0.25*np.random.dirichlet(np.zeros([len(valid_child_priors)], dtype=np.float32)+0.3)
        child_priors[action_idxs] = valid_child_priors
        return child_priors
    
    def expand(self, child_priors):
        self.is_expanded = True
        action_idxs = []; c_p = child_priors
        for action in self.game.actions(): # possible actions
            if action != []:
                initial_pos,final_pos,underpromote = action
                action_idxs.append(ed.encode_action(self.game,initial_pos,final_pos,underpromote))
        if action_idxs == []:
            self.is_expanded = False
        self.action_idxes = action_idxs
        for i in range(len(child_priors)): # mask all illegal actions
            if i not in action_idxs:
                c_p[i] = 0.0000000000
        if self.parent.parent == None: # add dirichlet noise to child_priors in root node
            c_p = self.add_dirichlet_noise(action_idxs,c_p)
        self.child_priors = c_p
    
    def decode_n_move_pieces(self,board,move):
        i_pos, f_pos, prom = ed.decode_action(board,move)
        for i, f, p in zip(i_pos,f_pos,prom):
            board.player = self.game.player
            board.move_piece(i,f,p) # move piece to get next board state s
            a,b = i; c,d = f
            if board.current_board[c,d] in ["K","k"] and abs(d-b) == 2: # if king moves 2 squares, then move rook too for castling
                if a == 7 and d-b > 0: # castle kingside for white
                    board.player = self.game.player
                    board.move_piece((7,7),(7,5),None)
                if a == 7 and d-b < 0: # castle queenside for white
                    board.player = self.game.player
                    board.move_piece((7,0),(7,3),None)
                if a == 0 and d-b > 0: # castle kingside for black
                    board.player = self.game.player
                    board.move_piece((0,7),(0,5),None)
                if a == 0 and d-b < 0: # castle queenside for black
                    board.player = self.game.player
                    board.move_piece((0,0),(0,3),None)
        return board
            
    
    def maybe_add_child(self, move):
        if move not in self.children:
            copy_board = copy.deepcopy(self.game) # make copy of board
            copy_board = self.decode_n_move_pieces(copy_board,move)
            self.children[move] = UCTNode(
              copy_board, move, parent=self)
        return self.children[move]
    
    def backup(self, value_estimate: float):
        current = self
        while current.parent is not None:
            current.number_visits += 1
            if current.game.player == 1: # same as current.parent.game.player = 0
                current.total_value += (1*value_estimate) # value estimate +1 = white win
            elif current.game.player == 0: # same as current.parent.game.player = 1
                current.total_value += (-1*value_estimate)
            current = current.parent
        

class DummyNode(object):
    def __init__(self):
        self.parent = None
        self.child_total_value = defaultdict(float)
        self.child_number_visits = defaultdict(float)


# def UCT_search(game_state, num_reads,net):
#     # Создаём корень дерева поиска с текущим состоянием доски game_state.
#     root = UCTNode(game_state, move=None, parent=DummyNode())
#     for i in range(num_reads):
#         leaf = root.select_leaf()
#         encoded_s = ed.encode_board(leaf.game); encoded_s = encoded_s.transpose(2,0,1)
#         device = next(net.parameters()).device
#         encoded_s = torch.from_numpy(encoded_s).float().to(device)
#         child_priors, value_estimate = net(encoded_s)
#         child_priors = child_priors.detach().cpu().numpy().reshape(-1)
#         value_estimate = value_estimate.item()
#         if leaf.game.check_status() == True and leaf.game.in_check_possible_moves() == []: # if checkmate
#             leaf.backup(value_estimate) 
#             continue
#         leaf.expand(child_priors) # need to make sure valid moves
#         leaf.backup(value_estimate)
#     return np.argmax(root.child_number_visits), root


def UCT_search_batched(game_state, num_reads, net, batch_size=32):
    # root and device
    root = UCTNode(game_state, move=None, parent=DummyNode())
    device = next(net.parameters()).device
    net.eval()

    leaves_to_eval = []   # list of UCTNode
    leaves_idx = []       # indices order to apply results after forward

    def flush_batch():
        if not leaves_to_eval:
            return
        # build batch tensor (C,H,W) already in encode_board -> transpose done as in original
        batch_tensors = []
        for leaf in leaves_to_eval:
            encoded_s = ed.encode_board(leaf.game).transpose(2,0,1)
            batch_tensors.append(encoded_s)
        batch_np = np.stack(batch_tensors, axis=0).astype(np.float32)  # (B,C,H,W)
        batch = torch.from_numpy(batch_np).to(device, non_blocking=True)
        with torch.no_grad():
            child_priors_batch, value_batch = net(batch)  # assume net handles batched input
            # child_priors_batch: (B,4672) ; value_batch: (B,1) or (B)
        child_priors_batch = child_priors_batch.detach().cpu().numpy()
        value_batch = value_batch.detach().cpu().numpy().reshape(-1)

        # apply outputs to corresponding leaves (in same order)
        for leaf, priors, val in zip(leaves_to_eval, child_priors_batch, value_batch):
            # handle terminal: if checkmate then backup without expand (same as original)
            if leaf.game.check_status() == True and leaf.game.in_check_possible_moves() == []:
                leaf.backup(float(val))
            else:
                # expand: mask only legal moves quickly
                action_idxs = []
                for action in leaf.game.actions():
                    if action != []:
                        i_pos, f_pos, underprom = action
                        action_idxs.append(ed.encode_action(leaf.game, i_pos, f_pos, underprom))
                if action_idxs == []:
                    leaf.is_expanded = False
                else:
                    c_p = np.zeros_like(priors, dtype=np.float32)  # mask others zero
                    c_p[action_idxs] = priors[action_idxs]
                    # root noise only for root node (same rule)
                    if leaf.parent.parent is None:
                        c_p = leaf.add_dirichlet_noise(action_idxs, c_p)
                    leaf.expand(c_p)  # sets child_priors inside node
                    leaf.backup(float(val))
        # clear lists
        leaves_to_eval.clear()
        leaves_idx.clear()

    for i in range(num_reads):
        leaf = root.select_leaf()
        # collect leaf for batched eval
        leaves_to_eval.append(leaf)
        leaves_idx.append(i)
        # flush when batch full
        if len(leaves_to_eval) >= batch_size:
            flush_batch()

    # flush leftovers
    flush_batch()
    return np.argmax(root.child_number_visits), root


def do_decode_n_move_pieces(board,move):
    i_pos, f_pos, prom = ed.decode_action(board,move)
    for i, f, p in zip(i_pos,f_pos,prom):
        board.move_piece(i,f,p) # move piece to get next board state s
        a,b = i; c,d = f
        if board.current_board[c,d] in ["K","k"] and abs(d-b) == 2: # if king moves 2 squares, then move rook too for castling
            if a == 7 and d-b > 0: # castle kingside for white
                board.player = 0
                board.move_piece((7,7),(7,5),None)
            if a == 7 and d-b < 0: # castle queenside for white
                board.player = 0
                board.move_piece((7,0),(7,3),None)
            if a == 0 and d-b > 0: # castle kingside for black
                board.player = 1
                board.move_piece((0,7),(0,5),None)
            if a == 0 and d-b < 0: # castle queenside for black
                board.player = 1
                board.move_piece((0,0),(0,3),None)
    return board

def get_policy(root):
    policy = np.zeros(4672, dtype=np.float32) #вектор распределения вероятностей для всех ходов
    for idx in np.where(root.child_number_visits!=0)[0]:
        this_move_used_count = root.child_number_visits[idx]
        total_simulations = root.child_number_visits.sum()
        policy[idx] = this_move_used_count / total_simulations
    return policy






def append_selfplay_h5(h5_path, game_states):        
    if not os.path.exists(h5_path):
        # создаём новый файл с расширяемыми массивами
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('s', data=np.array(game_states['s'], dtype=np.float32),
                             maxshape=(None,) + np.array(game_states['s']).shape[1:])
            f.create_dataset('p', data=np.array(game_states['p'], dtype=np.float32),
                             maxshape=(None, 4672))
            f.create_dataset('v', data=np.array(game_states['v'], dtype=np.float32),
                             maxshape=(None,))
    else:
        with h5py.File(h5_path, 'a') as f:
            for key, dtype in zip(['s','p','v'], [np.float32, np.float32, np.float32]):
                data = np.array(game_states[key], dtype=dtype)
                dset = f[key]
                old_len = dset.shape[0]
                dset.resize(old_len + len(data), axis=0)
                dset[old_len:] = data



def MCTS_self_play(chessnet,num_games, simulation_depth, max_moves, dataset_path, log_path, use_mlflow):    
    
    for idxx in range(0,num_games):
        print("Game:",idxx + 1, '\n')
        with open(log_path, "a") as f:
                f.write(
                    f"\nGame: {idxx}\n"
                )
        # запускаем игру
        current_board = c_board() #init доски 
        checkmate = False 

        
        dataset = [] # to get state, policy, value for neural network training
        # states = []
        states = Counter()
        value = 0
        while checkmate == False and current_board.move_count <= max_moves:
            # draw_counter = 0
            # for s in states:                                
            #     if np.array_equal(current_board.current_board,s):
            #         draw_counter += 1                    

            # if draw_counter == 3:                
            #     break

            board_tuple = tuple(current_board.current_board.flatten())

            states[board_tuple] += 1
            if states[board_tuple] == 3:
                # троекратное повторение ходов
                break
                            
            
            # states.append(copy.deepcopy(current_board.current_board))                        
            # энкодим доску
            board_state = copy.deepcopy(ed.encode_board(current_board))                    
            best_move, root = UCT_search_batched(current_board, simulation_depth, chessnet, simulation_depth, simulation_depth//2) 
            # ходим и обновляем состояние доски
            current_board = do_decode_n_move_pieces(current_board,best_move) 
            # получаем вектор распределения вероятностей ходов 
            policy = get_policy(root) 



            dataset.append([board_state,policy])                                    
            # print(current_board.current_board,current_board.move_count)
            # print(" ")
            with open(log_path, "a") as f:
                f.write(
                    f"Board:\n{current_board.current_board}\nMove count: {current_board.move_count}\n\n"
                )
            

            

            

            
            if current_board.check_status() == True and current_board.in_check_possible_moves() == []: # checkmate
                if current_board.player == 0: # black wins
                    value = -1
                elif current_board.player == 1: # white wins
                    value = 1
                checkmate = True
        
        

        if use_mlflow:
            mlflow.log_artifact(log_path, artifact_path="logs")

        game_states = {'s': [], 'p': [], 'v': []}
        for idx,data in enumerate(dataset):
            s,p = data
            game_states['s'].append(s)
            game_states['p'].append(p)
            if idx == 0:                
                game_states['v'].append(0)                
            else:
                game_states['v'].append(value)                       
        
        del dataset 

        append_selfplay_h5(dataset_path, game_states)
        
        
        


