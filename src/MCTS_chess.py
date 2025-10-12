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


def UCT_search(game_state, num_reads,net):
    # Создаём корень дерева поиска с текущим состоянием доски game_state.
    root = UCTNode(game_state, move=None, parent=DummyNode())
    for i in range(num_reads):
        leaf = root.select_leaf()
        encoded_s = ed.encode_board(leaf.game); encoded_s = encoded_s.transpose(2,0,1)
        device = next(net.parameters()).device
        encoded_s = torch.from_numpy(encoded_s).float().to(device)
        child_priors, value_estimate = net(encoded_s)
        child_priors = child_priors.detach().cpu().numpy().reshape(-1)
        value_estimate = value_estimate.item()
        if leaf.game.check_status() == True and leaf.game.in_check_possible_moves() == []: # if checkmate
            leaf.backup(value_estimate) 
            continue
        leaf.expand(child_priors) # need to make sure valid moves
        leaf.backup(value_estimate)
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


# def load_pickle(filename):
#     completeName = os.path.join("./datasets/",\
#                                 filename)
#     with open(completeName, 'rb') as pkl_file:
#         data = pickle.load(pkl_file)
#     return data



def save_as_pickle(completeName, data): 
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)


def MCTS_self_play(chessnet,num_games, simulation_depth, max_moves, dataset_path='./datasets/iter0'):
    os.makedirs(dataset_path, exist_ok=True)
    for idxx in range(0,num_games):
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
            best_move, root = UCT_search(current_board,simulation_depth,chessnet) 
            # ходим и обновляем состояние доски
            current_board = do_decode_n_move_pieces(current_board,best_move) 
            # получаем вектор распределения вероятностей ходов 
            policy = get_policy(root) 



            dataset.append([board_state,policy])                        
            print(current_board.current_board,current_board.move_count); 
            print(" ")
            

            
            if current_board.check_status() == True and current_board.in_check_possible_moves() == []: # checkmate
                if current_board.player == 0: # black wins
                    value = -1
                elif current_board.player == 1: # white wins
                    value = 1
                checkmate = True
        
        
        dataset_p = []
        for idx,data in enumerate(dataset):
            s,p = data
            if idx == 0:
                dataset_p.append([s,p,0])
            else:
                dataset_p.append([s,p,value])
        del dataset 
        
        
        save_as_pickle(
            os.path.join(dataset_path, "dataset_%i_%s" % (idxx, datetime.datetime.today().strftime("%Y-%m-%d"))),
            dataset_p,)



# if __name__=="__main__":
    
#     net_to_play="current_net_trained8_iter1.pth.tar"
#     mp.set_start_method("spawn",force=True)
#     net = ChessNet()
#     cuda = torch.cuda.is_available()
#     if cuda:
#         net.cuda()
#     net.share_memory()
#     net.eval()
#     print("hi")
#     #torch.save({'state_dict': net.state_dict()}, os.path.join("./model_data/",\
#     #                                "current_net.pth.tar"))
    
#     current_net_filename = os.path.join("./model_data/",\
#                                     net_to_play)
#     checkpoint = torch.load(current_net_filename)
#     net.load_state_dict(checkpoint['state_dict'])
#     processes = []
#     for i in range(5):
#         p = mp.Process(target=MCTS_self_play,args=(net,50,i))
#         p.start()
#         processes.append(p)
#     for p in processes:
#         p.join()
