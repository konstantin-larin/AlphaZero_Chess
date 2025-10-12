#!/usr/bin/env python

from alpha_net import ChessNet, train 
from MCTS_chess import MCTS_self_play 
import os 
import pickle 
import numpy as np
import torch
import torch.multiprocessing as mp
# evaluator пока не применяется

ITERATIONS = 10
NUM_GAMES = 50
MAX_MOVES = 100
SIMULATION_DEPTH = 666
SEED = 42
EPOCHS = 200

# for quick test
# ITERATIONS = 2
# NUM_GAMES = 2
# MAX_MOVES = 3
# SIMULATION_DEPTH = 10
# SEED = 42
# EPOCHS = 2

def run_pipeline(
        iterations=ITERATIONS, 
        num_games=NUM_GAMES,
        max_moves=MAX_MOVES,
        simulation_depth=SIMULATION_DEPTH,
        seed=SEED,
        epochs=EPOCHS,
        save_path='./model_data/', dataset_path='./datasets/'
        ):
    
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(dataset_path, exist_ok=True)    

    for iteration in range(iterations): # запускаем 10 итераций        
        # Runs MCTS
        net = ChessNet() # инициализируем сетку        

        #ставим на cuda
        cuda = torch.cuda.is_available() 
        if cuda:
            net.cuda()

        if iteration > 0:
            # подгружаем веса с предыдущей итерации                                    
            current_net_filename = os.path.join(save_path,\
                                            f"current_net_trained8_iter{iteration-1}.pth.tar")        
            checkpoint = torch.load(current_net_filename)
            net.load_state_dict(checkpoint['state_dict'])


        
        net.eval() #замораживаем веса сети переводим в инференес
        
        
        # играем этой моделью, собираем датасет 
        # processes1 = []
        # for i in range(5):
        #     #добавил сюда iteration чтоб dataset сохранялся для каждой итерации
        #     p1 = mp.Process(target=MCTS_self_play,args=(net,NUM_GAMES,i, iteration, SIMULATION_DEPTH)) 
        #     p1.start()
        #     processes1.append(p1)
        # for p1 in processes1:
        #     p1.join()
        MCTS_self_play(net, num_games, simulation_depth, max_moves, dataset_path=os.path.join(dataset_path, f"iter{iteration}"))
        # gather datasets
        game_states = {
            's': [],
            'p': [],
            'v': []
        }
        for j in range(iteration, -1, -1):
            data_path = os.path.join(dataset_path, f"iter{j}")
            for idx, file in enumerate(os.listdir(data_path)):
                filename = os.path.join(data_path, file)
                with open(filename, 'rb') as fo:
                    _game_states = pickle.load(fo, encoding='bytes')                    
                    game_states['s'].extend(_game_states['s'])
                    game_states['p'].extend(_game_states['p'])
                    game_states['v'].extend(_game_states['v'])                    
                            

        net.train()
        print("learn mistakes")
        current_net_filename = os.path.join(save_path,\
                                        f"current_net_trained8_iter{iteration}.pth.tar")        
        
        train(net,game_states,epochs,seed, save_path=save_path)
        # save results
        torch.save({'state_dict': net.state_dict()}, current_net_filename)



if __name__=="__main__":
    run_pipeline()