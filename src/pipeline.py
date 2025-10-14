#!/usr/bin/env python

from alpha_net import ChessNet, train, test
from MCTS_chess import MCTS_self_play 
import os 
import pickle 
import numpy as np
import torch
import torch.multiprocessing as mp
from evaluator import Arena
# добавляем supervised обучение
from supervised import preprocess_data 

# ITERATIONS = 10
# NUM_GAMES = 50
# MAX_MOVES = 100
# EVAL_NUM_GAMES = 3
# SIMULATION_DEPTH = 1000
# SEED = 42
# EPOCHS = 200

# for quick test
ITERATIONS = 2
NUM_GAMES = 2
EVAL_NUM_GAMES = 2
MAX_MOVES = 3
SIMULATION_DEPTH = 10
SEED = 42
EPOCHS = 2
absoute_path = r'C:\Users\konst\Desktop\workflow\chess\AlphaZero_Chess\src'


def run_pipeline(
        iterations=ITERATIONS, 
        num_games=NUM_GAMES,
        eval_num_games=EVAL_NUM_GAMES,
        max_moves=MAX_MOVES,
        simulation_depth=SIMULATION_DEPTH,
        seed=SEED,
        epochs=EPOCHS,
        save_path=os.path.join(absoute_path, 'model_data'),
        selfplay_data_path=os.path.join(absoute_path, 'selfplay_data.h5'),
        eval_path=os.path.join(absoute_path, 'evaluation_data'),
        supervised_source_path=os.path.join(absoute_path, 'pretrain.csv'),
        supervised_dest_path=os.path.join(absoute_path, 'supervised_data'), 
        sl=False,       
        rl=True,        
        ):
    
    os.makedirs(supervised_dest_path, exist_ok=True)
    os.makedirs(eval_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    

    best_net_filename = os.path.join(save_path,\
                                            f"best_net_trained8.pth.tar")        
    print('start')
    train_path, val_path, test_path = preprocess_data(supervised_source_path, supervised_dest_path, seed)




    # supervised pretraining - делаем best_net изначальную 
    if sl:                    
        net = ChessNet('base_supervised') # инициализируем сетку
        print('supervised learning')
        
        net.train()        
        train(
            net=net,
            train_datapath=train_path,
            val_datapath=val_path,
            epochs=epochs,
            seed=seed,
            save_path=save_path
        )        

        torch.save({'state_dict': net.state_dict()}, best_net_filename)
    else:
        net = ChessNet('just_net') # если supervised learning отключено, то просто чтоб не сломалось заливаем веса рандомной сети
        torch.save({'state_dict': net.state_dict()}, best_net_filename)
    
    
    



    
    # reinforce learning
    if rl:
        for iteration in range(iterations): 
            # Runs MCTS
            net = ChessNet(f'chessnet_iteration_{iteration}') # инициализируем сетку        

            #ставим на cuda
            cuda = torch.cuda.is_available() 
            if cuda:
                net.cuda()
    
            checkpoint = torch.load(best_net_filename) # для нулевой итерации здесь просто будет supervised модель
            net.load_state_dict(checkpoint['state_dict'])
                    
            net.eval()
            MCTS_self_play(net, num_games, simulation_depth, max_moves, dataset_path=selfplay_data_path)
            
            


            net.train()
            print("learn mistakes")
            current_net_filename = os.path.join(save_path,\
                                            f"current_net_trained8_iter{iteration}.pth.tar")        
            
            # обучаем на играх с самой сабой
            train(
                net=net,
                train_datapath=selfplay_data_path,
                val_datapath=None,                
                epochs=epochs,
                seed=seed,
                save_path=save_path
            )            
            # save results
            torch.save({'state_dict': net.state_dict()}, current_net_filename)
            print(f"Saved current net to {current_net_filename}")


            print('evaluate current net vs best net')
            best_net = ChessNet() 
            best_net.load_state_dict(checkpoint['state_dict'])
            if cuda:
                best_net.cuda()
            best_net.eval()
            net.eval()
            arena = Arena(current_chessnet=net, best_chessnet=best_net, max_moves=max_moves, simulation_depth=simulation_depth, dataset_path=eval_path)            

            if best_net == arena.evaluate(num_games=eval_num_games):
                print('best net is still best')
            else:
                torch.save({'state_dict': net.state_dict()},best_net_filename)            
                best_net = net

            
    # test best_net                
    test(net=best_net, test_datapath=test_path, seed=seed)
        
                

if __name__=="__main__":
    run_pipeline()