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
# EVAL_NUM_GAMES = 10
# SIMULATION_DEPTH = 666
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
        dataset_path=os.path.join(absoute_path, 'datasets'),
        eval_path=os.path.join(absoute_path, 'model_games'),
        supervised_source_path=os.path.join(absoute_path, 'pretrain.csv'),
        supervised_dest_path=os.path.join(absoute_path, 'supervised_data'), 
        sl=True,       
        rl=False,
        ):
    
    os.makedirs(supervised_dest_path, exist_ok=True)
    os.makedirs(eval_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(dataset_path, exist_ok=True)    

    best_net_filename = os.path.join(save_path,\
                                            f"best_net_trained8.pth.tar")        
    train_data, val_data, test_data = preprocess_data(supervised_source_path, supervised_dest_path, seed)


    # supervised pretraining - делаем best_net изначальную 
    if sl:            
        net = ChessNet() 
        net.train()
        train(
            net=net,
            train_data=train_data,
            val_data=val_data,
            epochs=epochs,
            seed=seed,
            save_path=save_path
        )

        torch.save({'state_dict': net.state_dict()}, best_net_filename)
    else:
        net = ChessNet() # если supervised learning отключено, то просто чтоб не сломалось заливаем веса рандомной сети
        torch.save({'state_dict': net.state_dict()}, best_net_filename)
    
    
    



    
    # reinforce learning
    if rl:
        for iteration in range(iterations): 
            # Runs MCTS
            net = ChessNet() # инициализируем сетку        

            #ставим на cuda
            cuda = torch.cuda.is_available() 
            if cuda:
                net.cuda()

            # if iteration > 0:
                # подгружаем веса с предыдущей итерации                                    
                # current_net_filename = os.path.join(save_path,\
                #                                 f"current_net_trained8_iter{iteration-1}.pth.tar")                        
                # checkpoint = torch.load(current_net_filename)
            checkpoint = torch.load(best_net_filename) # для нулевой итерации здесь просто будет supervised модель
            net.load_state_dict(checkpoint['state_dict'])
                    
            net.eval() #замораживаем веса сети переводим в инференес                        
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
            
            # обучаем на играх с самой сабой
            train(
                net=net,
                train_data=game_states,
                val_data=None,
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
    test(net=best_net, test_data=test_data, seed=seed)
        
                

if __name__=="__main__":
    run_pipeline()