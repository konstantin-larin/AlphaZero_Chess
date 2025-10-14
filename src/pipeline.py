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
import mlflow
absoute_path = r'C:\Users\konst\Desktop\workflow\chess\AlphaZero_Chess\src'

def get_or_create_experiment(experiment_name):    
    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)            

def run_pipeline(    
        mlflow_uri="http://localhost:5000/",
        experiment_name='alpha_zero',

        model_params = {
            "res_blocks_num": 19,
            "conv_planes": 256, 
            "conv_kernel_size": 3,
            "conv_stride": 1,
            "conv_padding": 1,
            "res_inplanes": 256, 
            "res_planes": 256, 
            "res_kernel_size": 3, 
            "res_stride": 1, 
            "res_padding": 1,
            "value_hidden_dim": 64,
            "policy_hidden_dim": 128,
        },

        sl_params = {                        
            "train_epochs": 2,
            "adam_lr": 3e-3,
            "scheduler_gamma": 0.2,
        },

        rl_params = {            
            "train_epochs": 2, 
            "adam_lr": 3e-3,
            "scheduler_gamma": 0.2,
            "iterations": 2, 
            "num_games": 3,
            "eval_num_games": 3,
            "max_moves": 5,
            "simulation_depth": 10,
        },

        sl=True,
        rl=True,
        is_debug=True,       
        batch_size=64,  
        seed=42,
        
        save_path=os.path.join(absoute_path, 'model_data'),
        selfplay_data_path=os.path.join(absoute_path, 'selfplay_data.h5'),
        eval_path=os.path.join(absoute_path, 'evaluation_data'),
        supervised_source_path=os.path.join(absoute_path, 'pretrain.csv'),
        supervised_dest_path=os.path.join(absoute_path, 'supervised_data'),                         
        ):
    

    # create experiment
    cuda = torch.cuda.is_available() 
    torch.backends.cudnn.benchmark = True  
    
    # mlflow.set_tracking_uri(mlflow_uri)        
    # experiment_id = get_or_create_experiment(experiment_name)
    # mlflow.set_experiment(experiment_id=experiment_id)
    

    print('starting pipeline')
    os.makedirs(supervised_dest_path, exist_ok=True)
    os.makedirs(eval_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    

    best_net_filename = os.path.join(save_path,\
                                            f"best_net_trained8.pth.tar")           
    train_path, val_path, test_path = preprocess_data(supervised_source_path, supervised_dest_path, seed)    


    # supervised pretraining - делаем best_net изначальную 
    if sl:                    
        net = ChessNet(name='base_supervised', **model_params) # инициализируем сетку
        
        print('supervised learning')
        if cuda:
            net.cuda()
        net.train()        
        train(
            batch_size=batch_size,
            net=net,
            train_datapath=train_path,            
            val_datapath=val_path,
            epochs=sl_params['train_epochs'],
            adam_lr=sl_params['adam_lr'],
            scheduler_gamma=sl_params['scheduler_gamma'],                        
            seed=seed,
            save_path=save_path,
            is_debug=is_debug
        )        

        torch.save({'state_dict': net.state_dict()}, best_net_filename)
    else:
        net = ChessNet(**model_params) # если supervised learning отключено, то просто чтоб не сломалось заливаем веса рандомной сети
        torch.save({'state_dict': net.state_dict()}, best_net_filename)
    
    
    



    
    # reinforce learning
    if rl:
        for iteration in range(rl_params['iterations']): 
            # Runs MCTS
            net = ChessNet(name=f'chessnet_iteration_{iteration}', **model_params) # инициализируем сетку        

            #ставим на cuda            
            if cuda:
                net.cuda()
    
            checkpoint = torch.load(best_net_filename) # для нулевой итерации здесь просто будет supervised модель
            net.load_state_dict(checkpoint['state_dict'])
                    
            net.eval()
            MCTS_self_play(net, rl_params['num_games'], rl_params['simulation_depth'], 
                           rl_params['max_moves'], dataset_path=selfplay_data_path)
            
            


            net.train()
            print("learn mistakes")
            current_net_filename = os.path.join(save_path,\
                                            f"current_net_trained8_iter{iteration}.pth.tar")        
            
            # обучаем на играх с самой собой
            train(
                net=net,
                train_datapath=selfplay_data_path,
                val_datapath=None,                
                epochs=rl_params['train_epochs'],
                batch_size=batch_size,
                adam_lr=rl_params['adam_lr'],                                
                scheduler_gamma=rl_params['scheduler_gamma'],                                
                seed=seed,
                save_path=save_path,
                is_debug=is_debug
            )            
            # save results
            torch.save({'state_dict': net.state_dict()}, current_net_filename)
            print(f"Saved current net to {current_net_filename}")


            print('evaluate current net vs best net')
            best_net = ChessNet(**model_params) 
            best_net.load_state_dict(checkpoint['state_dict'])
            if cuda:
                best_net.cuda()
            best_net.eval()
            net.eval()
            arena = Arena(current_chessnet=net, best_chessnet=best_net, 
                          max_moves=rl_params['max_moves'], 
                          simulation_depth=rl_params['simulation_depth'], dataset_path=eval_path)            

            if best_net == arena.evaluate(num_games=rl_params['eval_num_games']):
                print('best net is still best')
            else:
                torch.save({'state_dict': net.state_dict()},best_net_filename)            
                best_net = net

            
    # test best_net      
              
    test(net=best_net, batch_size=batch_size, test_datapath=test_path, seed=seed, is_debug=is_debug)
        
                

if __name__=="__main__":
    run_pipeline()