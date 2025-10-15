#!/usr/bin/env python

from alpha_net import ChessNet, train, test
from MCTS_chess import MCTS_self_play 
import os 
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
        use_mlflow=False,
        mlflow_params={
            "uri": "http://localhost:5000/",
            "experiment_name": 'AlphaZero',
            'run_name': "sl + rl",            
        },
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
        num_heads=5,
        seed=42,

        log_path=os.path.join(absoute_path, 'logs.txt'),
        save_path=os.path.join(absoute_path, 'model_data'),        
        selfplay_data_path=os.path.join(absoute_path, 'selfplay_data.h5'),
        eval_path=os.path.join(absoute_path, 'evaluation_data'),
        supervised_source_path=os.path.join(absoute_path, 'pretrain.csv'),
        supervised_dest_path=os.path.join(absoute_path, 'supervised_data'),                         
        ):
    
    
    try:
        # create experiment    
        if use_mlflow:
            mlflow.set_tracking_uri(mlflow_params['uri'])        
            experiment_id = get_or_create_experiment(mlflow_params['experiment_name'])
            mlflow.set_experiment(experiment_id=experiment_id)
            mlflow.start_run(run_name=mlflow_params['run_name'])                    


            if sl and rl:
                mlflow.log_params({**model_params, **sl_params, **rl_params})                    

            elif sl:
                mlflow.log_params({**model_params, **sl_params})        
            elif rl:
                mlflow.log_params({**model_params, **rl_params})        

        


        cuda = torch.cuda.is_available() 
        torch.backends.cudnn.benchmark = True      
            
        print('starting pipeline')
        os.makedirs(supervised_dest_path, exist_ok=True)
        os.makedirs(eval_path, exist_ok=True)
        os.makedirs(save_path, exist_ok=True)


        

        if os.path.exists(log_path):
            with open(log_path, "w") as f:
                pass  # очищаем

        

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
            if use_mlflow:
                mlflow.start_run(run_name=f"Supervised_Learning", nested=True)     
            train(
                batch_size=batch_size,
                net=net,
                use_mlflow=use_mlflow,
                train_datapath=train_path,            
                val_datapath=val_path,
                epochs=sl_params['train_epochs'],
                adam_lr=sl_params['adam_lr'],
                scheduler_gamma=sl_params['scheduler_gamma'],                        
                seed=seed,
                save_path=save_path,
                is_debug=is_debug,            
            )        
            torch.save({'state_dict': net.state_dict()}, best_net_filename)
            if use_mlflow:
                mlflow.pytorch.log_model(net, name='model')
                mlflow.end_run()                

        else:
            net = ChessNet(**model_params) # если supervised learning отключено, то просто чтоб не сломалось заливаем веса рандомной сети
            torch.save({'state_dict': net.state_dict()}, best_net_filename)
        
        
        



        
        # reinforce learning
        if rl:       
            if use_mlflow:
                mlflow.start_run(run_name='Reinforcement Learning', nested=True) 

            for iteration in range(rl_params['iterations']):                             
                if use_mlflow:
                    mlflow.start_run(run_name=f'Iteration_{iteration}', nested=True)

                print(f'ITERATION {iteration + 1}')
                mp.set_start_method("spawn",force=True)

                # Runs MCTS
                net = ChessNet(name=f'chessnet_iteration_{iteration}', **model_params) # инициализируем сетку        

                #ставим на cuda            
                if cuda:
                    net.cuda()
        
                checkpoint = torch.load(best_net_filename) # для нулевой итерации здесь просто будет supervised модель
                net.load_state_dict(checkpoint['state_dict'])
                        
                net.share_memory()
                net.eval()    

                processes = []                            

                for i in range(num_heads):
                    p = mp.Process(target=MCTS_self_play, args=(
                        net, rl_params['num_games'], rl_params['simulation_depth'], 
                        rl_params['max_moves'], selfplay_data_path, log_path, use_mlflow
                    ))
                    p.start()
                    processes.append(p)
                for p in processes:
                    p.join()                
                
                
                net.train()
                print("learn mistakes")
                current_net_filename = os.path.join(save_path,\
                                                f"current_net_trained8_iter{iteration}.pth.tar")        
                
                # обучаем на играх с самой собой
                train(
                    net=net,
                    train_datapath=selfplay_data_path,
                    val_datapath=val_path,                
                    epochs=rl_params['train_epochs'],
                    batch_size=batch_size,
                    adam_lr=rl_params['adam_lr'],                                
                    scheduler_gamma=rl_params['scheduler_gamma'],                                
                    seed=seed,
                    save_path=save_path,
                    is_debug=is_debug,
                    use_mlflow=use_mlflow
                )            
                # save results
                torch.save({'state_dict': net.state_dict()}, current_net_filename)
                if use_mlflow:
                    mlflow.pytorch.log_model(net, name='model')
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

                if best_net == arena.evaluate(num_games=rl_params['eval_num_games'], use_mlflow=use_mlflow):
                    print('best net is still best')
                else:
                    torch.save({'state_dict': net.state_dict()},best_net_filename)            
                    best_net = net
                if use_mlflow: 
                    mlflow.end_run()        
            if use_mlflow:
                mlflow.end_run()

                
        # test best_net      
        
        avg_loss, accuracy = test(net=best_net, batch_size=batch_size, test_datapath=test_path, seed=seed, is_debug=is_debug)        
        if use_mlflow:
            mlflow.log_metrics({
                'test_loss': avg_loss,
                'accuracy': accuracy,
            })
            mlflow.end_run()        
    except KeyboardInterrupt:
        print("\KeyboardInterrupt detected. Stopping gracefully...")
    finally:
        # Закрываем все открытые run'ы
        if use_mlflow:
            while mlflow.active_run() is not None:    
                print("Closing active MLflow run...")
                mlflow.end_run()


        
                

if __name__=="__main__":
    run_pipeline()

