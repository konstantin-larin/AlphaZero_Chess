#!/usr/bin/env python

from alpha_net import ChessNet, train 
from MCTS_chess import MCTS_self_play 
import os 
import pickle 
import numpy as np
import torch
import torch.multiprocessing as mp
# evaluator пока не применяется


def run_pipeline(
        ITERATIONS=10, NUM_GAMES=50, SIMULATION_DEPTH=666, SEED=42, EPOCHS=200,
        save_path='./model_data/', dataset_path='./datasets/'
        ):
    for iteration in range(ITERATIONS): # запускаем 10 итераций
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
        MCTS_self_play(net, NUM_GAMES, iteration, SIMULATION_DEPTH, dataset_path=dataset_path)
        # gather datasets
        datasets = []        
        for j in range(iteration, -1, -1):
            data_path = os.path.join(dataset_path, f"iter{j}")
            for idx, file in enumerate(os.listdir(data_path)):
                filename = os.path.join(data_path, file)
                with open(filename, 'rb') as fo:
                    datasets.extend(pickle.load(fo, encoding='bytes'))
                    
        
        datasets = np.array(datasets)                

        net.train()
        print("learn mistakes")
        current_net_filename = os.path.join(save_path,\
                                        f"current_net_trained8_iter{iteration}.pth.tar")        
        
        train(net,datasets,EPOCHS,SEED, save_path=save_path)
        # processes2 = []
        # for i in range(5):
        #     p2 = mp.Process(target=train,args=(net,datasets,0,200,i))
        #     p2.start()
        #     processes2.append(p2)
        # for p2 in processes2:
        #     p2.join()
        # save results
        torch.save({'state_dict': net.state_dict()}, current_net_filename)



# if __name__=="__main__":
#     for iteration in range(ITERATIONS): # запускаем 10 итераций
#         # Runs MCTS
#         net = ChessNet() # инициализируем сетку        

#         #ставим на cuda
#         cuda = torch.cuda.is_available() 
#         if cuda:
#             net.cuda()

#         if iteration > 0:
#             # подгружаем веса с предыдущей итерации                                    
#             current_net_filename = os.path.join("./model_data/",\
#                                             f"current_net_trained8_iter{iteration-1}.pth.tar")        
#             checkpoint = torch.load(current_net_filename)
#             net.load_state_dict(checkpoint['state_dict'])


        
#         net.eval() #замораживаем веса сети переводим в инференес
        
        
#         # играем этой моделью, собираем датасет 
#         # processes1 = []
#         # for i in range(5):
#         #     #добавил сюда iteration чтоб dataset сохранялся для каждой итерации
#         #     p1 = mp.Process(target=MCTS_self_play,args=(net,NUM_GAMES,i, iteration, SIMULATION_DEPTH)) 
#         #     p1.start()
#         #     processes1.append(p1)
#         # for p1 in processes1:
#         #     p1.join()
#         MCTS_self_play(net, NUM_GAMES, iteration, SIMULATION_DEPTH)
#         # gather datasets
#         datasets = []        
#         for j in range(iteration, -1, -1):
#             data_path = f'./datasets/iter{j}'
#             for idx, file in enumerate(os.listdir(data_path)):
#                 filename = os.path.join(data_path, file)
#                 with open(filename, 'rb') as fo:
#                     datasets.extend(pickle.load(fo, encoding='bytes'))
                    
        
#         datasets = np.array(datasets)                

#         net.train()
#         print("learn mistakes")
#         current_net_filename = os.path.join("./model_data/",\
#                                         f"current_net_trained8_iter{iteration}.pth.tar")        
        
#         train(net,datasets,EPOCHS,SEED)
#         # processes2 = []
#         # for i in range(5):
#         #     p2 = mp.Process(target=train,args=(net,datasets,0,200,i))
#         #     p2.start()
#         #     processes2.append(p2)
#         # for p2 in processes2:
#         #     p2.join()
#         # save results
#         torch.save({'state_dict': net.state_dict()}, current_net_filename)