import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from chess_board import board
from MCTS_chess import do_decode_n_move_pieces
import os
import encoder_decoder as ed
from tqdm import tqdm
import pickle
import h5py

# меняем только init
class FenBoard(board):        
    def __init__(self, fen):
        super().__init__()  # Инициализируем родительский класс для всех атрибутов
        parts = fen.split()
        piece_placement, active_color, castling, en_passant, halfmove, fullmove = parts
        
        # Парсим позиции фигур
        self.current_board = np.zeros([8,8], dtype=str)
        rows = piece_placement.split('/')
        for i, row in enumerate(rows):
            j = 0
            for char in row:
                if char.isdigit():
                    j += int(char)
                else:
                    self.current_board[i, j] = char
                    j += 1
        self.current_board[self.current_board == ""] = " "  # пустые клетки

        # Очередь хода
        self.player = 0 if active_color == "w" else 1

        # Рокировка
        self.K_move_count = 0 if 'K' in castling else 1
        self.R1_move_count = 0 if 'Q' in castling else 1
        self.R2_move_count = 0 if 'K' in castling else 1
        self.k_move_count = 0 if 'k' in castling else 1
        self.r1_move_count = 0 if 'q' in castling else 1
        self.r2_move_count = 0 if 'k' in castling else 1

        # En passant
        if en_passant != '-':
            file_to_index = {'a':0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6,'h':7}
            self.en_passant = file_to_index[en_passant[0]]
            self.en_passant_move = int(fullmove) - 1  # приблизительно
        else:
            self.en_passant = -999
            self.en_passant_move = 0

        # Полный и полупроходные ходы
        self.no_progress_count = int(halfmove)
        self.move_count = int(fullmove)
   

# для encode pgn
def uci_to_indices(move):
    # Преобразует строку хода 'h5f3' в координаты ((i, j), (x, y)).
    def file_to_col(f): return ord(f) - ord('a')
    def rank_to_row(r): return 8 - int(r)   

    from_sq = move[0:2]
    to_sq = move[2:4]  
    underpromote = None
    if len(move) == 5: # underpromotion
        underpromote = move[4]
         
    i, j = rank_to_row(from_sq[1]), file_to_col(from_sq[0])
    x, y = rank_to_row(to_sq[1]), file_to_col(to_sq[0])
    
    return (i, j), (x, y), underpromote





def create_game_states(data, full_path, proba_of_right_move=0.5):
    # если уже есть — просто загружаем
    if os.path.exists(full_path):    
        print(f"[INFO] Loading existing file: {full_path}")
        return full_path  # просто возвращаем путь, сам файл читаем через h5py

    # создаём HDF5 файл с пустыми массивами и maxshape=None для append
    with h5py.File(full_path, 'w') as f:
        max_s_shape = (None,) + ed.encode_board(FenBoard(data.iloc[0]['fen'])).shape
        f.create_dataset('s', shape=(0,) + max_s_shape[1:], maxshape=(None,) + max_s_shape[1:], dtype=np.float32)
        f.create_dataset('p', shape=(0, 4672), maxshape=(None, 4672), dtype=np.float32)
        f.create_dataset('v', shape=(0,), maxshape=(None,), dtype=np.float32)

        for _, row in tqdm(data.iterrows(), total=len(data), desc="Processing games"):
            board = FenBoard(row['fen'])
            s = ed.encode_board(board)
            

            # uci → action
            initial_pos, final_pos, underpromote = uci_to_indices(row['move'])
            promo_map = {'q': 'queen', 'r': 'rook', 'b': 'bishop', 'n': 'knight'}
            underpromote = promo_map.get(underpromote, underpromote)

            action_index = ed.encode_action(board, initial_pos, final_pos, underpromote=underpromote)                      
            p = np.zeros(4672, dtype=np.float32)
            
            action_idxs = []
            for action in board.actions()  : # possible actions                
                if action != []:
                    initial_pos,final_pos,underpromote = action
                    action_idxs.append(ed.encode_action(board,initial_pos,final_pos,underpromote))            
                
            if len(action_idxs) > 1:
                p[action_idxs] = (1 - proba_of_right_move) / (len(action_idxs) - 1)
                p[action_index] = proba_of_right_move                                                        
            else:
                p[action_index] = 1.0
            

            # применяем ход
            board = do_decode_n_move_pieces(board, action_index)

            v = 0
            if board.check_status() and board.in_check_possible_moves() == []:  # checkmate
                v = 1 if board.player == 1 else -1

            # append к HDF5 массивам
            for key, value in zip(['s','p','v'], [s, p, v]):
                dset = f[key]
                dset.resize(dset.shape[0] + 1, axis=0)
                dset[-1] = value

    print(f"[INFO] Saved new HDF5 file: {full_path}")
    return full_path


def preprocess_data(source_path, dest_path, seed, proba_of_right_move=0.5):
    data = pd.read_csv(source_path)
    train, test = train_test_split(data, test_size=0.2, random_state=seed, shuffle=True)
    test, val = train_test_split(test, test_size=0.5, random_state=seed, shuffle=True)

    train_path = os.path.join(dest_path, 'train.h5')
    create_game_states(train, train_path, proba_of_right_move)
    del train

    val_path = os.path.join(dest_path, 'val.h5')
    create_game_states(val, val_path, proba_of_right_move)
    del val

    test_path = os.path.join(dest_path, 'test.h5')
    create_game_states(test, test_path, proba_of_right_move)
    del test

    return train_path, val_path, test_path


# def supervised_learning(train_path, val_path, test_path):

# if __name__ == "__main__":    
      
