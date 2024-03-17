import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt

total_data_path = '/home/jabblee/Desktop/CRC_collections/CRC_update/2023_Gatherings/'

# 총 자음 모음 수만큼의 sheet
sheet_name_lst = [('ㄱ', 1), ('ㄴ', 2), ('ㄷ', 3), ('ㄹ', 4), ('ㅁ', 5), ('ㅂ', 6), 
                  ('ㅅ', 7), ('ㅇ', 8), ('ㅈ', 9), ('ㅊ', 10), ('ㅋ', 11), ('ㅍ', 12), 
                  ('ㅎ', 13), ('ㅏ', 14), ('ㅑ', 15), ('ㅓ', 16), ('ㅕ', 17), 
                  ('ㅗ', 18), ('ㅛ', 19), ('ㅜ', 20), ('ㅡ', 21), ('ㅣ', 22)]

class GestureDataset(Dataset):
    def __init__(self, data_path, train = True):
        self.data_path = data_path
        self.train = train
        self.bending_min = 1227.0
        self.bending_max = 4095.0
        self.yaw_min = -179
        self.yaw_max = 179
        self.pitch_min = -82
        self.pitch_max = 84
        self.roll_min = -177
        self.roll_max = 179
        
        if self.train:
            self.data_path = [file for file in glob.glob(self.data_path + 'TRAIN/*', recursive = True)
                              if file.endswith('.npy')]
        else:
            self.data_path = [file for file in glob.glob(self.data_path + 'TEST/*', recursive = True)
                              if file.endswith('.npy')]
        
        # # This is for visualizing valid data distribution at row counts
        # rows_lst, cols_lst = [], []
        # print('Total Length of self.data_path :', len(self.data_path))
        # for data_path in self.data_path:
        #     raw_data = np.load(data_path, allow_pickle = True)
        #     row, column = raw_data.shape
        #     rows_lst.append(row)
        #     cols_lst.append(column)
        
        # rows_arr = np.array(rows_lst)
        # print('Mean cost of rows ndarray :', np.mean(rows_arr))
        # print('Median cost of rows ndarray :', np.median(rows_arr))
           
        # sns.displot(rows_lst, rug = True)
        # plt.show()
        
    def replace_non_standard_minus(self, s):
        if isinstance(s, str):
            return s.replace('\u2011', '-')
        else:
            return s
        
    def __getitem__(self, idx):
        # Load npy file
        npy_file_path = self.data_path[idx]
        person_name, class_label, chunk_number = npy_file_path.split('/')[-1].split('_')

        # Convert npy file to DataFrame for solving the problem of loading ndarray with object type.
        raw_data_np = np.load(npy_file_path, allow_pickle = True)
        
        # Solve the problem of not recognizing minus cost when I load npy file and make it ndarray
        replace_func = np.vectorize(self.replace_non_standard_minus, otypes = [float])
        raw_data_np = replace_func(raw_data_np)
        raw_data_np = raw_data_np.astype(float)
        
        real_rows, _ = raw_data_np.shape

        # # Make valid counts at each person's each hangul characters to fixed cost
        # # 320 is for train rows according to valid_train_cnts.png
        # # 60 is for test rows according to valid_test_cnts.png
        # # This is for making fixed shape of input datas
        # if self.train:
        #     if real_rows < 40:
        #         raw_data_np = np.append(raw_data_np, np.zeros((40 - real_rows, 9)).astype(np.float), axis = 0)
        #     else:
        #         raw_data_np = raw_data_np[:40, :]
        # else:
        #     if real_rows < 40:
        #         raw_data_np = np.append(raw_data_np, np.zeros((40 - real_rows, 9)).astype(np.float), axis = 0)
        #     else:
        #         raw_data_np = raw_data_np[:40, :]
        
        # Divide Fingers Bending costs and Rotation costs
        divide_fingers_data = np.array(raw_data_np[:, 0:5], np.float)
        divide_rotations_data = np.array(raw_data_np[:, 5:-1], np.float)
        divide_label_data = np.array(raw_data_np[:, -1]).astype(int) 
        

        # Normalize(-1 ~ 1)
        divide_fingers_data = (divide_fingers_data - self.bending_min) / (self.bending_max - self.bending_min)
        divide_fingers_data = divide_fingers_data * 2 - 1
        
        # Divide Rotations
        divide_rotations_data_yaw = np.array(divide_rotations_data[:, 0:1], np.float)
        divide_rotations_data_pitch = np.array(divide_rotations_data[:, 1:2], np.float)
        divide_rotations_data_roll = np.array(divide_rotations_data[:, 2:], np.float)
        
        # Normalize(-1 ~ 1)
        divide_rotations_data_yaw = (divide_rotations_data_yaw - self.yaw_min) / (self.yaw_max - self.yaw_min)
        divide_rotations_data_pitch = (divide_rotations_data_pitch - self.pitch_min) / (self.pitch_max - self.pitch_min)
        divide_rotations_data_roll = (divide_rotations_data_roll - self.roll_min) / (self.roll_max - self.roll_min)
        
        divide_rotations = np.concatenate([divide_rotations_data_yaw, 
                                            divide_rotations_data_pitch, 
                                            divide_rotations_data_roll],
                                            axis = 1)
        
        variable_input_datas = np.concatenate([divide_fingers_data, divide_rotations],
                                            axis = 1)
        
        # print(variable_input_datas.shape)
        # Divide Preprocessed ndarray with 5 rows
        quota, rest = divmod(len(variable_input_datas), 5)
        
        if rest > 0:
            quota += 1
            padding_rows = 5 - rest
            variable_input_datas = np.concatenate([variable_input_datas, np.zeros((padding_rows, 8))])
            
        # print('After Padding, ', variable_input_datas.shape)
        fixed_input_datas = variable_input_datas.reshape((quota, 5, 8)).astype(np.float)
        class_label_np = np.array(int(class_label), int)

        return {'gesture_data' : fixed_input_datas, 'class_label' : class_label_np}
            
    def __len__(self):
        return len(self.data_path)
    
if __name__ == '__main__':
    gesture_train_dataset = GestureDataset(data_path = total_data_path, train = True)
    print(gesture_train_dataset.__getitem__(0))
    gesture_test_dataset = GestureDataset(data_path = total_data_path, train = False)
    print(gesture_test_dataset.__getitem__(0))
    
    train_loader = DataLoader(gesture_train_dataset, batch_size = 8, shuffle = True)
    test_loader = DataLoader(gesture_test_dataset, batch_size = 1, shuffle = False)

# # TODO
