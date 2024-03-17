import os
import glob
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

man_data_paths = '/home/jabblee/Desktop/CRC_collections/CRC_update/2023_Gatherings/2023_Gathering_m/Labeling/*.xlsx'
woman_data_paths = '/home/jabblee/Desktop/CRC_collections/CRC_update/2023_Gatherings/2023_Gathering_w/Labeling/*.xlsx'
total_data_path = '/home/jabblee/Desktop/CRC_collections/CRC_update/2023_Gatherings/'
save_train_data_path = total_data_path + 'TRAIN/'
save_test_data_path = total_data_path + 'TEST/'

os.makedirs(save_train_data_path, exist_ok = True)
os.makedirs(save_test_data_path, exist_ok = True)

man_xlsx_lst = sorted(glob.glob(man_data_paths))
woman_xlsx_lst = sorted(glob.glob(woman_data_paths))
total_xlsx_lst = man_xlsx_lst + woman_xlsx_lst
print('Total xlsx files :', len(total_xlsx_lst))

# 총 자음 모음 수만큼의 sheet
sheet_name_lst = [('ㄱ', 1), ('ㄴ', 2), ('ㄷ', 3), ('ㄹ', 4), ('ㅁ', 5), ('ㅂ', 6), 
                  ('ㅅ', 7), ('ㅇ', 8), ('ㅈ', 9), ('ㅊ', 10), ('ㅋ', 11), ('ㅍ', 12), 
                  ('ㅎ', 13), ('ㅏ', 14), ('ㅑ', 15), ('ㅓ', 16), ('ㅕ', 17), 
                  ('ㅗ', 18), ('ㅛ', 19), ('ㅜ', 20), ('ㅡ', 21), ('ㅣ', 22)]

for person_idx, person_data_path in enumerate(total_xlsx_lst):
    person_name = person_data_path.split('/')[-1].replace('.xlsx', '')
    # print('person_name :', person_name)

    person_train_fingers = []
    person_train_accelerates = []
    for sheet_num, (hangul_character, label_cost) in tqdm(enumerate(sheet_name_lst)):
        person_sheet_total = pd.read_excel(person_data_path, sheet_name = sheet_num,
                                           header = 2)
        person_sheet_total = person_sheet_total.iloc[:, 2:]
        
        person_sheet_label_cost = person_sheet_total.columns[-1]
        if person_sheet_label_cost == hangul_character:
            person_sheet_total.rename(columns = {hangul_character:'Label'}, inplace = True)
            
        person_sheet_valid = person_sheet_total.loc[(person_sheet_total['Label'] == label_cost) |
                                                    (person_sheet_total['Label'] == 0)]
        person_sheet_valid = person_sheet_valid.loc[:, ['Thumb', 'Index', 'Middle', 'Ring', 'Little', 'Yaw', 'Pitch', 'Roll', 'Label']]
        
        # Erase NaN values
        person_sheet_valid = person_sheet_valid.dropna(axis = 0)
        
        # Divide 0 and label_cost in valid DataFrame
        person_sheet_valid_zeros = person_sheet_valid.loc[person_sheet_valid['Label'] == 0]
        person_sheet_valid_label_costs = person_sheet_valid.loc[person_sheet_valid['Label'] != 0]
        
        person_sheet_valid_label_costs_train = person_sheet_valid_label_costs.sample(frac = 0.8, random_state = 7)
        person_sheet_valid_label_costs_test = pd.merge(person_sheet_valid_label_costs, person_sheet_valid_label_costs_train,
                                                       how = 'outer',
                                                       indicator = True).query(
                                                           '_merge == "left_only"'
                                                       ).drop(columns = ['_merge'])
        
        person_sheet_train = person_sheet_valid_label_costs_train
        person_sheet_test = person_sheet_valid_label_costs_test
        
        person_sheet_train_np = person_sheet_train.to_numpy()
        person_sheet_test_np = person_sheet_test.to_numpy()
        
        # Divide ndarray with 5 rows
        train_total_rows, train_total_columns = person_sheet_train_np.shape
        test_total_rows, test_total_columns = person_sheet_test_np.shape
        
        train_quota, train_rest = divmod(train_total_rows, 5)
        test_quota, test_rest = divmod(test_total_rows, 5)
        
        if train_rest > 0:
            for train_cnt in range(train_quota + 1):
                chunk_person_sheet_train_np = person_sheet_train_np[train_cnt * 5:train_cnt * 5 + 5, :]
                np.save(save_train_data_path + person_name + '_' + str(label_cost) + '_' + str(train_cnt),
                        chunk_person_sheet_train_np)
        else:
            for train_cnt in range(train_quota):
                chunk_person_sheet_train_np = person_sheet_train_np[train_cnt * 5:train_cnt * 5 + 5, :]
                np.save(save_train_data_path + person_name + '_' + str(label_cost) + '_' + str(train_cnt),
                        chunk_person_sheet_train_np)
                
        if test_rest > 0:
            for test_cnt in range(test_quota + 1):
                chunk_person_sheet_test_np = person_sheet_test_np[test_cnt * 5: test_cnt * 5 + 5, :]
                np.save(save_test_data_path + person_name + '_' + str(label_cost) + '_' + str(test_cnt),
                        chunk_person_sheet_test_np)
        else:
            for test_cnt in range(test_quota):
                chunk_person_sheet_test_np = person_sheet_test_np[test_cnt * 5: test_cnt * 5 + 5, :]
                np.save(save_test_data_path + person_name + '_' + str(label_cost) + '_' + str(test_cnt),
                        chunk_person_sheet_test_np)

print('Done for making Pandas DataFrame to Numpy npy file.')