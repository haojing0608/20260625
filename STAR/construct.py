# -*- coding: utf-8 -*-
import sys, time, os
import yaml
import numpy as np
import pandas as pd
from astropy.io import fits
from Utils import *

def cat_fits_filename(info, fits_path='./data/DR10/fits/'):
    filename = os.path.basename(info['combined_file'])
    subclass = info['combined_subclass'][0]  
    if subclass == 'A':
        subdir = 'A4522_'
    elif subclass == 'B':
        subdir = 'B4523_'
    elif subclass == 'F':
        subdir = 'F4530_'
    elif subclass == 'G':
        subdir = 'G4531_'
    elif subclass == 'K':
        subdir = 'K4533_'
    else:
        return None 
    
    full_path = os.path.join(fits_path, subdir, filename)
    
    if os.path.exists(full_path):
        return full_path
    else:
        return None

def parse_s(s, length):
    l = len(s)
    return '0'*(length-l) + s
def chose_snr(snr, info):
    if snr == '>30':
        if info['combined_snrr'] > 30 and info['combined_snri']> 30:
            return cat_fits_filename(info)
        else:
            return None
    elif snr == '10-30':
        if 10 < info['combined_snrr'] < 30 or 10 < info['combined_snri'] < 30:
            return cat_fits_filename(info)
        else:
            return None
    elif snr == '<10':
        if info['combined_snrr'] < 10 and info['combined_snri'] < 10:
            return cat_fits_filename(info)
        else:
            return None
    elif snr == '>10':
        if info['combined_snrr'] > 10 and info['combined_snri'] > 10:
            return cat_fits_filename(info)
        else:
            return None
    elif snr == 'all':
        return cat_fits_filename(info)
    else:
        print('snr input error\n')
        sys.exit()


def construct(config):

    classes = config['classes'].keys()
    classes_data = {} 
    classes_data_num = {}  
    classes_label = {}  
    for e, i in enumerate(classes):
        classes_data[i] = []
        classes_data_num[i] = 0
        classes_label[i] = e
    num_all = sum(config['classes'].values())
    for index, row in star_table.iterrows():
        if index%500==0:
            print(index)
            print(classes_data_num)
        snr_yn = chose_snr(config['snr'], row) 
        #filename_i =
        # print(snr_yn)
        if snr_yn != None:
            filename_i = snr_yn  
            #print(filename_i)
            class_i = row['combined_subclass'][0] 
            #print(class_i,classes)
            if row['combined_class']=='STAR' and class_i in classes: 
                if classes_data_num[class_i] < config['classes'][class_i]: 
                    if config['data_type'] == 'spectra':
                        sp_i = read_fits(filename_i)
                    elif config['data_type'] == 'line_index':
                        sp_i = read_line_index(filename_i)
                    sp_i = np.append(sp_i, classes_label[class_i]) 
                    # print(sp_i.shape)
                    classes_data[class_i].append(sp_i)
                    classes_data_num[class_i] += 1
        if sum(classes_data_num.values()) == num_all:
            f_save = open(config['save_filename'], 'w')
            for k, v in classes_data.items():
                np.savetxt(f_save, np.array(v), fmt='%.4f', delimiter=',')
            f_save.close()
            print('finish choose')
            break
        else:
            pass

    pass

def construct_sgq(config):

    classes = config['classes'].keys()
    classes_data = {}  
    classes_data_num = {}  
    classes_label = {}  
    for e, i in enumerate(classes):
        classes_data[i] = []
        classes_data_num[i] = 0
        classes_label[i] = e
    num_all = sum(config['classes'].values())
    # === ����������ʱ�ļ� ===
    save_file = config['save_filename']
    for c in classes:
        tmp_file = f"{save_file}.{c}.tmp"
        if os.path.exists(tmp_file):
            print(f"Loading {c} from tmp...")
            loaded = np.loadtxt(tmp_file, delimiter=',')
            classes_data[c] = loaded.tolist() if len(loaded.shape) > 1 else [loaded.tolist()]
            classes_data_num[c] = len(classes_data[c])
            print(f"{c}: {classes_data_num[c]}")
    # =========
    for index, row in star_table.iterrows():
        if index%500==0:
            print(index)
            print(classes_data_num)

        if row['class']=='STAR':
            snr_yn = chose_snr(config['snr'], row)  
        elif  row['class']=='QSO' or row['class']=='GALAXY':
            snr_yn = chose_snr('all', row)
        else:
            snr_yn = None
        #filename_i =
        # print(snr_yn)
        if snr_yn != None:
            filename_i = snr_yn
            #print(filename_i)
            class_i = row['combined_class']  
            #print(class_i,classes)
            if class_i in classes:  
                    # ======
                    try:
                        if config['data_type'] == 'spectra':
                            sp_i = read_fits(filename_i)
                        elif config['data_type'] == 'line_index':
                            sp_i = read_line_index(filename_i)
                        
                        if sp_i is None or len(sp_i) == 0:
                            continue
                            
                        sp_i = np.append(sp_i, classes_label[class_i]) 
                        classes_data[class_i].append(sp_i)
                        classes_data_num[class_i] += 1
                        
                        # ======
                        if classes_data_num[class_i] % 10000 == 0:
                            tmp_file = f"{save_file}.{class_i}.tmp"
                            np.savetxt(tmp_file, np.array(classes_data[class_i]), fmt='%.4f', delimiter=',')
                            print(f"Saved {class_i} tmp: {classes_data_num[class_i]}")
                        # ======
                        
                    except Exception as e:
                        print(f"Error at row {index}: {e}")
                        continue
                    # ======
        if sum(classes_data_num.values()) == num_all:
            f_save = open(config['save_filename'], 'w')
            for k, v in classes_data.items():
                np.savetxt(f_save, np.array(v), fmt='%.4f', delimiter=',')
            f_save.close()
            print('finish choose')
            break
        else:
            pass
    # === ѭ������Ҳ������ʱ�ļ� ===
    print(f"Final: {classes_data_num}")
    for c in classes:
        if classes_data_num[c] > 0:
            tmp_file = f"{save_file}.{c}.tmp"
            np.savetxt(tmp_file, np.array(classes_data[c]), fmt='%.4f', delimiter=',')
            print(f"Saved final tmp: {tmp_file}")
    # ======
    pass
def construct_sgq_remove_reshift(config):
    classes = config['classes'].keys()
    classes_data = {}  
    classes_data_num = {}  
    classes_label = {} 
    for e, i in enumerate(classes):
        classes_data[i] = []
        classes_data_num[i] = 0
        classes_label[i] = e
    num_all = sum(config['classes'].values())
    for index, row in star_table.iterrows():
        if index%500==0:
            print(index)
            print(classes_data_num)

        if row['class']=='STAR' and 0<row['z']<0.3 and row['z']!=-9999:
            snr_yn = chose_snr('>10', row) 
        elif row['class']=='GALAXY' and 0<row['z']<0.3 and row['z']!=-9999:
            snr_yn = chose_snr('all', row)
        elif row['class']=='QSO' and row['z']!=-9999 and 0<row['z']<0.3:
            snr_yn = chose_snr('all', row)
        else:
            snr_yn = None
        #filename_i =
        # print(snr_yn)
        if snr_yn != None:
            filename_i = snr_yn
            #print(filename_i)
            class_i = row['combined_class']  
            #print(class_i,classes)
            if class_i in classes:  
                if classes_data_num[class_i] < config['classes'][class_i]:  
                    if config['data_type'] == 'spectra':
                        sp_i = read_fits_remove_redshift(filename_i)

                    elif config['data_type'] == 'line_index':
                        sp_i = read_line_index(filename_i)
                    if sp_i is not None:
                        if len(sp_i) != 2580:
                            print(len(sp_i))
                            sys.exit()
                        sp_i = np.append(sp_i, classes_label[class_i])
                        classes_data[class_i].append(sp_i)
                        classes_data_num[class_i] += 1
        if sum(classes_data_num.values()) == num_all:
            f_save = open(config['save_filename'], 'w')
            for k, v in classes_data.items():
                np.savetxt(f_save, np.array(v), fmt='%.4f', delimiter=',')
            f_save.close()
            print('finish choose')
            break
        else:
            pass

    pass

if __name__ == '__main__':
    t1 = time.time()
    star_table = pd.read_csv('./data/spectra/merged_star_table.csv')
    t2 = time.time()
    print(t2 - t1)
    with open('config.yml', encoding='utf-8') as file_config:
        data_config = yaml.load(file_config, Loader=yaml.FullLoader) 
    construct(data_config['Star_1M_balanced'])
    # construct(data_config['Diff_Size_2'])
    # construct(data_config['Diff_Size_3'])
    # construct(data_config['Diff_SNR_h'])
    # construct(data_config['Diff_SNR_m'])
    # construct(data_config['Diff_SNR_l'])
    #construct_sgq_remove_reshift(data_config['SGQ_remove_shift'])
    # construct(data_config['Diff_Feature_LineIndex'])
    # construct(data_config['Diff_Feature_1Dspectra'])
    #construct(data_config['Diff_Size_4'])
    #construct(data_config['NormalSpectraStar'])
    #construct_sgq(data_config['NormalSpectraGQ'])
     # construct_sgq(data_config['SGQ_10000'])