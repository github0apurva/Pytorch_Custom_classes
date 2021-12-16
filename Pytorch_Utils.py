import dataiku
from dataiku import pandasutils as pdu
import pandas as pd
import numpy as np
import time as tp
import os
import DF_ChangeFormat as dfchange
import ex_data_prep as data_prep 
from datetime import datetime
import gc
from timeit import default_timer as timer


import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import *


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel.data_parallel import data_parallel
from torch.nn.utils.rnn import *
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score,recall_score, precision_score

from collections import defaultdict
import copy
import itertools
import pickle
import sys
import glob
import time




def UnderSampler ( df , sel_idx, pct ):
    n = len (sel_idx)
    bio_idxs  = sel_idx[ np.where (df.iloc[sel_idx].target)[0] ]
    nbio_idxs = sel_idx[ np.where ( df.iloc[sel_idx].target == 0)[0] ]
    bio = bio_idxs
    nbio_num = int(len(bio_idxs) * (1-pct) / pct )
    rng = np.random.default_rng(27)
    #nbio = np.random.choice(nbio_idxs, nbio_num, replace = False)
    nbio = rng.choice(nbio_idxs, nbio_num, replace = False)
    if pct == 1 :
        nbio = nbio_idxs
    idxs = np.hstack([bio,nbio])
    np.random.shuffle(idxs)
    idxs = idxs[:n]
    return idxs


def data_treatment (labels, features):
    labels.reset_index(inplace=True,drop=False)
    labels.columns  = ['PATIENT_ID','target']
    labels['target'] = labels['target'].astype(int)
    labels['PATIENT_ID'] = labels['PATIENT_ID'].astype(int)
    featuresDF = pd.DataFrame(features.reshape(features.shape[0],-1))
    consolDF = pd.concat([featuresDF,labels],1)
    print ( consolDF.info() )
    return consolDF


def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)

    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)

    else:
        raise NotImplementedError

def make_fold(df, mode='train-1',splits=5 ):
    if 'train' in mode:
        fold = int(mode[-1])
        kf = StratifiedKFold(n_splits=splits, random_state=123, shuffle=True)
        train_idx, valid_idx = [],[]
        for t, v in kf.split(df['PATIENT_ID'],df['target']):
            train_idx.append(t)
            valid_idx.append(v)
        return train_idx[fold], valid_idx[fold]

    if 'test' in mode:
        valid_idx = np.arange(len(df))
        return valid_idx
  

    
def make_split(df, mode, valid_ratio ):
    if 'train' in mode:
        req_df = df[['PATIENT_ID','target']]
        train,test = train_test_split(req_df['PATIENT_ID'], test_size = valid_ratio ,
                                      random_state= 27, stratify= req_df['target'] )
        return np.array(train.index) , np.array(test.index)

    if 'test' in mode:
        valid_idx = np.arange(len(df))
        return valid_idx    
  


   
    
    
class ExperimentsDataset(Dataset ) :
    def __init__(self, consolDF, idx_old, pct):
        super().__init__()
        idx = UnderSampler (consolDF,idx_old , pct )
        self.length = len(idx)
        self.idx = idx
        self.consolDF = consolDF
        self.features = self.consolDF.drop(['PATIENT_ID','target'],1)
        
        self.target  = self.consolDF['target'].values.reshape(-1,1)
        self.features = self.features.values.astype(np.float32).reshape(-1, 12,  62)    ### make dynamic


    def __str__(self):
        string  = ''
        string += '\tlen = %d\n'%len(self)
        return string

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        i = self.idx[index]
        r = {
            'index'     : index,
            'patient_id' : self.consolDF.PATIENT_ID[i],
            'target'    :self.target[i],
            'feature'   : self.features[i]

        }
        return r

    
    
    
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  #stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode ='w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1 ):
        if '\r' in message: is_file=0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            #time.sleep(1)

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass
    
    
    
def save_s3 ( folder , time_stamp = False, project_name = 'DS_AI_INNOV_CODES', folder_name = 'y38zDK7E' ) : 
    # example 'AC_Test__Nov25_13h_16m'
    if time_stamp == True:
        ts = datetime.now().strftime("__%b%d_%Hh_%Mm")
    else:
        ts = ''
    s3_path='s3://novartisrsagbusnvglassboxprodselfserve001/datalab/dataiku/managed_folders/PRJ_' + project_name + '/' + project_name + '/' + folder_name + '/'
    local_path = os.getcwd()
    pick_folder = local_path   +'/' + folder + '/'
    park_folder = s3_path + folder + ts + '/' # folder + '/'
    load_folder = "aws s3 cp " + pick_folder +" " + park_folder+" --recursive --sse aws:kms"
    print (load_folder)
    print ( os.system(load_folder) , ":  0 signifies succesful execution of code ") 
    print ( '\n  ***  exit  ***')
    
    
    
class UnderSampler_notusing1 (Sampler):
    def __init__ (self, dataset,sel_idx, bio_pct = 0.1  ):
        self.dataset = dataset
        self.n = len(dataset)
        self.sel_idx = sel_idx
        self.pct = bio_pct
    def __iter__(self):
     
        bio_idxs = np.where (self.dataset[self.sel_idx].target)[0] 
        nbio_idxs = np.where ( self.dataset[self.sel_idx].target == 0)[0] 
        bio = np.random.choice(bio_idxs, len(bio_idxs), replace = False)
        nbio_num = int(len(bio_idxs) * (1-self.pct) / self.pct ) 
        nbio = np.random.choice(nbio_idxs, nbio_num, replace = False)
        idxs = np.hstack([bio,nbio])
        np.random.shuffle(idxs)
        idxs = idxs[:self.n]
        return iter(idxs)  
    def __len__(self):
        return self.n
    

class UnderSampler_notusing2 (Sampler):
    def __init__ (self, dataset,sel_idx, bio_pct = 0.1  ):
        self.dataset = dataset
        self.n = len(dataset)
        self.sel_idx = sel_idx
        self.pct = bio_pct
    def __iter__(self):
        bio_idxs = self.sel_idx [ np.where (self.dataset.iloc[self.sel_idx].target)[0] ]
        nbio_idxs = self.sel_idx [ np.where ( self.dataset.iloc[self.sel_idx].target == 0)[0] ]
        bio = np.random.choice(bio_idxs, len(bio_idxs), replace = False)
        nbio_num = int(len(bio_idxs) * (1-self.pct) / self.pct ) + 1 
        nbio = np.random.choice(nbio_idxs, nbio_num, replace = False)
        idxs = np.hstack([bio,nbio])
        np.random.shuffle(idxs)
        idxs = idxs[:self.n]
        return iter(idxs)  
    def __len__(self):
        return self.n  

    
