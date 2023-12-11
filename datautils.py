import os
import numpy as np
import pandas as pd
import math
import random
from datetime import datetime
import pickle
from utils import pkl_load, pad_nan_to_target
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils import take_per_row, split_with_nan, centerize_vary_length_series, torch_pad_nan
from torch.utils.data import TensorDataset, DataLoader,ConcatDataset
import torch

def load_UCR(dataset):
    train_file = os.path.join('datasets/UCR', dataset, dataset + "_TRAIN.tsv")
    test_file = os.path.join('datasets/UCR', dataset, dataset + "_TEST.tsv")
    train_df = pd.read_csv(train_file, sep='\t', header=None)
    test_df = pd.read_csv(test_file, sep='\t', header=None)
    train_array = np.array(train_df)
    test_array = np.array(test_df)

    # Move the labels to {0, ..., L-1}
    labels = np.unique(train_array[:, 0])
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    train = train_array[:, 1:].astype(np.float64)
    train_labels = np.vectorize(transform.get)(train_array[:, 0])
    test = test_array[:, 1:].astype(np.float64)
    test_labels = np.vectorize(transform.get)(test_array[:, 0])

    # Normalization for non-normalized datasets
    # To keep the amplitude information, we do not normalize values over
    # individual time series, but on the whole dataset
    if dataset not in [
        'AllGestureWiimoteX',
        'AllGestureWiimoteY',
        'AllGestureWiimoteZ',
        'BME',
        'Chinatown',
        'Crop',
        'EOGHorizontalSignal',
        'EOGVerticalSignal',
        'Fungi',
        'GestureMidAirD1',
        'GestureMidAirD2',
        'GestureMidAirD3',
        'GesturePebbleZ1',
        'GesturePebbleZ2',
        'GunPointAgeSpan',
        'GunPointMaleVersusFemale',
        'GunPointOldVersusYoung',
        'HouseTwenty',
        'InsectEPGRegularTrain',
        'InsectEPGSmallTrain',
        'MelbournePedestrian',
        'PickupGestureWiimoteZ',
        'PigAirwayPressure',
        'PigArtPressure',
        'PigCVP',
        'PLAID',
        'PowerCons',
        'Rock',
        'SemgHandGenderCh2',
        'SemgHandMovementCh2',
        'SemgHandSubjectCh2',
        'ShakeGestureWiimoteZ',
        'SmoothSubspace',
        'UMD'
    ]:
        return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels
    
    mean = np.nanmean(train)
    std = np.nanstd(train)
    train = (train - mean) / std
    test = (test - mean) / std
    return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels


def load_UEA(dataset):
    train_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TRAIN.arff')[0]
    test_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TEST.arff')[0]
    
    def extract_data(data):
        res_data = []
        res_labels = []
        for t_data, t_label in data:
            t_data = np.array([ d.tolist() for d in t_data ])
            t_label = t_label.decode("utf-8")
            res_data.append(t_data)
            res_labels.append(t_label)
        return np.array(res_data).swapaxes(1, 2), np.array(res_labels)
    
    train_X, train_y = extract_data(train_data)
    test_X, test_y = extract_data(test_data)
    
    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)
    
    labels = np.unique(train_y)
    transform = { k : i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    return train_X, train_y, test_X, test_y
    
    
def load_forecast_npy(name, univar=False):
    data = np.load(f'datasets/{name}.npy')    
    if univar:
        data = data[: -1:]
        
    train_slice = slice(None, int(0.6 * len(data)))
    valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
    test_slice = slice(int(0.8 * len(data)), None)
    
    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    data = np.expand_dims(data, 0)

    pred_lens = [24, 48, 96, 288, 672]
    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, 0


def _get_time_features(dt):
    return np.stack([
        dt.minute.to_numpy(),
        dt.hour.to_numpy(),
        dt.dayofweek.to_numpy(),
        dt.day.to_numpy(),
        dt.dayofyear.to_numpy(),
        dt.month.to_numpy(),
        dt.weekofyear.to_numpy(),
    ], axis=1).astype(np.float)


def load_forecast_csv(name, univar=True):
    data = pd.read_csv(f'datasets/{name}.csv', index_col='date', parse_dates=True)
    dt_embed = _get_time_features(data.index)
    n_covariate_cols = dt_embed.shape[-1]
    
    # if univar:
    #     data = data[['OT']]
        
        
    data = data.to_numpy()
    # if name == 'ETTh1' or name == 'ETTh2':
    #     train_slice = slice(None, 12*30*24)
    #     valid_slice = slice(12*30*24, 16*30*24)
    #     test_slice = slice(16*30*24, 20*30*24)
    # elif name == 'ETTm1' or name == 'ETTm2':
    #     train_slice = slice(None, 12*30*24*4)
    #     valid_slice = slice(12*30*24*4, 16*30*24*4)
    #     test_slice = slice(16*30*24*4, 20*30*24*4)
    # else:
    train_slice = slice(None, int(0.6 * len(data)))
    valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
    test_slice = slice(int(0.8 * len(data)), None)
    
    scaler = StandardScaler().fit(data)
    data = scaler.transform(data)
    # if name in ('electricity'):
    #     data = np.expand_dims(data.T, -1)  # Each variable is an instance rather than a feature
    # else:
    data = np.expand_dims(data, 0)
    
    if n_covariate_cols > 0:
        dt_scaler = StandardScaler().fit(dt_embed[train_slice])
        dt_embed = np.expand_dims(dt_scaler.transform(dt_embed), 0)
        data = np.concatenate([np.repeat(dt_embed, data.shape[0], axis=0), data], axis=-1)

    datatemp=data[:,:,dt_embed.shape[2]:dt_embed.shape[2]+1]
    for i in range(dt_embed.shape[2]+1,data.shape[2]):
        temp_channel= data[:,:,i:i+1]
        datatemp=np.concatenate([datatemp,temp_channel],axis=0)
    
   
    pred_lens = [7]
 
        
    return datatemp, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols

def load_muti_forecast_csv(args,name,univar=True):
    data = pd.read_csv(f'datasets/{name}.csv', index_col='date', parse_dates=True)
    dt_embed = _get_time_features(data.index)
    n_covariate_cols = dt_embed.shape[-1]
   
    train_slice = slice(None, int(0.7 * len(data)))
    valid_slice = slice(int(0.7 * len(data)), int(0.8 * len(data)))
    test_slice = slice(int(0.8 * len(data)), None)
    scaler = StandardScaler().fit(data)
    data = scaler.transform(data)
    data = np.expand_dims(data, 0)
    ##
   
    dt_scaler = StandardScaler().fit(dt_embed)
    dt_embed = np.expand_dims(dt_scaler.transform(dt_embed), 0)
    data = np.concatenate([np.repeat(dt_embed, data.shape[0], axis=0), data], axis=-1)
    ##
   
    ##可以更改 改prelen
    pred_lens = [7]

    train_data=data[:,train_slice]
    sections = train_data.shape[1] // args.max_train_length
    if sections >= 2:
        train_data = np.concatenate(split_with_nan(train_data, sections, axis=1), axis=0)
    temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
    if temporal_missing[0] or temporal_missing[-1]:
        train_data = centerize_vary_length_series(train_data)
    train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]
    single_dset=[]
    for i in range(dt_embed.shape[2],train_data.shape[2]):
        temp=train_data[:,:,i:i+1]
        #tempdata=np.concatenate([train_data[:,:,:dt_embed.shape[2]], train_data[:,:,i:i+1]], axis=-1)
        train_dataset=TensorDataset(torch.from_numpy(temp).to(torch.float))
        single_dset.append(train_dataset)
    Concat_single_dataset=ConcatDataset(single_dset)
    train_loader = DataLoader(Concat_single_dataset, batch_size=min(args.batch_size, len(train_dataset)), shuffle=True, drop_last=True)
    return train_loader, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols



def load_anomaly(name):
    res = pkl_load(f'datasets/{name}.pkl')
    return res['all_train_data'], res['all_train_labels'], res['all_train_timestamps'], \
           res['all_test_data'],  res['all_test_labels'],  res['all_test_timestamps'], \
           res['delay']


def gen_ano_train_data(all_train_data):
    maxl = np.max([ len(all_train_data[k]) for k in all_train_data ])
    pretrain_data = []
    for k in all_train_data:
        train_data = pad_nan_to_target(all_train_data[k], maxl, axis=0)
        pretrain_data.append(train_data)
    pretrain_data = np.expand_dims(np.stack(pretrain_data), 2)
    return pretrain_data



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True

def set_device(usage=5, device=None):
    "set the device that has usage < default usage  "
    if device:
        torch.cuda.set_device(device)
        return
    device_ids = get_available_cuda(usage=usage)
    torch.cuda.set_device(device_ids[0])   # get the first available device
    
def get_available_cuda(usage=10):
    if not torch.cuda.is_available(): return
    # collect available cuda devices, only collect devices that has less that 'usage' percent 
    device_ids = []
    for device in range(torch.cuda.device_count()):
        if torch.cuda.utilization(device) < usage: device_ids.append(device)
    return device_ids        