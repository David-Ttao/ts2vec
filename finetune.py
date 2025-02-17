import torch
import numpy as np
import argparse
import os
import sys
import time
import datetime
from ts2vec import TS2Vec
import tasks
import datautils
from utils import init_dl_program, name_with_datetime, pkl_save, data_dropout
from torch.utils.data import ConcatDataset
from utils import save_one_experiment_result
from datautils import setup_seed,set_device
def save_checkpoint_callback(
    save_every=1,
    unit='epoch'
):
    assert unit in ('epoch', 'iter')
    def callback(model, loss):
        n = model.n_epochs if unit == 'epoch' else model.n_iters
        if n % save_every == 0:
            model.save(f'{run_dir}/model_{n}.pkl')
    return callback

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default= 'ETTh1',help='The dataset name')
    parser.add_argument('--run_name',default='forecast_multivar' ,help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('--loader', type=str, default='forecast_csv', help='The data loader used to load the experimental data. This can be set to UCR, UEA, forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch-size', type=int, default=8, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')
    parser.add_argument('--max-train-length', type=int, default=3000, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=None, help='The number of epochs')
    parser.add_argument('--save-every', type=int, default=None, help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--max-threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
    parser.add_argument('--eval', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--irregular', type=float, default=0, help='The ratio of missing observations (defaults to 0)')
    ###new
    parser.add_argument('--muti_dataset',type=str,default=None)
    parser.add_argument('--target',type=str,default='ETTh1')
    parser.add_argument('--model_path',type=str,default='/home/yupengz/ts2vec/training/ETTh1_epochs_250/model.pkl')
    parser.add_argument('--record',type=int,default=1)
    parser.add_argument('--random_seed',type=int,default=None)
    parser.add_argument('--model_name',type=str,default=None)
    args = parser.parse_args()
    
    print("Dataset:", args.dataset)
    print("Arguments:", str(args))
    
    #device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)
    setup_seed(args.random_seed)
    set_device(device=args.gpu)
    device=torch.device(args.gpu)
    #切割数据集字符
    if args.muti_dataset:
        args.muti_dataset=args.muti_dataset.split("_")

    print('Loading data... ', end='')
    #read data
    data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.target)
    train_data = data[:, train_slice]      
        

    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        output_dims=args.repr_dims,
        max_train_length=args.max_train_length
    )
    
    if args.save_every is not None:
        unit = 'epoch' if args.epochs is not None else 'iter'
        config[f'after_{unit}_callback'] = save_checkpoint_callback(args.save_every, unit)

    
    model = TS2Vec(
        input_dims=1,
        device=device,
        **config
    )

    model.load(args.model_path)

    # loss_log = model.fit(
    #     train_data,
    #     n_epochs=100,
    #     n_iters=args.iters,
    #     verbose=True
    # )
    eval_res = tasks.eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols)
    
    print('Evaluation result:', eval_res)
    print("Finished.")
    if args.record:
        save_one_experiment_result(args, eval_res)