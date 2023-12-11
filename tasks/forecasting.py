import numpy as np
import time
from . import _eval_protocols as eval_protocols

def generate_pred_samples(features, data, pred_len, drop=0):
    n = data.shape[1]
    features = features[:, :-pred_len]
    labels = np.stack([ data[:, i:1+n+i-pred_len] for i in range(pred_len)], axis=2)[:, 1:]
    features = features[:, drop:]
    labels = labels[:, drop:]
    return features.reshape(-1, features.shape[-1]), \
            labels.reshape(-1, labels.shape[2]*labels.shape[3])

def cal_metrics(pred, target):
    mask=target!=0
    
    return {
        'MSE': ((pred - target) ** 2).mean(),
        'MAE': np.abs(pred - target).mean(),
        'MAPE': np.abs((target[mask]-pred[mask])/target[mask]).mean()
    }
    
def eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols):
    padding = 200
    
    t = time.time()
    
  
    all_repr = model.encode(
        data,
        causal=True,
        sliding_length=1,
        sliding_padding=padding,
        batch_size=1024
    )
    ts2vec_infer_time = time.time() - t
    
    train_repr = all_repr[:, train_slice]
    valid_repr = all_repr[:, valid_slice]
    test_repr = all_repr[:, test_slice]
    
    # train_data = data[:, train_slice, n_covariate_cols:]
    # valid_data = data[:, valid_slice, n_covariate_cols:]
    # test_data = data[:, test_slice, n_covariate_cols:]
    train_data = data[:, train_slice, :]
    valid_data = data[:, valid_slice, :]
    test_data = data[:, test_slice, :]
    
    
    ours_result = {}
    lr_train_time = {}
    lr_infer_time = {}
    out_log = {}
    for pred_len in pred_lens:
        train_features, train_labels = generate_pred_samples(train_repr, train_data, pred_len, drop=padding)
        valid_features, valid_labels = generate_pred_samples(valid_repr, valid_data, pred_len)
        test_features, test_labels = generate_pred_samples(test_repr, test_data, pred_len)
        
        t = time.time()
        lr = eval_protocols.fit_ridge(train_features, train_labels, valid_features, valid_labels)
        lr_train_time[pred_len] = time.time() - t
        
        t = time.time()
        test_pred = lr.predict(test_features)
        lr_infer_time[pred_len] = time.time() - t

        ori_shape = test_data.shape[0], -1, pred_len, test_data.shape[2]
        test_pred = test_pred.reshape(ori_shape)
        test_labels = test_labels.reshape(ori_shape)
        ###inverse有bug 因为数据集维度变了，不执行invers，观察norm后的mse和mae即可
        # if test_data.shape[0] > 1:
        #     test_pred_inv = scaler.inverse_transform(test_pred.swapaxes(0, 3)).swapaxes(0, 3)
        #     test_labels_inv = scaler.inverse_transform(test_labels.swapaxes(0, 3)).swapaxes(0, 3)
        # else:
        #     test_pred_inv = scaler.inverse_transform(test_pred)
        #     test_labels_inv = scaler.inverse_transform(test_labels)
            
        out_log[pred_len] = {
            'norm': test_pred,
            # 'raw': test_pred_inv,
            'norm_gt': test_labels
            # 'raw_gt': test_labels_inv
        }
        # ours_result[pred_len] = {
        #     'norm': cal_metrics(test_pred, test_labels)
        #     'raw': cal_metrics(test_pred_inv, test_labels_inv)
        # }
        ours_result= cal_metrics(test_pred, test_labels)
        #     'raw': cal_metrics(test_pred_inv, test_labels_inv)
        # pro = cal_metrics(test_pred[-1,:,:],test_labels[-1,:,:])
        # city = cal_metrics(test_pred[-2,:,:],test_labels[-2,:,:])
        # area = cal_metrics(test_pred[-6:-2,:,:],test_labels[-6:-2,:,:])
        # gb = cal_metrics(test_pred[4093:-6,:,:],test_labels[4093:-6,:,:])
        # zb = cal_metrics(test_pred[:4093,:,:],test_labels[:4093,:,:])
    eval_res = {
        'all': ours_result,
        # 'province':pro,
        # 'city':city,
        # 'area':area,
        # 'gb':gb,
        # 'zb':zb,
        'ts2vec_infer_time': ts2vec_infer_time,
        'lr_train_time': lr_train_time,
        'lr_infer_time': lr_infer_time
    }
    # return out_log, eval_res
    print(eval_res)
    return ours_result
