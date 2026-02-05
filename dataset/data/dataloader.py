import numpy as np
import torch


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = torch.mean(data)
        self.std = torch.std(data)

    def transform(self, data):
        return (data - self.mean) / self.std

    def fit_transform(self, data):

        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):

        return data * self.std + self.mean
    


def load_dataset(cfg):

    folder_path = cfg.data.path
    data = np.load(folder_path)
    B, H, W = data.shape
    data = data.reshape(B, -1, 1)
    data_ts = np.array([(i // int(24/cfg.data.points_per_hour) %7, i % int(24/cfg.data.points_per_hour)) for i in range(data.shape[0])])



    data_value = torch.tensor(data).float()
    data_ts = torch.tensor(data_ts).float()



    print('shape:',data_value.shape)

    L, K,C = data_value.shape

    num_samples = L - (cfg.history_len + cfg.predict_len) + 1
    train_num = round(num_samples * 0.6)
    valid_num = round(num_samples * 0.2)
    test_num = num_samples - train_num - valid_num
    print("number of training samples:{0}".format(train_num))
    print("number of validation samples:{0}".format(valid_num))
    print("number of test samples:{0}".format(test_num))

    index_list = []
    for t in range(cfg.history_len, num_samples + cfg.history_len):
        index = (t-cfg.history_len, t, t+cfg.predict_len)
        index_list.append(index)

    train_index = index_list[:train_num]
    valid_index = index_list[train_num: train_num + valid_num]
    test_index = index_list[train_num +
                            valid_num: train_num + valid_num + test_num]    

    data_train=data_value[train_index[0][0]:train_index[-1][1],:,:]

    scaler = StandardScaler()
    scaler.fit(data_train)
    data_value = scaler.transform(data_value)
    data_train = scaler.transform(data_train)

    data_train=data_train.cpu().numpy()


    mask=torch.ones(cfg.predict_len + cfg.history_len,K,C)
    mask[cfg.history_len:]=0

    train_data = [
    {
        'observed_data': data_value[i[0]:i[0] + cfg.predict_len + cfg.history_len], 
        'gt_mask': mask,
        'timepoints': data_ts[i[0]:i[0] + cfg.predict_len + cfg.history_len],
    } 
    for i in train_index
                ]
    test_data = [
    {
        'observed_data': data_value[i[0]:i[0] + cfg.predict_len + cfg.history_len], 
        'gt_mask': mask,
        'timepoints': data_ts[i[0]:i[0] + cfg.predict_len + cfg.history_len],
    }
    for i in test_index
    ]
    val_data = [
    {
        'observed_data': data_value[i[0]:i[0] + cfg.predict_len + cfg.history_len], 
        'gt_mask': mask,
        'timepoints': data_ts[i[0]:i[0] + cfg.predict_len + cfg.history_len],
    }
    for i in valid_index
    ]

    train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=cfg.batch_size, shuffle=True,drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size = 2 * cfg.batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_data, num_workers=4, batch_size = 2 * cfg.batch_size, shuffle=False)


    all_targets=[]
    for x in test_loader:
        all_targets.append(x['observed_data'][:,cfg.history_len:,:,:])
    test_target_tensor = torch.cat(all_targets) #N,T,K,1
    test_target_tensor=scaler.inverse_transform(test_target_tensor)
    print('test_target_tensor size:',test_target_tensor.size())
    all_targets=[]
    for x in val_loader:
        all_targets.append(x['observed_data'][:,cfg.history_len:,:,:])
    val_target_tensor = torch.cat(all_targets)
    val_target_tensor=scaler.inverse_transform(val_target_tensor)
    print('val_target_tensor size:',val_target_tensor.size())    



    return train_loader,val_loader,test_loader,val_target_tensor,test_target_tensor,scaler

def fft_decomposition(data, threshold=0.1):
    T, K, _ = data.shape
    trend = np.zeros_like(data)
    residual = np.zeros_like(data)
    
    for k in range(K):
        series = data[:, k, 0]
        fft_coeffs = np.fft.fft(series)
        magnitudes = np.abs(fft_coeffs)
        max_magnitude = np.max(magnitudes)

        low_freq_coeffs = np.where(magnitudes > threshold * max_magnitude, fft_coeffs, 0)

        high_freq_coeffs = fft_coeffs - low_freq_coeffs
        residual[:, k, 0] = np.real(np.fft.ifft(high_freq_coeffs))
    
    return residual



def load_dataset_FFT(cfg):

    folder_path = cfg.data.path
    data = np.load(folder_path)
    data = np.load(folder_path)
    B, H, W = data.shape
    data = data.reshape(B, -1, 1)
    data_ts = np.array([(i // int(24/cfg.data.points_per_hour) %7, i % int(24/cfg.data.points_per_hour)) for i in range(data.shape[0])])



    data_value = torch.tensor(data).float()#.to(cfg.data.device)
    data_ts = torch.tensor(data_ts).float()#.to(cfg.data.device)


    print('shape:',data_value.shape)



    L, K,C = data_value.shape

    num_samples = L - (cfg.history_len + cfg.predict_len) + 1
    train_num = round(num_samples * 0.6)
    valid_num = round(num_samples * 0.2)
    test_num = num_samples - train_num - valid_num
    print("number of training samples:{0}".format(train_num))
    print("number of validation samples:{0}".format(valid_num))
    print("number of test samples:{0}".format(test_num))

    index_list = []
    for t in range(cfg.history_len, num_samples + cfg.history_len):
        index = (t-cfg.history_len, t, t+cfg.predict_len)
        index_list.append(index)

    train_index = index_list[:train_num]
    valid_index = index_list[train_num: train_num + valid_num]
    test_index = index_list[train_num +
                            valid_num: train_num + valid_num + test_num]


    data_train=data_value[train_index[0][0]:train_index[-1][1],:,:]

    scaler = StandardScaler()
    scaler.fit(data_train)
    data_value = scaler.transform(data_value)
    data_train = scaler.transform(data_train)
        
    data_train=data_train.cpu().numpy()



    residual = fft_decomposition(data_train, threshold=0.1)
    residual=torch.from_numpy(np.std(residual,axis=(0,2)))




    mask=torch.ones(cfg.predict_len + cfg.history_len,K,C)
    mask[cfg.history_len:]=0

    train_data = [
    {
        'observed_data': data_value[i[0]:i[0] + cfg.predict_len + cfg.history_len], 
        'gt_mask': mask,
        'timepoints': data_ts[i[0]:i[0] + cfg.predict_len + cfg.history_len],
        'scale_residual':residual
                } 
    for i in train_index
                ]
    test_data = [
    {
        'observed_data': data_value[i[0]:i[0] + cfg.predict_len + cfg.history_len], 
        'gt_mask': mask,
        'timepoints': data_ts[i[0]:i[0] + cfg.predict_len + cfg.history_len],
        'scale_residual': residual
                }
    for i in test_index
    ]
    val_data = [
    {
        'observed_data': data_value[i[0]:i[0] + cfg.predict_len + cfg.history_len], 
        'gt_mask': mask,
        'timepoints': data_ts[i[0]:i[0] + cfg.predict_len + cfg.history_len],
        'scale_residual': residual
                }
    for i in valid_index
    ]

    train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=cfg.batch_size, shuffle=True,drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size = 2 * cfg.batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_data, num_workers=4, batch_size = 2 * cfg.batch_size, shuffle=False)


    all_targets=[]
    for x in test_loader:
        all_targets.append(x['observed_data'][:,cfg.history_len:,:,:])
    test_target_tensor = torch.cat(all_targets) #N,T,K,1
    test_target_tensor=scaler.inverse_transform(test_target_tensor)
    print('test_target_tensor size:',test_target_tensor.size())
    all_targets=[]
    for x in val_loader:
        all_targets.append(x['observed_data'][:,cfg.history_len:,:,:])
    val_target_tensor = torch.cat(all_targets)
    val_target_tensor=scaler.inverse_transform(val_target_tensor)
    print('val_target_tensor size:',val_target_tensor.size())    



    return train_loader,val_loader,test_loader,val_target_tensor,test_target_tensor,scaler










def load_dataset_FFT_SST(cfg):

    folder_path = cfg.data.path

    
    data1 = np.load(folder_path+".npy")
    data2 = np.load(folder_path+"_ERA5.npy")
    B, H, W = data1.shape
    data1 = data1.reshape(B, -1, 1)
    B, H, W = data2.shape
    data2 = data2.reshape(B, -1, 1)
    data_ts1 = np.array([(i % 12, 0) for i in range(data1.shape[0])]) # 1850-1  _  2014-12 
    data_ts2 = np.array([(i % 12, 0) for i in range(data2.shape[0])]) # 1940-1  _  2025-2



    data_value1 = torch.tensor(data1).float()#.to(cfg.data.device)
    data_ts1 = torch.tensor(data_ts1).float()#.to(cfg.data.device)

    data_value2 = torch.tensor(data2).float()#.to(cfg.data.device)
    data_ts2 = torch.tensor(data_ts2).float()#.to(cfg.data.device)


    print('shape:',data_value1.shape)


    scaler = StandardScaler()
    scaler.fit(data_value1)
    data_value1 = scaler.transform(data_value1)
    L, K,C = data_value1.shape

    data_value2 = scaler.transform(data_value2)
    L2, K2,C2 = data_value2.shape

    data_value2_val=data_value2[:int(30*12),:,:]
    data_value2_test=data_value2[int(30*12):,:,:]

    data_ts2_val=data_ts2[:int(30*12),:]
    data_ts2_test=data_ts2[int(30*12):,:]


    train_num = L - (cfg.history_len + cfg.predict_len) + 1
    valid_num = int(30*12) - (cfg.history_len + cfg.predict_len) + 1
    test_num = L2-int(30*12)- (cfg.history_len + cfg.predict_len) + 1

    print("number of training samples:{0}".format(train_num))
    print("number of validation samples:{0}".format(valid_num))
    print("number of test samples:{0}".format(test_num))

    train_index = []
    for t in range(cfg.history_len, train_num + cfg.history_len):
        index = (t-cfg.history_len, t, t+cfg.predict_len)
        train_index.append(index)
    valid_index = []
    for t in range(cfg.history_len, valid_num + cfg.history_len):
        index = (t-cfg.history_len, t, t+cfg.predict_len)
        valid_index.append(index)
    test_index = []
    for t in range(cfg.history_len, test_num + cfg.history_len):
        index = (t-cfg.history_len, t, t+cfg.predict_len)
        test_index.append(index)

    data_train=data_value1.cpu().numpy()

    residual = fft_decomposition(data_train, threshold=0.1)
    residual=torch.from_numpy(np.std(residual,axis=(0,2)))


    mask=torch.ones(cfg.predict_len + cfg.history_len,K,C)
    mask[cfg.history_len:]=0

    train_data = [
    {
        'observed_data': data_value1[i[0]:i[0] + cfg.predict_len + cfg.history_len], 
        'gt_mask': mask,
        'timepoints': data_ts1[i[0]:i[0] + cfg.predict_len + cfg.history_len],
        'scale_residual':residual
                } 
    for i in train_index
                ]
    test_data = [
    {
        'observed_data': data_value2_test[i[0]:i[0] + cfg.predict_len + cfg.history_len], 
        'gt_mask': mask,
        'timepoints': data_ts2_test[i[0]:i[0] + cfg.predict_len + cfg.history_len],
        'scale_residual': residual
                }
    for i in test_index
    ]
    val_data = [
    {
        'observed_data': data_value2_val[i[0]:i[0] + cfg.predict_len + cfg.history_len], 
        'gt_mask': mask,
        'timepoints': data_ts2_val[i[0]:i[0] + cfg.predict_len + cfg.history_len],
        'scale_residual': residual
                }
    for i in valid_index
    ]

    train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=cfg.batch_size, shuffle=True,drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size = 2 * cfg.batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_data, num_workers=4, batch_size = 2 * cfg.batch_size, shuffle=False)


    all_targets=[]
    for x in test_loader:
        all_targets.append(x['observed_data'][:,cfg.history_len:,:,:])
    test_target_tensor = torch.cat(all_targets) #N,T,K,1
    test_target_tensor=scaler.inverse_transform(test_target_tensor)
    print('test_target_tensor size:',test_target_tensor.size())
    all_targets=[]
    for x in val_loader:
        all_targets.append(x['observed_data'][:,cfg.history_len:,:,:])
    val_target_tensor = torch.cat(all_targets)
    val_target_tensor=scaler.inverse_transform(val_target_tensor)
    print('val_target_tensor size:',val_target_tensor.size())    



    return train_loader,val_loader,test_loader,val_target_tensor,test_target_tensor,scaler




















def load_dataset_SST(cfg):

    folder_path = cfg.data.path

    
    data1 = np.load(folder_path+".npy")
    data2 = np.load(folder_path+"_ERA5.npy")
    B, H, W = data1.shape
    data1 = data1.reshape(B, -1, 1)
    B, H, W = data2.shape
    data2 = data2.reshape(B, -1, 1)
    data_ts1 = np.array([(i % 12, 0) for i in range(data1.shape[0])]) # 1850-1  _  2014-12 
    data_ts2 = np.array([(i % 12, 0) for i in range(data2.shape[0])]) # 1940-1  _  2025-2



    data_value1 = torch.tensor(data1).float()#.to(cfg.data.device)
    data_ts1 = torch.tensor(data_ts1).float()#.to(cfg.data.device)

    data_value2 = torch.tensor(data2).float()#.to(cfg.data.device)
    data_ts2 = torch.tensor(data_ts2).float()#.to(cfg.data.device)


    print('shape:',data_value1.shape)


    scaler = StandardScaler()
    scaler.fit(data_value1)
    data_value1 = scaler.transform(data_value1)
    L, K,C = data_value1.shape

    data_value2 = scaler.transform(data_value2)
    L2, K2,C2 = data_value2.shape

    data_value2_val=data_value2[:int(30*12),:,:]
    data_value2_test=data_value2[int(30*12):,:,:]

    data_ts2_val=data_ts2[:int(30*12),:]
    data_ts2_test=data_ts2[int(30*12):,:]


    train_num = L - (cfg.history_len + cfg.predict_len) + 1
    valid_num = int(30*12) - (cfg.history_len + cfg.predict_len) + 1
    test_num = L2-int(30*12)- (cfg.history_len + cfg.predict_len) + 1

    print("number of training samples:{0}".format(train_num))
    print("number of validation samples:{0}".format(valid_num))
    print("number of test samples:{0}".format(test_num))

    train_index = []
    for t in range(cfg.history_len, train_num + cfg.history_len):
        index = (t-cfg.history_len, t, t+cfg.predict_len)
        train_index.append(index)
    valid_index = []
    for t in range(cfg.history_len, valid_num + cfg.history_len):
        index = (t-cfg.history_len, t, t+cfg.predict_len)
        valid_index.append(index)
    test_index = []
    for t in range(cfg.history_len, test_num + cfg.history_len):
        index = (t-cfg.history_len, t, t+cfg.predict_len)
        test_index.append(index)

    data_train=data_value1.cpu().numpy()

    residual = fft_decomposition(data_train, threshold=0.1)
    residual=torch.from_numpy(np.std(residual,axis=(0,2)))


    mask=torch.ones(cfg.predict_len + cfg.history_len,K,C)
    mask[cfg.history_len:]=0

    train_data = [
    {
        'observed_data': data_value1[i[0]:i[0] + cfg.predict_len + cfg.history_len], 
        'gt_mask': mask,
        'timepoints': data_ts1[i[0]:i[0] + cfg.predict_len + cfg.history_len],
        'scale_residual':residual
                } 
    for i in train_index
                ]
    test_data = [
    {
        'observed_data': data_value2_test[i[0]:i[0] + cfg.predict_len + cfg.history_len], 
        'gt_mask': mask,
        'timepoints': data_ts2_test[i[0]:i[0] + cfg.predict_len + cfg.history_len],
        'scale_residual': residual
                }
    for i in test_index
    ]
    val_data = [
    {
        'observed_data': data_value2_val[i[0]:i[0] + cfg.predict_len + cfg.history_len], 
        'gt_mask': mask,
        'timepoints': data_ts2_val[i[0]:i[0] + cfg.predict_len + cfg.history_len],
        'scale_residual': residual
                }
    for i in valid_index
    ]

    train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=cfg.batch_size, shuffle=True,drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size = 2 * cfg.batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_data, num_workers=4, batch_size = 2 * cfg.batch_size, shuffle=False)


    all_targets=[]
    for x in test_loader:
        all_targets.append(x['observed_data'][:,cfg.history_len:,:,:])
    test_target_tensor = torch.cat(all_targets) #N,T,K,1
    test_target_tensor=scaler.inverse_transform(test_target_tensor)
    print('test_target_tensor size:',test_target_tensor.size())
    all_targets=[]
    for x in val_loader:
        all_targets.append(x['observed_data'][:,cfg.history_len:,:,:])
    val_target_tensor = torch.cat(all_targets)
    val_target_tensor=scaler.inverse_transform(val_target_tensor)
    print('val_target_tensor size:',val_target_tensor.size())    



    return train_loader,val_loader,test_loader,val_target_tensor,test_target_tensor,scaler







def load_dataset_Det(cfg):
    if cfg.data.name=='SST':
        return load_dataset_SST(cfg)
    else:
        return load_dataset(cfg)

def load_dataset_Diff(cfg):
    if cfg.data.name=='SST':
        return load_dataset_FFT_SST(cfg)
    else:
        return load_dataset_FFT(cfg)

   












