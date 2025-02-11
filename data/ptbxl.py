from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import wfdb
import ast
import os
from torchvision import transforms
import torch

import lightning.pytorch as pl



class PTBXL(Dataset):
    
    def __init__(self, data_root, 
                folds=[1,2,3,4,5,6,7,8,9,10], 
                class_map = {"NORM":0, "MI":1, "STTC":2, "CD":3, "HYP":4},
                sampling_rate = 500,
                verbose=False,
                transform=None
                ):
        
            self.data_root = data_root
            y = pd.read_csv(os.path.join(self.data_root, 'ptbxl_database.csv'), index_col='ecg_id')
            self.folds = folds
            self.class_map = class_map
            self.sampling_rate = sampling_rate
            self.verbose = verbose
            self.transform = transform

            y = y.loc[y.strat_fold.isin(self.folds)]

             # Load scp_statements.csv for diagnostic aggregation
            agg_df = pd.read_csv(os.path.join(data_root, "scp_statements.csv"), index_col=0)
            self.agg_df = agg_df[agg_df.diagnostic == 1]

            # Apply diagnostic superclass
            y.scp_codes = y.scp_codes.apply(lambda x: ast.literal_eval(x))
            y['diagnostic_superclass'] = y.scp_codes.apply(self.aggregate_diagnostic)

            # Convert to Class numbers
            y["class_ids"] = y.diagnostic_superclass.apply(self.map_class_num)

            self.y = y

            if self.verbose:
                print("unique super classes=", self.agg_df.diagnostic_class.unique())
                print("unique folds=",self.y.strat_fold.unique())
                print(self.agg_df)
                print(self.y.scp_codes)
                print("Class labels=", self.y.diagnostic_superclass)
                print("Class ids=", self.y.class_ids)

    def aggregate_diagnostic(self, y_dic):
        tmp = []
        #print(y_dic)
        for key in y_dic.keys():
            if key in self.agg_df.index:
                tmp.append(self.agg_df.loc[key].diagnostic_class)
        #print("temp =",  tmp)
        return list(set(tmp))

    def map_class_num(self, class_labels):
        temp = []
        try:
            for l in class_labels:
                class_id = self.class_map[l]
                temp.append(class_id)
        except:
            print("These labels are wrong:", class_labels)
        return temp

    def read_row_data(self, data_path):
        signal, meta = wfdb.rdsamp(data_path)
        #data = np.array([signal for signal, meta in data])
        if self.verbose:
            print(signal)
            print(meta)
        return np.array(signal), meta
        
    
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        y_row = self.y.iloc[idx]
        class_ids = y_row.class_ids

        class_encoded = np.zeros(len(self.class_map))
        class_encoded[class_ids] = 1

        if self.verbose:
            print(class_ids)
            print(class_encoded)

        # To get sample rate 100 ECGs
        if self.sampling_rate == 100:
            data_path = os.path.join(self.data_root, y_row.filename_lr)
            ecg, meta = self.read_row_data(data_path)
        # To get sample rate 500 ECGs
        elif self.sampling_rate == 500:
            data_path = os.path.join(self.data_root, y_row.filename_hr)
            ecg, meta = self.read_row_data(data_path)

        else:
            print("Wrong sample rate")
            exit

        # Get transpose
        #print(ecg.shape)
        ecg = ecg.transpose()
        #print(ecg.shape)
        ecg = torch.from_numpy(ecg).to(torch.float32)
        class_encoded = torch.from_numpy(class_encoded).to(torch.float32)

        if self.transform: # Not in use
            ecg = self.transform(ecg)
        
        #print("ecg_shape=", ecg.shape)
        sample = {"ecg":ecg, "class":class_encoded }
        return sample




class PTBXLDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, train_folds:list =[1,2,3,4,5], 
                val_folds:list = [6,7,8], 
                test_folds:list = [9, 10], 
                predict_folds:list = [6, 7, 8, 9, 10],
                sampling_rate:int = 100,
                bs = 32
                ):
        super().__init__()

        self.transform = None #transforms.Compose([transforms.ToTensor()])
        self.root_dir = root_dir
        self.train_folds = train_folds
        self.val_folds = val_folds
        self.test_folds = test_folds
        self.predict_folds = predict_folds
        self.bs = bs

    def prepare_data(self):
        pass

    def setup(self, stage:str):
        if stage == "fit":
           self.ptbxl_train = PTBXL(self.root_dir, folds=self.train_folds, transform=self.transform)
           self.ptbxl_val = PTBXL(self.root_dir, folds=self.val_folds, transform=self.transform)

        if stage == "test":
            self.ptbxl_test = PTBXL(self.root_dir, folds=self.test_folds, transform=self.transform)

        if stage == "predict":
            self.ptbxl_predict = PTBXL(self.root_dir, folds=self.predict_folds, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.ptbxl_train, batch_size=self.bs)

    def val_dataloader(self):
        return DataLoader(self.ptbxl_val, batch_size=self.bs)

    def test_dataloader(self):
        return DataLoader(self.ptbxl_test, batch_size=self.bs)

    def predict_dataloader(self):
        return DataLoader(self.ptbxl_predict, batch_size=self.bs)
    


if __name__=="__main__":
    data_root = "/global/D1/homes/vajira/data/ecg/ptbxl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    data = PTBXL(data_root, verbose=False)
    print(len(data))
    print(data[300])