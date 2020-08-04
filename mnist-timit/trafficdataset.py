import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class TrafficDataset(Dataset):

  def __init__(self, file_name):
    #read csv, load row data into variables
    file_out = pd.read_csv(file_name)
    file_out = file_out.drop(columns = ['Timestamp'])
    x = file_out.iloc[0:, 0:-1].values
    y = file_out.iloc[0:, -1].values
    
    #convert to torch tensors
    self.X = torch.tensor(x, dtype=torch.float32)
    self.y = torch.tensor(y)
  
  def __len__(self):
    return len(self.y)
  
  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]

def stratified_split(df, prop):
    IDs = df[['ID','Fraction_Observed']].drop_duplicates()
    rates = {.01,.02,.05,.1,.2}
    train_IDs = []
    test_IDs = []
    for rate in rates:
        rateIDs = IDs[IDs['Fraction_Observed']==rate]
        rateIDs['ID'].tolist()
        train_append = rateIDs[0:int(prop*len(rateIDs))]
        test_append = rateIDs[int(prop*len(rateIDs)):]
        train_IDs += train_append['ID'].tolist()
        test_IDs += test_append['ID'].tolist()
    train = df[df['ID'].isin(train_IDs)]
    test = df[df['ID'].isin(test_IDs)]
    train.to_csv('train.csv')
    test.to_csv('test.csv')
