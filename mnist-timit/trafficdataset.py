import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class FeatureDataset(Dataset):

  def __init__(self, file_name):
    #read csv, load row data into variables
    file_out = pd.read_csv(file_name)
    file_out = file_out.drop(columns = ['Timestamp'])
    x = file_out.iloc[0:-1, 0:-1].values
    y = file_out.iloc[0:-1, -1].values
    
    #convert to torch tensors
    self.X = torch.tensor(x, dtype=torch.float32)
    self.y = torch.tensor(y)
  
  def __len__(self):
    return len(self.y)
  
  def __getitem(self, idx):
    return self.X[idx], self.y[idx]
