import cv2
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image 

class UTKFaceDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, PATH_IMG, transform=None):
        list_paths = glob.glob(PATH_IMG + '*/*')
        self.list_paths = list_paths
        list_files_name = [x.split('/')[-1].split('.')[0] for x in list_paths]
        self.list_age = [int(x.split('_')[0]) for x in list_files_name]

        self.transform = transform

    def __len__(self):
        # return 100
        return len(self.list_paths)

    def __getitem__(self, idx):
        # image1 = cv2.imread(self.list_paths[idx])
        # image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image1 = Image.open(self.list_paths[idx])
                
        if self.transform:
            image1 = self.transform(image1)


        age_temp = int (self.list_age[idx]/10)
        if age_temp > 5:
            age_temp = 5

        return image1, torch.tensor(age_temp)