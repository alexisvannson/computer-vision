import os
import shutil  #for file operations (faster than subprocess)
import pandas as pd
from torchvision import transforms, datasets
from torch.utils.data import Dataset

class DatasetLoader(Dataset):
    def __init__(self, data_path='./data', label_path='./data/train_labels.csv', resize_value=128, gray_scale=False, skip_creation=False, skip_path= None):
        self.dataset_path = os.path.join(data_path, 'Dataset')
        self.labels = pd.read_csv(label_path, dtype={"Id": str})
        if not skip_creation:
            os.makedirs(self.dataset_path, exist_ok=True)
            
            for index, row in self.labels.iterrows():
                label_dir = os.path.join(self.dataset_path, str(row['Label']))
                os.makedirs(label_dir, exist_ok=True)
                
                img_filename = row['Id']+ '.png'
                src_path = os.path.join(data_path, 'train', img_filename)
                dst_path = os.path.join(label_dir, img_filename)
                
                if os.path.exists(src_path):
                    shutil.copy(src_path, dst_path)
                else:
                    print(f"Warning: Image not found at {src_path}")
        
        self.resize_value = resize_value
        self.transform = transforms.Compose([
            transforms.Resize((self.resize_value, self.resize_value)),
            transforms.Grayscale(num_output_channels=1) if gray_scale else transforms.Lambda(lambda x: x),
            transforms.ToTensor()
        ])
        
        self.dataset = datasets.ImageFolder(skip_path if skip_creation else self.dataset_path, transform=self.transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


if __name__ == "__main__":
    dataset = DatasetLoader(skip_creation=True, skip_path='data/Dataset')
    print(len(dataset))
    print(dataset[0])