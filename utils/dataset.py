import os
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2

CATEGORY_TO_TARGET = {"5": 0,
                      "3": 1} 

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, val_transform=None):
      
        self.anns = pd.DataFrame(
            list(map(lambda x: (x.split("_")[0], x.split("_")[1],x), os.listdir(img_dir))),
             columns = ["type","category","img_name"])
        if transform:
          self.anns = self.anns[self.anns.type == "train2017"]
        if val_transform:
          self.anns = self.anns[self.anns.type == "val2017"]

        self.transform = transform
        self.val_transform = val_transform
        self.img_dir = img_dir
        self.transform = transform
        self.val_transform = val_transform

    def read_image(self, fpath):
        return cv2.imread(fpath)

    def __len__(self):
        return self.anns.shape[0]

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.anns.iloc[idx]["img_name"])
        image = self.read_image(img_path)
        label = self.anns.iloc[idx]["category"]
        label = CATEGORY_TO_TARGET[label]
        if self.transform:
            image = self.transform(image = image)["image"]
        if self.val_transform:
            image = self.val_transform(image = image)["image"]
        return image, label



# training_data = CustomImageDataset(img_dir = "/content/images", transform = transform_train)
# val_data = CustomImageDataset(img_dir = "/content/images", val_transform=transform_val)

# train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
# val_dataloader = DataLoader(val_data, batch_size=64, shuffle=False)

# for i in training_data:
#   break

# plt.imshow(i[0]["image"])