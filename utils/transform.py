import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import matplotlib.pyplot as plt

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SIZE = 50

# Declare an augmentation pipeline
transform_train = A.Compose([
    A.LongestMaxSize(max_size = SIZE),
    A.PadIfNeeded(min_height = SIZE, min_width = SIZE, value = 0, border_mode = cv2.BORDER_CONSTANT ),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean = MEAN, std = STD),
    ToTensorV2()
])

transform_val = A.Compose([
    A.LongestMaxSize(max_size = SIZE),
    A.PadIfNeeded(min_height = SIZE, min_width = SIZE, value = 0, border_mode = cv2.BORDER_CONSTANT ),
    A.Normalize(mean = MEAN, std = STD),
    ToTensorV2()
])


# img = io.imread(coco.loadImgs(imgIds[2])[0]["coco_url"])
# img.shape
# plt.imshow(img)
# plt.show()
# plt.imshow(transform_train(image = img)["image"].numpy().transpose(1, 2, 0))
