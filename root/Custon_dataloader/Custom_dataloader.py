import glob
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch


class CustomDataset(Dataset):
    """
    path_to_data - где находятся папка с классами
    """

    def __init__(self, path_to_data) -> None:
        self.imgs_path = path_to_data
        file_list = glob.glob(self.imgs_path + "*")
        file_list.sort()
        self.data = []
        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            for img_path in glob.glob(class_path + "/*.png"):
                self.data.append([img_path, class_name])

        self.class_map = {}
        for i in range(len(file_list)):
            self.class_map.update({file_list[i].split("/")[-1]: i})

        self.img_dim = (128, 128)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str: torch.tensor, str: dict[str, int]]:
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32)
        img = img / 255

        img = cv2.resize(img, self.img_dim, interpolation=cv2.INTER_AREA)
        img = img[None, :, :]

        class_id = self.class_map[class_name]
        # print(img.shape)
        img_tensor = torch.from_numpy(img)
        # img_tensor = img_tensor.permute(2, 0, 1)
        class_id = torch.tensor(class_id)

        return {'img': img_tensor, 'label': class_id}
