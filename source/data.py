import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from utils import *

DATA_MASK_DIR = "masks"
DATA_IMAGE_DIR = "images"


class ModisDataset(Dataset):
    in_channel = 1
    out_channel = 1

    def __init__(
        self,
        data_path,  # save directory
        image_path=None,  # ground_truth hdf path
        mask_path=None,  # cloud_covered hdf path
        transform=None,
        target_transform=None,
        image_size=1200,
        from_existing=False,
    ):
        self.data_path = os.path.join(data_path)
        self.transform = transform
        self.target_transform = target_transform
        self.image_size = image_size
        self.collection = pd.DataFrame(
            columns=["sample", "ground_truth"]
        )  # Maps images to original

        # For pre-processed datasets
        if from_existing:
            self.collection = pd.read_csv(os.path.join(data_path, "file_mappings.csv"))
        else:
            self.image_path = os.path.join(image_path)
            self.mask_path = os.path.join(mask_path)

            # Create directories for root, root/images, root/mask
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)
            if not os.path.exists(os.path.join(self.data_path, DATA_MASK_DIR)):
                os.makedirs(os.path.join(self.data_path, DATA_MASK_DIR))
            if not os.path.exists(os.path.join(self.data_path, DATA_IMAGE_DIR)):
                os.makedirs(os.path.join(self.data_path, DATA_IMAGE_DIR))

            # Construct masked images
            for tile in os.listdir(image_path):
                tile_lst = get_lst_day(os.path.join(self.image_path, tile))
                tile_filename = tile[0:-4] + ".npy"

                np.save(
                    os.path.join(self.data_path, DATA_IMAGE_DIR, tile_filename),
                    tile_lst
                )

                # Retrieve masks from .hdf files
                for mask in os.listdir(mask_path):
                    cloud_mask = get_cloud_mask(os.path.join(self.mask_path, mask))

                    # Checks for error flags
                    assert list(np.unique(cloud_mask)) == [0, 1], "Mask has data error"

                    # Apply mask over image and save as image
                    masked = np.multiply(cloud_mask, tile_lst)
                    masked_filename = (
                        parse_modis_label(mask)["production_date"] + "." + tile_filename
                    )

                    np.save(
                        os.path.join(self.data_path, DATA_MASK_DIR, masked_filename),
                        masked,
                    )

                    self.collection.loc[len(self.collection.index)] = [
                        masked_filename,
                        tile_filename,
                    ]

            # Save file name mapping
            self.collection.to_csv(os.path.join(self.data_path, "file_mappings.csv"))

    def __len__(self):
        return len(self.collection)

    def __getitem__(self, index):
        mask = np.load(
            os.path.join(
                self.data_path, DATA_MASK_DIR, self.collection.iloc[index]["sample"]
            )
        )
        image = np.load(
            os.path.join(
                self.data_path,
                DATA_IMAGE_DIR,
                self.collection.iloc[index]["ground_truth"],
            )
        )

        if self.transform:
            mask = self.transform(mask)

        if self.target_transform:
            image = self.target_transform(mask)

        mask = torch.from_numpy(mask.astype(np.float32))
        image = torch.from_numpy(image.astype(np.float32))

        return mask, image
