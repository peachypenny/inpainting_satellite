import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
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
        image_size=(1200, 1200),
        tile_size=(128,128),
        data_filter=None,
        from_existing=False
    ):
        """Creates pytorch dataset from MODIS hdf files.

        Args:
            data_path (str): Path to save data
            target_transform (str, optional): Specify transform here. Defaults to None.
            image_size (tuple, optional): Image size. Defaults to (1200,1200).
            tile_size (tuple, optional): Desired size of masks/images.
            data_filter (func): Boolean function filtering by set criteria. 
            from_existing (bool, optional): Specify if importing pre-processed dataset. Defaults to False.
        """
        self.data_path = os.path.join(data_path)
        self.transform = transform
        self.target_transform = target_transform
        self.image_size = image_size
        self.tile_size = tile_size
        self.collection = pd.DataFrame(
            columns=["sample", "ground_truth"]
        )  # Maps masks to source image
        self.data_filter = data_filter

        # For pre-processed datasets
        if from_existing:
            self.collection = pd.read_csv(os.path.join(data_path, "collection.csv"))
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

            # Retile and save ground truth
            print('Generating images...')
            for img in os.listdir(image_path):
                img_lst = get_lst_day(os.path.join(self.image_path, img))
                img_date = parse_modis_label(img)['production_date']
                img_tiles = retile(img_lst, self.tile_size) # Generate tiles
                img_tile_names = []
                
                # Save ground truth
                for i in range(len(img_tiles)):
                    tile_filename = img_date + '_{0}'.format(i)
                    img_tile_names.append(tile_filename)
                    np.save(os.path.join(self.data_path, DATA_IMAGE_DIR, tile_filename + '.npy'), img_tiles[i])

                # Create masked images
                for mask in os.listdir(mask_path):
                    cloud_mask = get_cloud_mask(os.path.join(self.mask_path, mask))
                    mask_date = parse_modis_label(mask)['production_date']

                    # Checks for error flags
                    assert list(np.unique(cloud_mask)) == [0, 1], 'Mask has data error'

                    masked_tiles = retile(cloud_mask, self.tile_size)
                    
                    for masked_tile_idx in range(len(masked_tiles)):
                        for img_tile_idx in range(len(img_tiles)):
                            if self.data_filter is not None and self.data_filter(masked_tiles[masked_tile_idx]):   # Filter masks based on criteria 
                                masked = np.multiply(masked_tiles[masked_tile_idx], img_tiles[img_tile_idx])
                                masked_filename = mask_date + '_{0}{1}.'.format(masked_tile_idx, img_tile_idx) + img_tile_names[img_tile_idx] + '.npy'
                            
                                np.save(
                                    os.path.join(self.data_path, DATA_MASK_DIR, masked_filename),
                                    masked,
                                )
                                
                                # Save to collection
                                self.collection.loc[len(self.collection.index)] = [ 
                                    masked_filename,
                                    tile_filename + '.npy',
                                ]
                            else:
                                continue
            
            assert len(self.collection) == len(os.listdir(os.path.join(self.data_path, DATA_MASK_DIR))), 'Amount of masked files ({0}) does not match collection length ({1})'.format(len(os.path.join(self.data_path, DATA_MASK_DIR)), len(self.collection))
            # Save file name mapping
            print('Saving collection...')
            self.collection.to_csv(os.path.join(self.data_path, "collection.csv"))
            print('Dataset created!')
            print('Ground truth images: {0}'.format(len(np.unique(self.collection['ground_truth']))))
            print('Masked images created: {0}'.format(len(self.collection)))

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
