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
        image_size=(1200, 1200),
        tile_size=(128,128),
        data_filter=None,
        img_filter=lambda x: get_percent_coverage(x) == 0, 
        from_existing=False
    ):
        """Creates pytorch dataset from MODIS hdf files.

        Args:
            data_path (str): Path to save data
            transform (Any, optional): Specify transform here. Defaults to None.
            image_size (tuple, optional): Image size. Defaults to (1200,1200).
            tile_size (tuple, optional): Desired size of masks/images.
            data_filter (func): Boolean function filtering samples by set criteria. 
            img_filter (func): Boolean function filtering ground_truth by set criteria
            from_existing (bool, optional): Specify if importing pre-processed dataset. Defaults to False.
        """
        self.data_path = os.path.join(data_path)
        self.transform = transform
        self.image_size = image_size
        self.tile_size = tile_size
        self.collection = pd.DataFrame(
            columns=["sample", "ground_truth"]
        )  # Maps masks to source image
        self.data_filter = data_filter
        self.img_filter = img_filter

        # For pre-processed datasets
        if from_existing:
            self.collection = pd.read_csv(os.path.join(data_path, "collection.csv"))
            return
        
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
        print('Generating ground truth...')
        imgs = {}
        for img in os.listdir(image_path):
            img_date = parse_modis_label(img)['production_date']
            img_tiles = retile(get_lst_day(os.path.join(self.image_path, img)), self.tile_size) # Generate tiles
            
            # Save ground truth
            for i in range(len(img_tiles)):
                if self.img_filter is not None and self.img_filter(img_tiles[i]): # check if ground truth has no cloud coverage
                    tile_filename = img_date + '_{0}'.format(i)
                    imgs[tile_filename] = img_tiles[i]
                    
                    np.save(os.path.join(self.data_path, DATA_IMAGE_DIR, tile_filename + '.npy'), img_tiles[i])
        print(len(imgs))
        
        # Retile masks and apply to ground truth
        print('Creating masks and generating samples...')
        for mask in os.listdir(mask_path):
            cloud_mask = get_cloud_mask(os.path.join(self.mask_path, mask))
            mask_date = parse_modis_label(mask)['production_date']

            # Checks for error flags
            assert list(np.unique(cloud_mask)) == [0, 1], 'Mask has data error'

            masked_tiles = retile(cloud_mask, self.tile_size)
            counter = 0
            for masked_tile_idx in range(len(masked_tiles)):
                if not self.data_filter or not self.data_filter(masked_tiles[masked_tile_idx]):   # Filter masks based on criteria 
                    continue

                for name in imgs.keys():
                    masked = np.multiply(masked_tiles[masked_tile_idx], imgs[name])
                    masked_filename = mask_date + '_{0}{1}_'.format(masked_tile_idx, counter) + name + '.npy'

                    np.save(
                        os.path.join(self.data_path, DATA_MASK_DIR, masked_filename),
                        masked,
                    )
                    
                    # Save to collection
                    self.collection.loc[len(self.collection.index)] = [ 
                        masked_filename,
                        name + '.npy',
                    ]
                    
                    counter += 1

    
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

        return mask, image

class LSTDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        mask_path: str,
        save_path: str
    ):
        """LSTDataset

        Args:
            data_path (str): path to data
            mask_path (str): path to masks
            save_path (str): path to save masked data
        """
        self.data_path = os.path(data_path)
        self.mask_path = os.path(mask_path)
        self.save_path = os.path(save_path)
        self.collection = pd.DataFrame(
            columns=["sample", "ground_truth"]
        ) 
        
        for image_path in os.listdir(data_path):
            image = np.load(os.path.join(self.data_path, image_path))
            for mask in os.listdir(mask_path):
                cloud_mask = np.load(os.path.join(mask_path, mask)) # Retrieve mask
                sample = np.multiply(image, cloud_mask) # Create sample
                sample_name = os.path.splitext(os.path.basename(image_path))[0] + '_' + \
                    os.path.splitext(os.path.basename(mask))[0] + '.npy'
                np.save(os.path.join(self.save_path, sample_name), sample) # Save sample
                
                self.collection.loc[len(self.collection.index)] = [ 
                    sample_name,
                    image_path,
                ]       
                 
    def __len__(self):
        return len(self.collection)
    
    def __getitem__(self, index):
        mask = np.load(
            os.path.join(self.save_path, 
                         self.collection.iloc[index]["sample"]))
        image = np.load(
            os.path.join(self.data_path,
                         self.collection.iloc[index]["ground_truth"])
            )
        
        return mask, image

                    

        