import math
import torch
import torch.nn as nn
import numpy as np
import earthaccess
from pyhdf.SD import SD, SDC
from skimage.metrics import peak_signal_noise_ratio
import os
import json

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant_(m.bias.data, 0.0)

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += peak_signal_noise_ratio(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

def data_augmentation(image, mode):
    out = np.transpose(image, (1,2,0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2,0,1))

def query_modis(
    date_range: tuple,
    bounding_box: tuple,
    cloud_cover: tuple,
    count: int,
    path: str,
):
    """Queries MODIS data using earth access

    Args:
        date_range (tuple): date range of query (YYYY-MM-DD)
        bounding_box (tuple): bounding coordinates to search (lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat)
        cloud_cover (tuple): range of percent cloud coverage (max, min)
        count (int): number to retrieve from query
        path (str): path to save data
    """
    
    # DOI Link: https://doi.org/10.5067/MODIS/MOD11A1.061
    # also see: https://search.earthdata.nasa.gov/search/granules?p=C1748058432-LPCLOUD&pg[0][v]=f&pg[0][gsk]=-start_date&ff=Available%20in%20Earthdata%20Cloud&tl=1699239359!3!!&fst0=Land%20Surface&lat=-83.40416670000002&long=-118.82965395
    # data format: https://lpdaac.usgs.gov/documents/715/MOD11_User_Guide_V61.pdf
    # bounding box params: lower left lon, lower left lat, upper right lon, upper right lat

    earthaccess.login()

    # Retrieve data
    datasets = earthaccess.search_data(
        doi="10.5067/MODIS/MOD11A1.061",
        cloud_hosted=True,
        temporal=date_range,
        bounding_box=bounding_box,
        cloud_cover=cloud_cover,
        count=count,
    )

    # Declare Paths
    path = os.path.join(path)

    assert len(os.listdir(path)) == 0, 'Save path not empty'        

    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(os.path.join(path, 'raw'))
    

    # Download files
    earthaccess.download(datasets, os.path.join(path, 'raw'))

    # Save query and metadata
    with open(os.path.join(os.path.join(path,'query.json')), "w") as query_file:
        json.dump({
            'query':{
                'date_range': date_range,
                'bounding_box': bounding_box,
                'cloud_cover': cloud_cover,
                'count':count
            },
            'metadata':dict(zip([i['meta']['native-id'] for i in datasets], datasets))
            }, 
            query_file
    )

def divide_image(image, tile_size, stride):
    """
    Divide an image into tiles with a given tile size and stride.
    
    Parameters:
        image (numpy.ndarray): Input image.
        tile_size (tuple): Size of each tile in the format (height, width).
        stride (tuple): Stride of the sliding window in the format (vertical, horizontal).
    
    Returns:
        List of numpy.ndarray: List of divided tiles.
    """
    tiles = []
    height, width, = image.shape
    tile_height, tile_width = tile_size
    stride_vertical, stride_horizontal = stride

    for y in range(0, height - tile_height + 1, stride_vertical):
        for x in range(0, width - tile_width + 1, stride_horizontal):
            tile = image[y:y+tile_height, x:x+tile_width]
            tiles.append(tile)

    return tiles

def get_lst_day(hdf_path: str) -> np.array:
    """Produces array of lst (kelvin) of hdf file

    Args:
        hdf_path (str): path to hdf file

    Returns:
        np.array: array of LST in kelvin
    """
    ds = SD(hdf_path, SDC.READ)
    data = np.array(ds.select("LST_Day_1km").get()) * 0.02  # in kelvin
    return data

