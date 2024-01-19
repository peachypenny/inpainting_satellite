import earthaccess
import pandas as pd
import geopandas as gpd
from pyhdf.SD import SD, SDC
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import json


def _read_flag(value: int) -> int:
    """Reads flags of MODIS data 

    Args:
        value (int): value of MODIS pixel

    Returns:
        _type_: data flag
    """
    bit_string = bin(value)
    if bit_string[-2::] == "00" or bit_string[-2::] == "01":  # lst produced
        return 0
    elif bit_string[-2::] == "10":  # lst not produced, cloud covered
        return 1
    elif bit_string[-2::] == "11":  # lst not produced, other
        return -1
    else:
        return value


def get_cloud_mask(hdf_path: str) -> np.array:
    """Creates cloud mask from data flags. Maps over array using _read_flag

    Args:
        hdf_path (str): path to hdf file

    Returns:
        np.array: cloud mask: 0-lst, 1-cloud, -1-error
    """
    ds = SD(hdf_path, SDC.READ)
    data = np.array(ds.select("QC_Day").get())
    vectorized = np.vectorize(_read_flag)
    qc_day_filtered = vectorized(data)
    return qc_day_filtered


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


def get_position(hdf_path: str) -> dict:
    """Provides position of hdf file

    Args:
        hdf_path (str): path to hdf file

    Returns:
        dict: 
    """
    ds = SD(hdf_path, SDC.READ)
    group = "GROUP=GRINGPOINT"
    end_group = "END_GROUP=GRINGPOINT"
    lon_str = "OBJECT=GRINGPOINTLONGITUDE"
    lat_str = "OBJECT=GRINGPOINTLATITUDE"
    seq_str = "OBJECT=GRINGPOINTSEQUENCENO"

    meta = ds.attributes().get("CoreMetadata.0").split("\n")
    meta = [str.strip(x).replace(" ", "") for x in meta]

    pos = meta[meta.index(group) : meta.index(end_group)]
    pos_dict = {}

    for obj in [lon_str, lat_str, seq_str]:
        str_list = meta[meta.index(obj) + 3].strip("VALUE=")[1:-1].split(",")
        pos_dict[obj[7::]] = tuple(map(float, str_list))

    return pos_dict


def parse_modis_label(path):
    labels = path.split(".")
    return {
        "product_name": labels[0],
        "acquisition_date": labels[1],
        "acquisition_time": labels[2],
        "collection_version": labels[3],
        "production_date": labels[4],
    }


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
    data_path = os.path.join(path, "data")

    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(data_path)

    # Download files
    earthaccess.download(datasets, data_path)

    # Save query
    query = {
        "date": date_range,
        "bounding_box": bounding_box,
        "cloud_cover": cloud_cover,
        "count": count,
    }

    with open(os.path.join(path, "query.json"), "w") as query_file:
        json.dump(query, query_file)

def geneate_lst_image(img_path: str, save_path: str) -> None:
    """Creates jpg images from image

    Args:
        img_path (str): Path to hdf file 
        save_path (str): Directory to save image
    """
    lst = get_lst_day(img_path)
    img_path = os.path.basename(img_path)
    plt.imsave(
        os.path.join(
            save_path, img_path.replace(img_path.split(".")[-1], "") + "jpg"
        ),
        lst,
        cmap="gray",
    )

def generate_mask(img_path: str, save_path: str) -> None:
    """Creates binary mask from image

    Args:
        img_path (str): Path to hdf file
        save_path (str): Directory to save image
    """
    sample_mask = get_cloud_mask(img_path)
    img_path = os.path.basename(img_path)
    plt.imsave(
        os.path.join(
            save_path, img_path.replace(img_path.split(".")[-1], "") + "jpg"
        ),
        sample_mask,
        cmap="gray",
    )

def retile(img: np.array, tile_size: tuple) -> list:
    """Creates subtiles of images. Assumes dimension of original is divisible by height/width

    Args:
        img (np.array): original image
        tile_size (tuple): desired size of tiles (h, w)
        image_size (tuple): original image size (h, w)

    Returns:
        list: list containing np.arrays of tiles
    """
    h, w = tile_size[0], tile_size[1]
    return [
        img[x : x + h, y : y + w]
        for x in range(0, img.shape[0]//h)
        for y in range(0, img.shape[1]//w)
    ]
