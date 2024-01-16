"""
@File           : mdl_io.py
@Author         : Gefei Kong
@Time:          : 20.04.2023 18:07
------------------------------------------------------------------------------------------------------------------------
@Description    : as below

"""
import os
import json
import laspy
import numpy as np
import geopandas as gpd

def load_cloud(filepath:str)->np.ndarray:
    """
    load roof_id's multi planes' point cloud data
    :param filepathes: [path_plane_0, path_plane_1,...]
    :return:
           roof_planes: [m, n, 3]
                        m: the number of planes
                        n: the number of points in a plane
                        3: xyz
    """
    cloud_ij = laspy.read(filepath)
    xyz_ij = np.vstack((cloud_ij.x, cloud_ij.y, cloud_ij.z)).transpose()

    return xyz_ij


def load_json(filepath:str) -> dict:
    """
    load json file
    :param filepath:
    :return:
    """
    with open(filepath, encoding="utf-8", mode="r+") as rlf:
        info = json.load(rlf)

    return info


def load_shp(shp_path:str) -> gpd.GeoDataFrame:
    gpd_df = gpd.read_file(shp_path)
    return gpd_df


def gpd2shp(savename:str, gpd_df:gpd.GeoDataFrame):
    gpd_df.to_file(savename)


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    return folder_path