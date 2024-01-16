"""
@File           : mdl_eval.py
@Author         : Gefei Kong
@Time:          : 18.05.2023 14:15
------------------------------------------------------------------------------------------------------------------------
@Description    : as below
evaluation module
"""
import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import geopandas as gpd

from modules.eval_basic_ol import intersection_union, hausdorff_dis_v2, load_shp_FKB, load_shp_Isprs_v2
from utils import polis
from utils.mdl_geo import obj2Geo
from utils.mdl_io import load_json


def main_eval(res_folder: str,
              res_type:str,
              shp_gt_path:str,
              dataset_type:str,
              out_folder: str,
              res_base:str,
              bld_list:np.ndarray or list,
              is_save_res:bool):

    if dataset_type=="trd":
        poly_gt_eval = load_shp_FKB(shp_gt_path, bld_list)
    else:
        poly_gt_eval = load_shp_Isprs_v2(shp_gt_path, bld_list)

    num_bld = len(bld_list)
    bol_pred_res = []
    with tqdm(range(num_bld)) as pbar_blds:
        for bldi in pbar_blds:
            pbar_blds.set_description(f"[eval] :: bld_{bldi}-{bld_list[bldi]}")

            bldi_name = bld_list[bldi]

            ##############
            # load simplified polygon data
            ##############
            b_oli_path = os.path.join(res_folder, f"{bldi_name}{res_type}")
            b_oli_js = load_json(b_oli_path)
            b_oli = obj2Geo(b_oli_js)  # type: shapely.geometry.Polygon

            ##############
            # calculate iou
            ##############
            if dataset_type == "trd":
                b_oli_iou = intersection_union(b_oli, poly_gt_eval=poly_gt_eval, bid=int(bldi_name))
                b_oli_hd = hausdorff_dis_v2(b_oli, poly_gt_eval=poly_gt_eval, bid=int(bldi_name))
            else:
                b_oli_iou = intersection_union(b_oli, poly_gt_eval=poly_gt_eval, bid=bldi_name)
                b_oli_hd = hausdorff_dis_v2(b_oli, poly_gt_eval=poly_gt_eval, bid=bldi_name)

            ##############
            # add it to res_list
            ##############
            bol_pred_res.append([bldi_name, b_oli, b_oli_iou, b_oli_hd])

    ######################
    # save res to gpd
    ######################
    res_df = pd.DataFrame(bol_pred_res, columns=["bid", "geometry", "IOU", "HD"])
    res_df = res_df.dropna()

    ######################
    # save res to gpd
    ######################
    # print(res_df.shape)
    print(f"{dataset_type}'s mean_IOU: {res_df['IOU'].mean()}")
    print(f"{dataset_type}'s mean_HD: {res_df['HD'].mean()}")

    if is_save_res:
        savename = os.path.join(out_folder, f"{dataset_type}_{res_base}.shp")
        # save json
        res_overall_json = {"mean_IoU": res_df['IOU'].mean(), "mean_HD": res_df['HD'].mean()}
        with open(savename.replace(".shp", "_overall.json"), "w+") as res_js:
            json.dump(res_overall_json, res_js, indent=4)
        # save csv
        res_df.to_csv(savename.replace(".shp", ".csv"), index=False)
        # save shp
        res_df = gpd.GeoDataFrame(res_df, geometry=res_df.geometry)
        res_df.to_file(savename)

        # polis
        # only when saving res, calculate polis
        polis.score(shp_gt_path, savename, savename.replace(".shp", "_polis.shp"))


if __name__ == "__main__":
    # # Isprs dataset
    # dataset_type = "isprs"
    # shp_gt_path = "/home/gefeik/PhD-work/footprint polygon/footprint polygon/original materials/ISPRS dataset/Vaihingen-res/shp/isprs_gt.shp"
    # simp_res_folder = "/home/gefeik/PhD-work/footprint polygon/footprint_polygon_v2/res/main_all/isprs/to=0.50_di=0.10_haus_siou-1.00/2_sbol/"
    # bld_list = np.loadtxt("../config/isprs_test.txt", dtype="str")

    # trd dataset
    dataset_type = "trd"
    shp_gt_path = "/home/gefeik/PhD-work/footprint polygon/footprint polygon/own training/res/shp/true_shape/real_footprint.shp"
    simp_res_folder = "/home/gefeik/PhD-work/footprint polygon/footprint_polygon_v2/res/main_all_gu/trd/to=0.30_di=0.00_haus_siou-1.00/2_sbol"
    bld_list = np.loadtxt("../config/trd_test.txt").astype("int")
    print(os.path.dirname(simp_res_folder))
    out_folder = os.path.join(os.path.dirname(simp_res_folder), "eval_shp")
    print(out_folder)

    main_eval(res_folder=simp_res_folder,
              res_type=".json",
              shp_gt_path=shp_gt_path,
              dataset_type=dataset_type,
              out_folder=out_folder,
              res_base=os.path.basename(simp_res_folder),
              bld_list=bld_list,
              is_save_res=True)