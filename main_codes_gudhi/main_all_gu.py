"""
@File           : main_all_gu.py
@Author         : Gefei Kong
@Time:          : 05.06.2023 14:38
------------------------------------------------------------------------------------------------------------------------
@Description    : as below

"""

import os
import json

import numpy as np
import yaml

try:
    import mdl1_bolPH_gu,mdl2_simp_bol,mdl_eval
except:
    from main_codes_gudhi import mdl1_bolPH_gu, mdl2_simp_bol, mdl_eval

from utils.mdl_io import create_folder


def get_params(cfg_params:dict) -> (int, int, float, float, float, str):
    pre_cloud_num = cfg_params["pre_cloud_num"]
    down_sample_num = cfg_params["down_sample_num"]
    simp_type = cfg_params["simp"]["type"]
    thres_simpiou = -1
    if simp_type == "iou":
        thres_simpiou = cfg_params["simp"]["thres_iou"]


    # calculate other params
    bfr_tole = np.ceil(cfg_params["point_spacing"] * 10) / 10  # delta_rb+
    bfr_otdiff = np.floor(cfg_params["point_spacing"]/3*10)/10 # delta_rb-

    return pre_cloud_num, down_sample_num, bfr_tole, bfr_otdiff, thres_simpiou, simp_type


def get_data_pathes(cfg_data:dict,
                    bfr_tole:float,
                    bfr_otdiff:float,
                    simp_type:str,
                    thres_simpiou:float) -> (str, str, str, str, str, str, str, bool, str):
    # cfg_data = cfg["data"]
    ds_type = cfg_data["dataset"] # e.g. "isprs" or "trd"

    cloud_root_folder = cfg_data["input"]["cloud_folder"]
    cloud_list_path = cfg_data["input"]["cloud_list_path"]
    cloud_type = cfg_data["input"]["cloud_type"]

    out_root_folder = cfg_data["output"]["out_root_folder"]

    # get other folders for output, and create related folder at the same time.
    out_base_folder = os.path.join(out_root_folder, ds_type,
                                  f"to={bfr_tole:.2f}_di={bfr_otdiff:.2f}_{simp_type}_siou{thres_simpiou:.2f}")
    out_bol_folder =  create_folder(os.path.join(out_base_folder, "1_bol")) # return folder path itself
    out_simp_folder = create_folder(os.path.join(out_base_folder, "2_sbol"))
    out_eval_folder = create_folder(os.path.join(out_base_folder, "eval_shp"))

    # get the saved buffer_radius_optim values
    is_use_saved_bfr = False
    saved_bfr_optims_path = cfg_data["input"]["saved_bfr_optim_path"]
    if saved_bfr_optims_path != "":
        is_use_saved_bfr = True
    else:
        saved_bfr_optims_path = os.path.join(os.path.dirname(out_base_folder), f"{ds_type}_all_bfrs_optim.csv")
        if os.path.exists(saved_bfr_optims_path): # auto check whether the planned saving path exists. if yes, the bfrs_optim has been gotten.
            is_use_saved_bfr = True


    return ds_type, cloud_root_folder, cloud_list_path, cloud_type, \
           out_bol_folder, out_simp_folder, out_eval_folder, is_use_saved_bfr, saved_bfr_optims_path


def get_evalparams(cfg_eval:dict) -> (str,str):
    eval_gt_path = cfg_eval["eval_gt_path"]
    is_save_res  = cfg_eval["is_save_res"]

    return eval_gt_path, is_save_res

def main(cfg:dict):
    #########################
    # 1. get params and data-io paths.
    #########################
    # 1.1. params
    # bfr_tole: delta_rb+; bfr_otdiff: delta_rb-
    pre_cloud_num, down_sample_num, bfr_tole, bfr_otdiff, thres_simpiou, simp_type = get_params(cfg["params"])
    # 1.2. data pathes and out pathes
    ds_type, cloud_root_folder, cloud_list_path, cloud_type, \
    out_bol_folder, out_simp_folder, out_eval_folder, \
    is_use_saved_bfr, save_bfroptim_path = get_data_pathes(cfg["data"], bfr_tole, bfr_otdiff, simp_type, thres_simpiou)
    # 1.3 isdebug
    isdebug = cfg["params"].get("isDebug", False)

    #########################
    # 2. load point cloud data which whose building outlines will be extracted.
    #########################
    if "trd" in cloud_list_path:
        bld_list = np.loadtxt(cloud_list_path).astype("int")
    else:
        bld_list = np.loadtxt(cloud_list_path, dtype="str")

    #########################
    # 3. main workflow.
    #########################
    # 3.1 step 1: get basic oultines (bol) by using persistent homology (PH)
    mdl1_bolPH_gu.main_basicOL(cloud_root_folder, cloud_type, out_bol_folder,
                               bld_list,
                               pre_cloud_num, down_sample_num,
                               bfr_tole=bfr_tole, bfr_otdiff=bfr_otdiff,
                               is_use_saved_bfr=is_use_saved_bfr,
                               savename_bfr=save_bfroptim_path,
                               is_unrefresh_save=False,
                               is_Debug=isdebug)

    # 3.2 stpe 2: get simplified result based on the choosed simplification method,
    mdl2_simp_bol.main_simp_ol(out_bol_folder, data_basic_ol_type=".json",
                               dataset_type=ds_type,
                               out_folder=out_simp_folder,
                               bld_list=bld_list,
                               bfr_tole=bfr_tole, bfr_otdiff=bfr_otdiff,
                               simp_method=simp_type,
                               savename_bfr=save_bfroptim_path,
                               is_unrefresh_save=False,
                               is_save_fig=cfg["data"]["output"]["is_save_simpfig"],
                               is_Debug=isdebug)

    #########################
    # 4. evaluation
    #########################
    cfg_eval = cfg["eval"]
    if cfg_eval["is_eval"]==True:
        eval_gt_path, is_save_res = get_evalparams(cfg_eval)
        mdl_eval.main_eval(out_simp_folder, res_type=".json",
                           shp_gt_path=eval_gt_path,
                           dataset_type=ds_type,
                           out_folder=out_eval_folder,
                           res_base=os.path.basename(out_simp_folder),
                           bld_list=bld_list,
                           is_save_res=is_save_res)


    return "done."


def parse_args():
    import argparse
    parser = argparse.ArgumentParser('phshape')
    parser.add_argument("--config", type=str, default="../config/trd/config_trd_gu_400.yaml",
                        help="the path of config file.")
    return parser.parse_args()


if __name__ == "__main__":
    ###############
    # load yaml file
    ###############
    args = parse_args()
    cfg_path = args.config
    with open(cfg_path, "r+") as cfg_f:
        cfg = yaml.load(cfg_f, Loader=yaml.FullLoader)

    main(cfg)