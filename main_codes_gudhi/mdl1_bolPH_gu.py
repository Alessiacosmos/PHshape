"""
@File           : mdl1_bolPH_gu.py
@Author         : Gefei Kong
@Time:          : 05.06.2023 14:41
------------------------------------------------------------------------------------------------------------------------
@Description    : as below
get basic building outlines by PH (gudhi version)
"""

import json
import os
import numpy as np
import pandas as pd

from tqdm import tqdm

from utils.mdl_geo import poly2Geojson
from utils.mdl_io import load_cloud
from modules.get_basic_ol_v2_gu import get_autooptim_bf_radius_GU, get_build_bf
from utils.mdl_procs import pre_downsampling

# # for debug showing ##############
# np.set_printoptions(suppress=True)
# pd.set_option('display.max_columns', None) #show all columns
# pd.set_option('max_colwidth',100) #set the visual length of values to 100. default=50
# pd.set_option('display.width',300) #set the overall display width



def main_basicOL(data_root_folder:str,
                 data_type:str,
                 out_folder:str,
                 bld_list:list or np.ndarray,
                 pre_cloud_num:int=5000,
                 down_sample_num:int=400,
                 bfr_tole:float=5e-1,
                 bfr_otdiff:float=1e-2,
                 is_use_saved_bfr:bool=False,
                 savename_bfr:str="",
                 is_unrefresh_save:bool=False,
                 is_Debug:bool=False):

    num_bld = len(bld_list)

    # savename_bfr = os.path.join(os.path.dirname(out_folder), f"all_bfrs_optim.csv")
    if is_use_saved_bfr:
        all_bfr_optim = pd.read_csv(savename_bfr)
        if is_Debug:
            print("the file saving bfr_optim existed.")
    else:
        all_bfr_optim = []
        if is_Debug:
            print("the file saving bfr_optim DOESN'T existed, and will be calculted...")

    with tqdm(range(num_bld)) as pbar_blds:
        for bldi in pbar_blds:
            pbar_blds.set_description(f"[basicOL] :: bld_{bldi}-{bld_list[bldi]}")

            bldi_name = bld_list[bldi]  # e.g., "area1_37"

            savename = os.path.join(out_folder, f"{bldi_name}.json")
            if (is_unrefresh_save) and (os.path.exists(savename)):
                continue

            # read point cloud data
            bldi_path = os.path.join(data_root_folder, f"{bldi_name}{data_type}")
            bldi_2d = load_cloud(bldi_path)[:,:2]
            if is_Debug:
                print(f"[bldi_name={bldi_name}] :: the shape of input cloud is: ", bldi_2d.shape)

            ##############
            # pre-processing
            # reduce the number of points by voxel_sampling to speed up PH(0d)
            ##############
            if bldi_2d.shape[0] > pre_cloud_num:
                bldi_2d, used_vs = pre_downsampling(bldi_2d, target_num=pre_cloud_num, start_voxel_size=0.5,
                                                    isDebug=is_Debug)
                if is_Debug:
                    print(f"[bldi_name={bldi_name}] :: the shape of down_sampled input cloud is: ", bldi_2d.shape)

            ##############
            # get auto-optimized buffer radius
            ##############
            if (is_use_saved_bfr == False) or (len(all_bfr_optim) == 0):
                bldi_bfr_optim, bldi_bfr_0d, bldi_bfr_1d, bldi_pers_1d \
                    = get_autooptim_bf_radius_GU(bldi_2d, down_sample_num=down_sample_num, is_down=True, isDebug=is_Debug)
                all_bfr_optim.append([bldi_name, bldi_bfr_optim])
            else:
                try:
                    bldi_bfr_optim = all_bfr_optim.loc[all_bfr_optim["bid"]==bldi_name, "bfr_optim"].values[0]
                    if is_Debug:
                        print(f"is using existed bfr_optim={bldi_bfr_optim}")
                except:
                    bldi_bfr_optim, bldi_bfr_0d, bldi_bfr_1d, bldi_pers_1d \
                        = get_autooptim_bf_radius_GU(bldi_2d, down_sample_num=down_sample_num, is_down=True, isDebug=is_Debug)

            ##############
            # get buffer polygon, whose linearRings is regardes as the basic outlines of the building
            ##############
            bldi_bf_optnew, bldi_bf_optim = get_build_bf(bldi_2d, bfr_optim=bldi_bfr_optim, bf_tole=bfr_tole, bf_otdiff=bfr_otdiff, isDebug=is_Debug)


            ##############
            # save result as Geojson
            ##############
            bldi_olpoly_json = poly2Geojson(bldi_bf_optnew, round_precision=6)
            with open(savename, "w+") as sf:
                json.dump(bldi_olpoly_json, sf, indent=4)

            # # ##############
            # # # plot
            # # ##############
            # from utils_gu.mdl_visual import drawmultipolygon
            # # draw new buffer result based on 0d-PH and 1d-PH
            # drawmultipolygon(bldi_bf_optnew, bldi_2d, title=f"{bldi_name}: new buffer (r={bldi_bfr_optim:.4f})")

    ##############
    # save bfr_optim
    ##############
    if is_use_saved_bfr==False and len(all_bfr_optim)!=0:
        all_bfr_optim_bf = pd.DataFrame(all_bfr_optim, columns=["bid", "bfr_optim"])
        all_bfr_optim_bf.to_csv(savename_bfr, index=False)
