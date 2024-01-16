"""
@File           : mdl2_simp_bol.py
@Author         : Gefei Kong
@Time:          : 14.05.2023 20:55
------------------------------------------------------------------------------------------------------------------------
@Description    : as below
module 2: simple basic building outlines
"""

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm


from modules.simp_basic_ol import simp_poly_Fd, stop_by_IoU, simp_poly_Extmtd
from utils.mdl_io import load_cloud, load_json
from utils.mdl_geo import obj2Geo, get_PolygonCoords_withInter, arr2Geo, poly2Geojson
from utils.mdl_visual import drawmultipolygon, show_ifd_shape


def main_simp_ol(data_basic_ol_folder: str,
                 data_basic_ol_type: str,
                 dataset_type: str,
                 out_folder: str,
                 bld_list: list or np.ndarray,
                 bfr_tole: float = 0.5,
                 bfr_otdiff: float = 0.0,
                 simp_method: str = "haus",
                 savename_bfr: str="",
                 is_unrefresh_save: bool = False,
                 is_save_fig: bool = False,
                 is_Debug: bool = False,
                 **kwargs):
    num_bld = len(bld_list)

    # savename_bfr = os.path.join(os.path.dirname(out_folder), f"all_bfrs_optim.csv")
    try:
        all_bfr_optim = pd.read_csv(savename_bfr)  # np.loadtxt(savename_bfr)
    except Exception as e:
        print(e)

    with tqdm(range(num_bld)) as pbar_blds:
        for bldi in pbar_blds:
            pbar_blds.set_description(f"[simpOL] :: bld_{bldi}-{bld_list[bldi]}")

            bldi_name = bld_list[bldi]  # "area1_37" # bld_list[0]

            ##################
            # out dir and save name
            ##################
            savename = os.path.join(out_folder, f"{bldi_name}.json")
            if (is_unrefresh_save) and (os.path.exists(savename)):
                continue

            ##############
            # read basic polygon data
            ##############
            b_oli_path = os.path.join(data_basic_ol_folder, f"{bldi_name}{data_basic_ol_type}")
            b_oli_js = load_json(b_oli_path)
            b_oli = obj2Geo(b_oli_js)
            # print("the b_oli is: ", b_oli)

            ##############
            # simplify by choosed method.
            ##############
            if simp_method == "haus":
                ##############
                # simplify by using Fourier descriptor -- stop by haus_dis
                ##############
                bldi_bfr_optim = all_bfr_optim.loc[all_bfr_optim["bid"] == bldi_name, "bfr_optim"].values[0]
                thres_haus = (1 - np.cos(30 / 180 * np.pi)) * (
                            bldi_bfr_optim + bfr_tole - bfr_otdiff)  # bfr_tole/2 # np.ceil(0.0625 * (bldi_bfr_optim + bfr_tole) / 2 * 100) / 100
                b_oli_simp_ext, b_oli_simp_ints, b_oli_simp = simp_poly_Fd(b_oli,
                                                                           thres_mode="haus", thres_haus=thres_haus,
                                                                           isDebug=is_Debug)
            elif simp_method == "iou":
                ##############
                # simplify by using Fourier descriptor -- stop by iou
                ##############
                try:
                    thres_simparea = kwargs["thres_simparea"]
                except Exception as e:
                    raise e
                b_oli_simp_ext, b_oli_simp_ints, b_oli_simp = simp_poly_Fd(b_oli, thres_mode="iou",
                                                                           thres_simparea=thres_simparea, isDebug=is_Debug)
            else:
                ##############
                # simplify by using existing methods from shapely libray
                ##############
                b_oli_simp_ext, b_oli_simp_ints, b_oli_simp = simp_poly_Extmtd(b_oli,
                                                                               bfr_otdiff=bfr_otdiff, bfr_tole=bfr_tole)

            ##############
            # save result
            ##############
            # save polygon
            b_oli_simp_json = poly2Geojson(b_oli_simp, round_precision=6)
            with open(savename, "w+") as sf:
                json.dump(b_oli_simp_json, sf, indent=4)

            # save figure
            if is_save_fig:
                # create folder
                out_path_fig = os.path.join(os.path.dirname(out_folder),
                                            'figures', os.path.basename(out_folder), f"2_simp_FdR_haus_{bldi_name}_0")
                if not os.path.exists(out_path_fig):
                    os.makedirs(out_path_fig)

                # draw overall
                drawmultipolygon(b_oli_simp, title=f"simp_poly_{bldi_name}",
                                 savepath=os.path.join(out_path_fig, f"0-overall_simp_poly_{bldi_name}"))

                # draw each poly in this basic_outline_poly
                # Get coords arrs of all exterior and interior polys
                b_oli_ext, b_oli_ints = get_PolygonCoords_withInter(b_oli)
                for i in range(1 + len(b_oli_simp_ints)):
                    if i == 0:
                        b_oli_poly = np.asarray(b_oli_ext)
                        b_oli_poly_simp = np.asarray(b_oli_simp_ext)
                    else:
                        b_oli_poly = np.asarray(b_oli_ints[i - 1])
                        b_oli_poly_simp = np.asarray(b_oli_simp_ints[i - 1])
                    show_ifd_shape(b_oli_poly, b_oli_poly_simp,
                                   title=f"FdHaus_simplied_coords (P={len(b_oli_poly_simp)})",
                                   savepath=os.path.join(out_path_fig,
                                                         f"FdHausS{i} (P={len(b_oli_poly_simp)})_newscale"))