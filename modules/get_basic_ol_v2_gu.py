"""
@File           : get_basic_ol_v2_gu.py
@Author         : Gefei Kong
@Time:          : 05.06.2023 14:42
------------------------------------------------------------------------------------------------------------------------
@Description    : as below
get basic building outline results by 0d- and 1d- persistent homology (gudhi version)
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from shapely.geometry import Polygon
from sklearn.cluster import DBSCAN

from utils.mdl_geo import arr2Geo
from utils.mdl_procs import down_sample_cloud
from utils.mdl_PH_gu import calc_PH_gu, calc_PH_0d_gu, calc_PH_1d_gu


def optim_bf_radius_gu(pers_1d: pd.DataFrame, bfr_0d: float, isDebug: bool = False) -> (float, pd.DataFrame):
    """
    optimize buffer radius with considering the holes (get r_a).
    3.  Usually, the bfr_1d can be used as the auto-optimized buffer radius.
        However, when here are inner-holes of this building footprint, this bfr_1d will make the holes disappear.
        Hence, we need to further choose a proper bfr in the pers_1d,
        which can save the obvious holes and fill the small gaps caused by the uneven point distribution.
        The method is:
            cluster the 'pers' (meaning pers_time) and 'death' in 'pers_1d' by DBSCAN, clustering_raiuds =  bfr_0d + 1e-2.
            Choose the cluster with the min cluster-center value in both 'pers' and 'death' clusters, as the cluster which can properly fill small holes.
            Ultimately, the max 'death' value in this cluster, will be used as the diameter of the optimized buffer.
            The final optimized bfr (buffer radius) = diameter / 2
    :param pers_1d: the 1d-PH result. [birth, death, pers]
    :param bfr_0d:  the buffer_radius from 0d-PH
    :param isDebug: whether open debug mode and print related info.
    :return:
    """
    # #############
    # analyse pers0d and pers1d, get the proper buffer radius
    # #############
    # 1. get pers and death to use them do DBSCAN
    pers_1d_perl = pers_1d["pers"].values
    pers_1d_death = pers_1d["death"].values
    # 2. DBSCAN, to cluster the pers time based on density
    pers_1d_perl = np.array(pers_1d_perl).reshape(-1, 1)
    cls_1d_perl = DBSCAN(eps=bfr_0d + 1e-2, min_samples=1).fit(pers_1d_perl)
    pers_1d["gr"] = cls_1d_perl.labels_

    pers_1d_death = np.array(pers_1d_death).reshape(-1, 1)
    cls_1d_death = DBSCAN(eps=bfr_0d + 1e-2, min_samples=1).fit(pers_1d_death)
    pers_1d["gr_d"] = cls_1d_death.labels_
    if isDebug:
        print(f"[1-get_basic_ol/optim_bf_radius()] :: DEBSCAN :: "
              f"dbcls_num (pers) = {len(np.unique(cls_1d_perl.labels_))}, "
              f"dbcls_num (death) = {len(np.unique(cls_1d_death.labels_))}")

    ###############
    # get r_a
    ###############
    clsid_bypers = pers_1d.groupby(by=["gr"])["pers"].mean().sort_values().index[0]
    clsid_bydeath = pers_1d.groupby(by=["gr_d"])["death"].mean().sort_values().index[0]
    bfr_optm_d = pers_1d.loc[(pers_1d["gr"] == clsid_bypers) & (pers_1d["gr_d"] == clsid_bydeath),
                             "death"].max()
    bfr_optm = bfr_optm_d / 2
    if isDebug:
        print(f"[1-get_basic_ol/optim_bf_radius()] :: bfr_optm :: final bfr_optm = {bfr_optm}")

    return bfr_optm, pers_1d


def get_autooptim_bf_radius_GU(bldi_2d: np.ndarray, down_sample_num: float = 500,
                               is_down:bool=True,
                               isDebug: bool = False) -> (float, float, float, pd.DataFrame):
    """
    get the optimized buffer radius (using gudhi)
    strategy:
    1.  calculate the 0d-PH, and the max_pers in pers_0d (bfr_0d)
        represents the radius that can make all points connected with their nearest neighbors.
    2.  calculate the 1d-PH, and the max_death in pers_1d (bfr_1d)
        represents the radius that can make all points create a polygon without holes.
    3.  Usually, the bfr_1d can be used as the auto-optimized buffer radius.
        However, when here are inner-holes of this building footprint, this bfr_1d will make the holes disappear.
        Hence, we need to further choose a proper bfr in the pers_1d,
        which can save the obvious holes and fill the small gaps caused by the uneven point distribution.
    :param bldi_2d:         shape=[n,2], 2 means (x,y). The 2d coordinates of a building roof's point clouds
    :param down_sample_num: the target number of the down-sampled point cloud data.
                            It's used to reduce the size of input cloud, to increase the running time of calc_1d_PH.
                            When down_sample_num <= 0, the downsampling won't be implemented.
    :param isDebug:         whehter open debug mode and print related info.
    :return:
           bfr_optim:       float, the auto-optimized buffer radius
           bfr_0d           float, the buffer radius from 0d-PH
           bfr_1d           float, the buffer radius from 1d-PH
           pers_1d          pd.Dataframe, the pers_1d result.
                                          [birth_index, death_index, birth, death, pers, gr]
                                          where 'gr' means the group(class)_label of this pers pair calculated by DBSCAN.
    """

    assert len(bldi_2d.shape) == 2, f"the expected input cloud should be 2d, but {len(bldi_2d.shape)} was gotten."

    if bldi_2d.shape[1] > 2:
        bldi_2d = bldi_2d[:, :2]

    ##############
    # get buffer radius by 0d and 1d PH (using gudhi)
    ##############

    if is_down==False: # not down_sampling for speed-up 1d
        pers_0d, bfr_0d, pers_1d, bfr_1d = calc_PH_gu(bldi_2d, isDebug=True)# , isDebug=isDebug)
    else:
        every_k_point = bldi_2d.shape[0] // down_sample_num  # floor(float)->int
        if (down_sample_num > 0) and (every_k_point > 0):
            bldi_2d_down = down_sample_cloud(bldi_2d, mode="uniform", value=every_k_point)
            if isDebug:
                print(f"[1-get_basic_ol/get_optim_bf_radius()] :: downsampling :: every_k_point={every_k_point}, "
                      f"downsample_pcd_shape={bldi_2d_down.shape}")

            pers_0d, bfr_0d = calc_PH_0d_gu(bldi_2d)
            pers_1d, bfr_1d = calc_PH_1d_gu(bldi_2d_down, isDebug=True)
        else:
            pers_0d, bfr_0d, pers_1d, bfr_1d = calc_PH_gu(bldi_2d, isDebug=True)  # , isDebug=isDebug)


    if isDebug:
        print(f"[1-get_basic_ol/get_optim_bf_radius()] :: calc_PH_0d :: bfr_0d={bfr_0d}, pers_0d=\n{pers_0d}")
        print(f"[1-get_basic_ol/get_optim_bf_radius()] :: calc_PH_1d :: bfr_1d={bfr_1d}, pers_1d=\n{pers_1d}")


    # #############
    # analyse pers0d and pers1d, get the proper buffer radius
    # #############
    bfr_optim, pers_1d = optim_bf_radius_gu(pers_1d=pers_1d, bfr_0d=bfr_0d, isDebug=isDebug)

    return bfr_optim, bfr_0d, bfr_1d, pers_1d


def get_build_bf(bld_2d: np.ndarray, bfr_optim: float, bf_tole: float = 5e-1, bf_otdiff: float = 1e-2,
                 isDebug: bool = False) -> (Polygon, Polygon):
    # convert point array -> geo bld_2d
    bld_geo = arr2Geo(bld_2d, "point").run()  # geo multipoint

    # create buffer
    bld_bf_optim = bld_geo.buffer(distance=bfr_optim + bf_tole)
    if isDebug:
        print(f"[1-get_basic_ol/get_build_bf()] :: bld_bf_optim :: bld_bf_optim is {bld_bf_optim.geom_type}")

    # if multipolygon, choose the polygon with the largest area
    if bld_bf_optim.geom_type == "MultiPolygon":
        bld_bf_optim_areas = [_.area for _ in bld_bf_optim.geoms]
        bld_bf_optim = bld_bf_optim.geoms[bld_bf_optim_areas.index(max(bld_bf_optim_areas))]
        if isDebug:
            print(f"[1-get_basic_ol/get_build_bf()] :: bld_bf_optim MultiPoly -> Poly :: "
                  f"the final area of the bld_bf_optim Poly is {bld_bf_optim.area}.")
    if isDebug:
        print(f"[1-get_basic_ol/get_build_bf()] :: bld_bf_optim has {len(bld_bf_optim.interiors)} interiors,"
              f"and their area is: {[Polygon(inter).area for inter in bld_bf_optim.interiors]}")

    # remove interiors whose area is very small
    inters_save = []
    for inter in bld_bf_optim.interiors:
        inter_poly = Polygon(inter)
        if inter_poly.area > (bfr_optim * 2) ** 2:
            inters_save.append(inter)

    bld_bf_optnew = Polygon(bld_bf_optim.exterior.coords, holes=inters_save)
    if isDebug:
        print(f"[1-get_basic_ol/get_build_bf()] :: {len(inters_save)} interiors are saved."
              f"the area of the final polygon is {bld_bf_optnew.area}")

    
    bld_bf_optnew = bld_bf_optnew.buffer(distance=-(bfr_optim + (bf_tole - bf_otdiff)))  # , quadsegs=1)

    new_bfr = 0.1
    while bld_bf_optnew.geom_type == "MultiPolygon" and new_bfr < bf_tole:
        bld_bf_optnew = bld_bf_optnew.buffer(distance=new_bfr)
        new_bfr += 0.1

    if bld_bf_optnew.geom_type == "MultiPolygon":
        bld_bf_optnew_areas = [_.area for _ in bld_bf_optnew.geoms]
        bld_bf_optnew = bld_bf_optnew.geoms[bld_bf_optnew_areas.index(max(bld_bf_optnew_areas))]

    return bld_bf_optnew, bld_bf_optim

