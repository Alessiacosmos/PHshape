"""
@File           : simp_basic_ol.py
@Author         : Gefei Kong
@Time:          : 02.05.2023 20:44
------------------------------------------------------------------------------------------------------------------------
@Description    : as below
simplify basic building outlines
"""

import numpy as np
import shapely
from shapely.geometry import Polygon

from utils.mdl_FD import get_fd, trunc_fft, recon_by_fdLow
from utils.mdl_geo import get_PolygonCoords_withInter, arr2Geo


def stop_by_IoU(poly_simp: np.ndarray, poly_used_geo:shapely.geometry):
    try:
        if isinstance(poly_simp, np.ndarray):
            poly_used_fdRi_geo = arr2Geo(poly_simp, "poly").run()
        else:
            poly_used_fdRi_geo = poly_simp
        inter_simp_used = poly_used_fdRi_geo.intersection(poly_used_geo)
        union_simp_used = poly_used_fdRi_geo.union(poly_used_geo)
    except:
        inter_simp_used = None
        union_simp_used = None

    if inter_simp_used is not None:
        RO_area_inter = inter_simp_used.area / union_simp_used.area
    else:
        RO_area_inter = 0

    return RO_area_inter


def stop_by_Hausdoff(poly_simp: np.ndarray, poly_used_geo: shapely.geometry):
    try:
        poly_used_fdRi_geo = arr2Geo(poly_simp, "poly").run()
        hausd = poly_used_geo.hausdorff_distance(poly_used_fdRi_geo)
    except:
        hausd = 10

    return hausd

def get_proper_simp_res(poly_used:np.ndarray,
                        poly_used_geo:shapely.geometry,
                        thres_mode:str,
                        thres_simparea:float,
                        thres_haus:float) -> list:
    """
    for a polygon, save its 3 simplified poly whose area / original poly's area >= thres_simparea
    :param poly_used:           the pts of poly_used
    :param poly_used_geo:       the Polygon of poly_used
    :param thres_mode:          the thres mode. ["haus"(hausdorff distance), "iou"]
    :param thres_simparea:      the threshold to determine whether a simplified poly can be saved
    :param thres_haus:          hausdorff threshold
    :return:
        poly_simp_sele_list:    top-3 simplified polys meeting the threshold requirement.
                                Why top-3? ensure the poly meeting the threshold is not a random result
    """
    # get the max_num for simplifying
    poly_vnum = poly_used.shape[0]

    #########################
    # 3.1 preparation of Fourier descriptor
    # normalize data
    #########################
    # if shape_pts[:,0].min != 0: # 该数据没有经过归一化
    poly_min = np.min(poly_used, axis=0)
    poly_used = poly_used - poly_min

    #########################
    # 3.2 get Fourier descriptor of the shape
    #########################
    poly_fd = get_fd(poly_used)

    #########################
    # 3.3 trunc Fd by using different number parameter, get the simplifed shape
    # & judge whether the threshold is proper or not
    #########################
    poly_simp_sele_list = []
    for i in range(3, int(poly_vnum)):
        num_sele_fd = i

        # 3.3.1 trunc Fd to achieve the simplification, top_num decides the vertex number of the simplified shape
        poly_fdLow = trunc_fft(poly_fd, top_num=num_sele_fd)
        # 3.3.2 inverse fft to achieve the rebuilding of the shape
        poly_scale = np.max(poly_used) # (poly_used[:, 0])
        poly_simp = recon_by_fdLow(poly_fdLow, scale=poly_scale)
        # re-normalize
        poly_simp += poly_min

        # 3.3.3 judge whether the iteration stop or not
        if poly_simp.shape[0] >= 3: # 有三个以上的点
            if thres_mode=="iou":
                value_of_thres = stop_by_IoU(poly_simp, poly_used_geo)
                if value_of_thres >= thres_simparea:
                    poly_simp_sele_list.append([num_sele_fd, poly_simp])
            elif thres_mode=="haus":
                value_of_thres = stop_by_Hausdoff(poly_simp, poly_used_geo)
                if value_of_thres <= thres_haus:
                    poly_simp_sele_list.append([num_sele_fd, poly_simp])
            else:
                raise ValueError(f"The expected 'thres_mode' is in ['haus', 'iou'], but {thres_mode} was gotten.")

        # when the simplification result is stable (>=3 simplified polygon's hauses/ious continuely meet the thred), stop and output
        if len(poly_simp_sele_list) > 2:
            break

    return poly_simp_sele_list

def simp_poly_Fd(poly:shapely.geometry,
                 thres_mode:str="haus",
                 thres_simparea:float=0.99,
                 thres_haus:float=0.5,
                 isDebug:bool=False) -> (np.ndarray or list, list, Polygon):
    """
    simplify the polygon by Fourier descriptor
    :param poly:            the Polygon
    :param thres_mode:      the thres mode. ["haus"(hausdorff distance), "iou"]
    :param thres_simparea:  the threshold to determine whether a simplified poly can be saved as the result.
                            The work principle of the threshold is:
                                when the Ratio between
                                (1) intersected poly's area between the poly_simp by FD and its original poly_used_geo and
                                (2) the area of the original poly_used_geo
                                >= the specified thres_simparea, such as 0.99(99%),
                                THIS simplified poly can be saved as a candidate result
    :param thres_haus:      The hausdorff_distance threshold between poly_simp and poly_used_geon, adaptive.
    :param isDebug:
    :return:
        poly_ext_simp, poly_ints_simp: the simplified poly_arr of exterior and interiors
        poly_simp_geo:                 the shapely.geometry.Polygon object of the simplified result
    """
    ####################
    # 1. Get coords arrs of all exterior and interior polys
    ####################
    poly_ext, poly_ints = get_PolygonCoords_withInter(poly)

    ####################
    # 2. Get polygons of all exterior and interior polys: type:LinearRings
    ####################
    poly_ext_geo, poly_ints_geo = poly.exterior, poly.interiors

    ####################
    # 3. For each sub-polys in the basic footprint polygon,
    # calculate their Fourier descriptor with different simplify param.
    ####################
    poly_ext_simp, poly_ints_simp = [], []
    for pi in range(1 + len(poly_ints)):
        # choose the polygon need to be calculated
        if pi == 0:  # when pi=0, calculate input_poly's exterior's poly
            poly_used = poly_ext
            poly_used_geo = Polygon(poly_ext_geo)
        else:  # when pi>0, calculate input_poly's interiors' poly
            poly_used = poly_ints[pi - 1]
            poly_used_geo = Polygon(poly_ints_geo[pi -1])

        # if isDebug:
        #     print(f"[simp_basic_ol/simp_poly_Fd()] :: for poly_used[{pi}], "
        #           f"the poly_used is: {poly_used}")
        #           # f"\n the poly_used_geo is: {poly_used_geo}")

        # list -> np.ndarray
        poly_used = np.asarray(poly_used)
        assert len(
            poly_used.shape) == 2, f"The shape of used polygon which will be simplified by Fourier descriptor " \
                                   f"is {poly_used.shape}," \
                                   f"but a 2d array is expected."


        #########################
        # 3.1-3.3 get Fourier descriptor of each poly_used, and save several proper simplified result (3 polys),
        # the end of iteration depends on the thres_mode
        #########################
        poly_simp_sele_list = get_proper_simp_res(poly_used, poly_used_geo, thres_mode, thres_simparea, thres_haus)

        #########################
        # 3.4 get final simplified result for each poly used: by using the first poly can meed the threshold in 3.1-3.3
        #########################
        if isDebug:
            try:
                print(f"[simp_basic_ol/simp_poly_Fd()] :: for poly_used[{pi}], "
                      f"the No. of simplified poly's vertices = {poly_simp_sele_list[0][0]},"
                      f"(original num = {poly_used.shape[0]})")
            except:
                print(f"[simp_basic_ol/simp_poly_Fd()] :: for poly_used[{pi}], "
                      f"the simplified poly = {poly_simp_sele_list},"
                      f"(original num = {poly_used.shape[0]})")
        if pi==0: # when pi=0, is calculating poly's exterior's poly
            poly_ext_simp = poly_simp_sele_list[0][1]
        else:
            poly_ints_simp.append(poly_simp_sele_list[0][1])

    ####################
    # 4. Get simplified Geo result
    ####################
    if len(poly_ints)!=0:
        poly_simp_geo = Polygon(poly_ext_simp, holes=poly_ints_simp)
    else:
        poly_simp_geo = Polygon(poly_ext_simp)

    ####################
    # 5. Return result
    ####################
    return poly_ext_simp, poly_ints_simp, poly_simp_geo



def simp_poly_Extmtd(poly:shapely.geometry, bfr_otdiff:float, bfr_tole:float) -> \
        (np.ndarray or list, list, shapely.geometry):
    """
    simple polygon by using exsiting methods in shapely library
    :param poly:
    :param bfr_otdiff: diff value between bfr_optim and bfr_tole(rance)
    :param bfr_tole:   the buffer_radiu_tolerance.
    :return:
           poly_simp:  the simplified polygon
    """
    # automatic set a series of simplify radius
    simp_r_base = bfr_otdiff / 2
    simp_r_max = bfr_tole * 1 / 2  # * np.sqrt(1/2) # simp_r_base * 5
    if simp_r_base != 0:
        if simp_r_max > simp_r_base * 2:
            simp_rs = np.flip(np.arange(simp_r_base * 2, simp_r_max + 0.01, simp_r_base))
        else:
            simp_rs = [simp_r_max]
    else:
        simp_rs = np.flip(np.arange(simp_r_base * 2, simp_r_max + 0.01, 0.05))

    # automatic find the proper simplify result by using a series of
    # simplify_radius(large->small, the larger simplify_radius means the more simplified result).
    for simp_r in simp_rs:
        # simp_r = 1e-1 *3 # * 2  # 1e-1 is from the buffer tolerance from get_build_bf() -> get_build_bf()
        poly_simp = poly.simplify(simp_r)
        iou_simp_org = stop_by_IoU(poly_simp, poly)
        if iou_simp_org >= 0.90:
            break

    b_oli_simp_ext, b_oli_simp_ints = get_PolygonCoords_withInter(poly_simp)

    return b_oli_simp_ext, b_oli_simp_ints, poly_simp



def simp_by_choosed_mtd(b_oli:shapely.geometry, method_name:str="haus", **kwargs):
    assert method_name in ["iou", "haus", "extm"], \
        ValueError(f"only method_name in ['iou;, 'haus', 'extm'] is accecpted, but {method_name} was gotten.")

    if method_name=="haus":
        try:
            thres_haus, isDebug = kwargs["thres_haus"], kwargs["isDebug"]
        except Exception as e:
            raise e
        b_oli_simp_ext, b_oli_simp_ints, b_oli_simp = simp_poly_Fd(b_oli, thres_mode="haus", thres_haus=thres_haus,
                                                                   isDebug=isDebug)
    elif method_name=="iou":
        try:
            thres_simparea, isDebug = kwargs["thres_simparea"], kwargs["isDebug"]
        except Exception as e:
            raise e
        b_oli_simp_ext, b_oli_simp_ints, b_oli_simp = simp_poly_Fd(b_oli, thres_mode="iou", thres_simparea=thres_simparea)
    else:
        try:
            bfr_otdiff, bfr_tole = kwargs["bfr_otdiff"], kwargs["bfr_tole"]
        except Exception as e:
            raise e
        b_oli_simp_ext, b_oli_simp_ints, b_oli_simp = simp_poly_Extmtd(b_oli, bfr_otdiff=bfr_otdiff, bfr_tole=bfr_tole)

    return b_oli_simp_ext, b_oli_simp_ints, b_oli_simp




if __name__ == "__main__":
    simp_by_choosed_mtd("haus", thres_haus = 0.5)