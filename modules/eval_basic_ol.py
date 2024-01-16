"""
@File           : eval_basic_ol.py
@Author         : Gefei Kong
@Time:          : 07.05.2023 17:38
------------------------------------------------------------------------------------------------------------------------
@Description    : as below
evaluation of the prediction result
"""

import numpy as np
import geopandas as gpd
import shapely
from shapely.ops import cascaded_union, unary_union
from shapely.geometry import Polygon
from shapely.measurement import hausdorff_distance

# Id's of buildings that have shared property and the ID they share with
all_multi = (   [10455510, 10523206],
                [10477107, 182459054],
                [10477867, 182460028],
                [10478413, 182793620],
                [10479436, 182793639],
                [10482038, 182459097],
                [10486831, 182742597],
                [10493889, 182799475],
                [10498821, 182459496],
                [10512875, 196102698, 196102701],
                [10527457, 182486531],
                [10529271, 182765341],
                [10539242, 182224995],
                [10551498, 182283177],
                [21022071, 182319457],
                [21058025, 21058033, 21058041, 21058068],
                [21062618, 21062626],
                [182142220, 182724610],
                [182149098, 196089969, 300429102],
                [182149446, 182149454],
                [182214833, 182769754],
                [182217271, 196112278],
                [182222321, 182222313],
                [182247480, 196098062],
                [182248754, 21092207],
                [182249041, 182746991],
                [182277967, 182277959],
                [182278416, 182278408],
                [182278629, 196111328],
                [182280119, 182748501],
                [182281948, 182776181],
                [182283193, 196098259],
                [182283320, 182283312],
                [182283347, 182748544],
                [182283770, 182748552, 182283789],
                [182283800, 182748560],
                [182285463, 182769665],
                [182290955, 182751286],
                [182292613, 182292621],
                [182315788, 196108912],
                [182317535, 300440947],
                [182338575, 182338583],
                [182340529, 182340510],
                [182377724, 196070168],
                [182378488, 196071083, 21090425],
                [182379174, 196079602],
                [182380075, 196120114],
                [182394971, 182756946, 196071857],
                [182433101, 182433063, 182433071, 182433098],
                [182444936, 300288993],
                [182446130, 182760595],
                [182703354, 196111905],
                [182728985, 182728977, 182274720, 182728969],
                [182729027, 182281972],
                [182733636, 182283827],
                [182744999, 182214590],
                [182745006, 182214604],
                [182748609, 182284912],
                [182749877, 182286540],
                [182761540, 182222399],
                [196070001, 182394491, 182756768],
                [300089089, 300089124],
                [300228747, 182225711],
                [300429640, 182311162],
                [300557684, 300557689],
                [182338923, 182338931])


def union_poly_shared_property(poly_eval: gpd.GeoDataFrame, poly_all: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    for poly_ids in all_multi:
        #     print(poly_ids)
        tmp_df = poly_all[poly_all['building_id'].isin(poly_ids)]
        tmp_arr = tmp_df.geometry.to_numpy()
        new_geom = gpd.GeoSeries(unary_union(tmp_arr))# (cascaded_union(tmp_arr))
        poly_eval.loc[poly_eval.name == poly_ids[0], "geometry"] = new_geom[0]

    return poly_eval


def load_shp_FKB(shp_path:str, bld_list:list or np.ndarray) -> gpd.GeoDataFrame:
    poly_all = gpd.read_file(shp_path)
    try:
        poly_all = poly_all[['BYGGNR    ', 'geometry']]
        poly_all = poly_all.rename(columns={"BYGGNR    ": "building_id", 'geometry': 'geometry'})  # rename
    except:
        poly_all = poly_all[['NAME', 'geometry']]
        poly_all = poly_all.rename(columns={"NAME": "building_id", 'geometry': 'geometry'})  # rename
    poly_all['building_id'] = poly_all['building_id'].astype('int64')

    poly_eval = poly_all[poly_all.building_id.isin(bld_list)]
    poly_eval = poly_eval.set_index(['building_id'])
    poly_eval['area'] = poly_eval.apply(lambda row: row['geometry'].area, axis=1)
    poly_eval = poly_eval.sort_index()
    poly_eval['name'] = poly_eval.index
    poly_eval['bid'] = poly_eval.index

    poly_eval = union_poly_shared_property(poly_eval, poly_all)

    return poly_eval


def load_shp_Isprs(shp_path:str, bld_list:list or np.ndarray) -> gpd.GeoDataFrame:
    poly_all = gpd.read_file(shp_path)
    poly_all["name"] = poly_all["bid"]
    poly_all = poly_all.rename(columns={"name": "building_id", 'geometry': 'geometry'})

    poly_eval = poly_all[poly_all.building_id.isin(bld_list)]  # Filter med relevante id
    poly_eval = poly_eval.set_index(['building_id'])
    poly_eval['area'] = poly_eval.apply(lambda row: row['geometry'].area, axis=1)
    poly_eval = poly_eval.sort_index()
    poly_eval['name'] = poly_eval.index
    poly_eval['area_number'] = poly_eval['name'].str.split('_').str[0]

    return poly_eval


def load_shp_Isprs_v2(shp_path:str, bld_list:list or np.ndarray) -> gpd.GeoDataFrame:
    poly_all = gpd.read_file(shp_path)
    poly_all["building_id"] = poly_all["bid"]

    poly_eval = poly_all[poly_all.building_id.isin(bld_list)]  # Filter med relevante id
    poly_eval = poly_eval.set_index(['building_id'])
    poly_eval['area'] = poly_eval.apply(lambda row: row['geometry'].area, axis=1)
    poly_eval = poly_eval.sort_index()
    poly_eval['name'] = poly_eval.index
    poly_eval['area_number'] = poly_eval['name'].str.split('_').str[0]

    return poly_eval


def load_shp_otherres(shp_path:str, bld_list:list or np.ndarray) -> gpd.GeoDataFrame:
    poly_all = gpd.read_file(shp_path)
    poly_all = poly_all[['bid', 'IOU', 'geometry']]
    poly_all["building_id"] = poly_all["bid"] # poly_all.rename(columns={"bid": "building_id", 'geometry': 'geometry'})
    poly_eval = poly_all[poly_all.building_id.isin(bld_list)]  # Filter med relevante id
    poly_eval = poly_eval.set_index(['building_id'])
    poly_eval = poly_eval.sort_index()

    return poly_eval


def make_valid(polygon):
    # simple_value = 0.5
    # if (not polygon.is_valid):
    buffer_size = 0
    while True:
        if (buffer_size > 2):
            return None
        pp2 = polygon.buffer(buffer_size, cap_style=3)
        if (pp2.geom_type == "Polygon"):
            potential_polygon = Polygon(list(pp2.exterior.coords))
            potential_polygon = potential_polygon.buffer(-buffer_size, cap_style=3)
            return potential_polygon
        else:
            buffer_size = buffer_size + 0.05


def intersection_union(pred_poly:Polygon, poly_gt_eval: gpd.GeoDataFrame, bid:float or int or str) -> float or None:
    """
    calculate IoU between gt and pred result.
    Codes is from Jakob.
    :param pred_poly:
    :param poly_gt_eval:
    :param bid:
    :return:
    """
    if(pred_poly is None):
        return None

    if not pred_poly.is_valid:
        pred_poly = make_valid(pred_poly)

    gt_poly = poly_gt_eval.loc[bid].geometry
    polygon_intersection = gt_poly.intersection(pred_poly).area
    polygon_union = gt_poly.union(pred_poly).area
    IOU = polygon_intersection / polygon_union
    return IOU


def hausdorff_dis(pred_poly:Polygon, poly_gt_eval: gpd.GeoDataFrame, bid:float or int or str) -> float or None:
    """
    calculate IoU between gt and pred result.
    Codes is from Jakob.
    :param pred_poly:
    :param poly_gt_eval:
    :param bid:
    :return:
    """
    if(pred_poly is None):
        return None

    if not pred_poly.is_valid:
        pred_poly = make_valid(pred_poly)

    gt_poly = poly_gt_eval.loc[bid].geometry
    hd = gt_poly.hausdorff_distance(pred_poly)
    return hd


def hausdorff_dis_v2(pred_poly:Polygon, poly_gt_eval: gpd.GeoDataFrame, bid:float or int or str) -> float or None:
    """
    calculate IoU between gt and pred result.
    Codes is from Jakob.
    :param pred_poly:
    :param poly_gt_eval:
    :param bid:
    :return:
    """
    if(pred_poly is None):
        return None

    if not pred_poly.is_valid:
        pred_poly = make_valid(pred_poly)

    gt_poly = poly_gt_eval.loc[bid].geometry
    hd = hausdorff_distance(gt_poly, pred_poly , densify=0.1)

    return hd




