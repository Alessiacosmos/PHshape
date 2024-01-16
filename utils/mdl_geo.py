"""
@File           : mdl_geo.py
@Author         : Gefei Kong
@Time:          : 21.04.2023 18:35
------------------------------------------------------------------------------------------------------------------------
@Description    : as below
geographic-related operations
"""

import numpy as np
import pandas as pd

import shapely
from shapely import wkt
from shapely.ops import unary_union
from shapely.geometry import Polygon, LineString, Point, MultiPoint, MultiPolygon, mapping


def get_PolygonCoords(polygon, round_precision=None) -> np.ndarray:
    # DONOT round coordinates ->> over high precision, e.g., 0.0000000001
    if round_precision is None:
        if polygon.geom_type == "Polygon":
            coords = list(polygon.exterior.coords)
        elif polygon.geom_type == "MultiPolygon":
            coords = [list(_.exterior.coords) for _ in polygon.geoms]
            coords = sum(coords, [])
        else:
            raise TypeError(f"the geometry type of input data is {polygon.geom_type}, but Polygon or MultiPolygon expected")
    # round coordinates
    else:
        if polygon.geom_type == "Polygon":
            coords = np.round(polygon.exterior.coords, round_precision)#.tolist()  # list(polygon.exterior.coords)
        elif polygon.geom_type == "MultiPolygon":
            coords = [np.round(_.exterior.coords, round_precision) for _ in polygon.geoms]
            coords = sum(coords, [])
        else:
            raise TypeError(f"the geometry type of input data is {polygon.geom_type}, but Polygon or MultiPolygon expected")
        # remove the same coordinates after round, because some coordinates may have the same rounded res.
        unq_ind = np.unique(coords, axis=0, return_index=True)[1].tolist()
        unq_ind = unq_ind + [coords.shape[0]-1] # because the get_polygoncoords function will save the final coordinates which is the same as the first one.
        coords = coords[sorted(unq_ind)]

    coords = np.asarray(coords) # shape=[n,2]
    return coords


def get_PolygonCoords_withInter(polygon:shapely.geometry) -> (list, list):
    assert polygon.geom_type == "Polygon" or polygon.geom_type == "MultiPolygon", \
        "only 'Polygon' or 'MultiPOlygon' are accepted for this function"

    if polygon.geom_type == "MultiPolygon": # find the largest one.
        max_area = -1
        for poly in polygon.geoms:
            if poly.area > max_area:
                max_area = poly.area
                poly_new = poly
        polygon = poly_new
    poly_exter = list(polygon.exterior.coords)
    if len(polygon.interiors) == 0:
        poly_inter = []
    else:
        poly_inter = [list(_.coords) for _ in polygon.interiors]

    return poly_exter, poly_inter


def poly2Geojson(polygon:shapely.geometry, round_precision:int=-1) -> dict:
    if round_precision>=0:
        polygon = wkt.loads(wkt.dumps(polygon, rounding_precision=round_precision))

    poly_json = mapping(polygon)
    return poly_json

def poly2WKT(polygon:shapely.geometry, round_precision:int=-1) -> str:
    if round_precision>=0:
        poly_wkt = wkt.dumps(polygon, rounding_precision=round_precision)
    else:
        poly_wkt = wkt.dumps(polygon)

    return poly_wkt

def obj2Geo(in_obj:str or dict) -> shapely.geometry:
    if isinstance(in_obj, str): # wkt mode
        geo_obj = wkt.loads(in_obj)
    elif isinstance(in_obj, dict): # geojson mode
        geo_obj = shapely.geometry.shape(in_obj)
    else:
        raise TypeError("only wkt format (str) or geojson format (dict) are supported now.")

    return geo_obj

def create_buffer(geo_obj:shapely.geometry, b_radius:float) -> Polygon:
    bf = geo_obj.buffer(b_radius)
    return bf


class arr2Geo:
    """
    convert np.ndarray to geographic objects
    :param geo_arr: np.ndarray
    :param geo_type: str, ['point', 'line', 'poly']
    """
    def __init__(self, geo_arr:np.ndarray, geo_type:str):
        self.geo_arr = geo_arr
        self.geo_type = str.lower(geo_type)

    def arr2Point(self, pts_arr:np.ndarray) -> Point:
        """
        convert points to shapely object
        :param pts_arr:
        :return:
        """
        assert len(pts_arr.shape) <= 2, f"the points_arr is expected as 1d or 2d array, " \
                                       f"but {len(pts_arr.shape)} was gotten."

        if len(pts_arr.shape)==1 :
            pts_geo = Point(pts_arr)
        elif pts_arr.shape[0]==1:
            pts_geo = Point(pts_arr[0])
        else:
            # gdf = self.arr2MultiPoint(pts_arr).values
            # print(gdf)
            # pts_geo = gdf.loc[0, "geometry"]
            pts_geo = MultiPoint(pts_arr)

        return pts_geo

    def arr2LineString(self, line_arr:np.ndarray) -> LineString:
        """
        convert line array to shapely object
        :param line_arr:
        :return:
        """
        assert len(line_arr.shape)==2, f"the line_arr is expected as 2d array, but {len(line_arr.shape)} was gotten."
        line_geo = LineString(line_arr)

        return line_geo

    def arr2Polygon(self, polygon_arr:np.ndarray) -> Polygon:
        """
        convert points to shapely object
        :param polygon_arr:
        :return:
        """
        assert len(polygon_arr.shape) >= 2, \
            f"the points_arr is expected as at least 2d array, but {len(polygon_arr.shape)} was gotten."
        poly_geo = Polygon(polygon_arr)

        return poly_geo

    def run(self)->shapely.geometry:
        if "point" in self.geo_type:
            geo_res = self.arr2Point(self.geo_arr)
        elif "line" in self.geo_type:
            geo_res = self.arr2LineString(self.geo_arr)
        elif "poly" in self.geo_type:
            geo_res = self.arr2Polygon(self.geo_arr)
        else:
            raise ValueError(f"the expected input should include one of these key words: ['point', 'line', 'poly'], "
                             f"but '{self.geo_type}' was gotten.")

        return geo_res






