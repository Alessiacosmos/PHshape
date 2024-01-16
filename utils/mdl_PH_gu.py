"""
@File           : mdl_PH_gu.py
@Author         : Gefei Kong
@Time:          : 05.06.2023 14:44
------------------------------------------------------------------------------------------------------------------------
@Description    : as below
calculate PH by gudhi
"""

import gudhi
import numpy as np
import time
import pandas as pd



def crt_simptree_gu(data:np.ndarray, max_dim:int) -> gudhi.SimplexTree:
    ####################
    # create VR complex
    ####################
    rips_complex = gudhi.RipsComplex(points=data)

    ####################
    # create simplex tree
    ####################
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim)  # 2 let 0- and 1-d PH can be tracked

    return simplex_tree


def calc_PH_0d_gu(data:np.ndarray, isDebug=False) -> (pd.DataFrame, float):
    """
    run 0-d PH by gudhi
    :param data:
    :param isDebug:
    :return:
        pers_0d: dataframe saving the pers pairs info.
                 columns=[birth, death, pers]
        maxr_0d: max pers time in pers_0d
    """
    if isDebug:
        st_time = time.time()
        print(f"[calc_PH_0d_gu()] :: start to compute PH...")

    ####################
    # get simplex tree
    ####################
    simplex_tree = crt_simptree_gu(data, max_dim=1)

    ####################
    # comput persistence homology
    # format: [[dim, (birth, death)], [dim, (birth, death)], ...]
    ####################
    diag = simplex_tree.persistence()


    if isDebug:
        ed_time = time.time()
        print(f"[calc_PH_0d_gu()] :: finish PH computation, time={ed_time-st_time}(s).")

    ####################
    # flat diag
    # and get 0d and 1d pers and needed radius
    ####################
    # 1. flat diag to shape=[n, 3], where n is the number of <birth, death> pairs at 0 and 1 dimension
    diag_flat = np.array([[_[0], _[1][0], _[1][1]] for _ in diag if _[1][1] != np.inf])

    # 2. get 0d pers & the needed radius
    diag_0d = diag_flat[diag_flat[:, 0] == 0]
    pers_0d = pd.DataFrame(diag_0d[:, 1:], columns=["birth", "death"])
    pers_0d["pers"] = pers_0d["death"] - pers_0d["birth"]
    # get max radius
    maxr_0d = pers_0d.loc[:, 'pers'].values.max() * 1/2


    return pers_0d, maxr_0d


def calc_PH_1d_gu(data: np.ndarray, isDebug=False):
    """
    run 1-d PH by gudhi
    :param data:
    :param isDebug:
    :return:
    """
    if isDebug:
        st_time = time.time()
        print(f"[calc_PH_1d_gu()] :: start to compute PH...")

    ####################
    # get simplex tree
    ####################
    simplex_tree = crt_simptree_gu(data, max_dim=2)

    ####################
    # comput persistence homology
    # format: [[dim, (birth, death)], [dim, (birth, death)], ...]
    ####################
    diag = simplex_tree.persistence()

    if isDebug:
        ed_time = time.time()
        print(f"[calc_PH_1d_gu()] :: finish PH computation, time={ed_time - st_time}(s).")

    ####################
    # flat diag
    # and get 0d and 1d pers and needed radius
    ####################
    # 1. flat diag to shape=[n, 3], where n is the number of <birth, death> pairs at 0 and 1 dimension
    diag_flat = np.array([[_[0], _[1][0], _[1][1]] for _ in diag if _[1][1] != np.inf])

    # 3. get 1d pers
    diag_1d = diag_flat[diag_flat[:, 0] == 1]
    pers_1d = pd.DataFrame(diag_1d[:, 1:], columns=["birth", "death"])
    pers_1d["pers"] = pers_1d["death"] - pers_1d["birth"]
    # get max radius
    maxr_1d = pers_1d.loc[:, 'death'].values.max() * 1 / 2

    return pers_1d, maxr_1d


def calc_PH_gu(data: np.ndarray, isDebug=False):
    """
    run 0- and 1-d PH by gudhi
    :param data:
    :param isDebug:
    :return:
    """
    if isDebug:
        st_time = time.time()
        print(f"[calc_PH_gu()] :: start to compute PH...")

    simplex_tree = crt_simptree_gu(data, max_dim=2)


    ####################
    # comput persistence homology
    # format: [[dim, (birth, death)], [dim, (birth, death)], ...]
    ####################
    diag = simplex_tree.persistence()
    # print(simplex_tree.persistence_pairs())
    if isDebug:
        ed_time = time.time()
        print(f"[calc_PH_gu()] :: finish PH computation, time={ed_time - st_time}(s).")

    ####################
    # flat diag
    # and get 0d and 1d pers and needed radius
    ####################
    # 1. flat diag to shape=[n, 3], where n is the number of <birth, death> pairs at 0 and 1 dimension
    diag_flat = np.array([[_[0], _[1][0], _[1][1]] for _ in diag if _[1][1] != np.inf])

    # 2. get 0d pers & the needed radius
    diag_0d = diag_flat[diag_flat[:, 0] == 0]
    pers_0d = pd.DataFrame(diag_0d[:, 1:], columns=["birth", "death"])
    pers_0d["pers"] = pers_0d["death"] - pers_0d["birth"]
    # get max radius
    maxr_0d = pers_0d.loc[:, 'pers'].values.max() * 1 / 2

    # 3. get 1d pers
    diag_1d = diag_flat[diag_flat[:, 0] == 1]
    pers_1d = pd.DataFrame(diag_1d[:, 1:], columns=["birth", "death"])
    pers_1d["pers"] = pers_1d["death"] - pers_1d["birth"]
    # get max radius
    maxr_1d = pers_1d.loc[:, 'death'].values.max() * 1 / 2

    return pers_0d, maxr_0d, pers_1d, maxr_1d


