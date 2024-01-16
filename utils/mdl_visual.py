"""
@File           : mdl_visual.py
@Author         : Gefei Kong
@Time:          : 29.04.2023 19:29
------------------------------------------------------------------------------------------------------------------------
@Description    : as below

"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from matplotlib.patches import ConnectionPatch
from shapely.geometry import Polygon, LineString, Point, MultiPoint
from tsc.utils.viz import plot_persistence


def drawmultipolygon(polygon:shapely.geometry, pts:np.ndarray=None, title:str="", savepath:str=""):
    fcolor=["r","b","g","c"]
    fig, axs = plt.subplots()
    # axs.set_aspect('equal', 'datalim')
    if polygon.geom_type=="MultiPolygon":
        for gi, geom in enumerate(polygon.geoms):
            xs, ys = geom.exterior.xy
            axs.fill(xs, ys, alpha=0.5, fc=fcolor[gi%len(fcolor)], ec='black') # ec='none'
            # draw interiors
            inters = [list(inter.coords) for inter in geom.interiors]
            for inter in inters:
                inter = np.asarray(inter)
                axs.plot(inter[:, 0], inter[:, 1], c='yellow')  # ec='none'
    else:
        # draw exterior
        xs, ys = polygon.exterior.xy
        axs.fill(xs, ys, alpha=0.5, fc='r', ec='black')  # ec='none'
        # draw interiors
        inters = [list(inter.coords) for inter in polygon.interiors]
        for inter in inters:
            inter = np.asarray(inter)
            axs.plot(inter[:,0], inter[:,1], c='yellow')  # ec='none'

    if pts is not None:
        axs.scatter(pts[:, 0], pts[:, 1], c='C1', edgecolor='black', s=2)

    plt.title(title)

    if savepath=="":
        plt.show()
    else:
        plt.savefig(savepath, dpi=300)
    plt.close()


def show_ifd_shape(org_data:np.ndarray, ifd_topP_coords:np.ndarray,
                   title:str="", savepath:str=""):
    fig = plt.figure(figsize=(10, 10))

    ax1 = fig.add_subplot(111)
    ax1.plot(org_data[:, 0], org_data[:, 1], c="blue", marker="o", mfc="blue")
    for i in range(len(org_data)):
        ax1.text(org_data[i, 0], org_data[i, 1], f"{i - 1}")
    ax1.set_xlabel('x'), ax1.set_ylabel('y')
    # ax1.set_title("org")

    # ax2 = fig.add_subplot(211)
    ax1.plot(ifd_topP_coords[:, 0], ifd_topP_coords[:, 1], c="orange", linewidth=2, marker="o", mfc="red")
    for i in range(len(ifd_topP_coords)):
        ax1.text(ifd_topP_coords[i, 0], ifd_topP_coords[i, 1], f"{i - 1}")
    # ax2.set_xlabel('x'), ax2.set_ylabel('y')
    # ax2.set_title("ifd")

    plt.suptitle(title)
    plt.tight_layout()

    if savepath=="":
        plt.show()
        plt.close()
    else:
        plt.savefig(savepath, dpi=300)
        plt.close()

# def plot_persistence(pers_data,
#                      bounds=None,
#                      title='Persistence Diagram',
#                      save=False, path='./figures/',
#                      name='persistence.png', show=True, dpi=150,
#                      figsize=(6, 6),
#                      color='C0',
#                      fig=None, ax=None, alpha=0.5):
#     """
#     plot persistence diagram
#     codes from https://github.com/gjkoplik/topology-demos/blob/cfbe4e3d796ddc099a334cc58782f2630d00fae5/viz/viz.py#L84
#     :param pers_data (pd df): df of persistence information
#     :param bounds (tuple of ints, default `None`): (min, max) x and y values for figure
#         if `None`, will infer shape by max persistence value
#     :param title (str, default 'Persistence Diagram'): title of resulting figure
#     :param save (bool, default False): whether or not to save the plot
#     :param path (str, default './figures/): path to where to save plot
#     :param name (str, default 'persistence.png): name of png file to save
#     :param show (bool, default True): whether or not to show the plot
#     :param dpi (int > 0, default 150): pixel density to save figure at
#     :param figsize (tuple of positive floats, default (6, 4) ): (horizontal, vertical) dimensions of figure
#     :param color (str, default 'C0'): color of the persistence values in the figure
#     :param fig (matplotlib fig): defaults to None (makes own fig)
#     :param ax (matplotlib ax): defaults to None (makes own ax)
#     :param alpha (float in [0, 1]): transparancy value for points
#     :return fig, ax:
#     """
#
#     if fig is None and ax is None:
#         fig, ax = plt.subplots(figsize=figsize)
#     # build 45 degree line
#
#     if bounds is not None:
#         ax.set_xlim(bounds[0], bounds[1])
#         ax.set_ylim(bounds[0], bounds[1])
#         ax.plot([0, bounds[1]], [0, bounds[1]], c='black')
#     else:
#         max_death = pers_data.death.values.max()
#         min_death = min(0, pers_data.death.values.min())
#         min_birth = min(0, pers_data.birth.values.min())
#         min_val = min(min_birth, min_death)
#         ax.plot([min_val, max_death], [min_val, max_death], c='black')
#
#     if pers_data.shape[0] != 0:
#         ax.scatter(pers_data.birth, pers_data.death,
#                    alpha=alpha, c=color)
#
#     ax.set_xlabel("Birth")
#     ax.set_ylabel("Death")
#
#     ax.set_title(title)
#     if save:
#         plt.savefig(os.path.join(path, name), dpi=dpi, bbox_inches='tight')
#         plt.close()
#     if show:
#         plt.show()
#         plt.close()
#
#     return fig, ax


def plot_PH_withsignal(data: np.ndarray, pers: pd.DataFrame,
                       comp_strategy:str="comp_strategy",
                       bounds: list or tuple = [0, 1],
                       savepath:str="", savename:str=""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5),
                                   gridspec_kw={'width_ratios': [6., 4.]})

    ax1.plot(data[:, 0], data[:, 1], c="black")
    ax1.scatter(data[:, 0], data[:, 1], c="black")
    ax1.set_title("Signal")

    # plot persistence values
    plot_persistence(pers,
                     bounds=bounds,
                     c='black',
                     fig=fig, ax=ax2, alpha=1)

    # line up figures
    ax1.set_ylim(bounds[0], bounds[1])

    # signal point will be at death index
    cons = []
    for i in range(pers.shape[0]):
        # death line
        xyA = data[pers.death_index.values[i], :]
        # corresponding pers value will be it's birth and death
        xyB = [pers.birth.values[i], pers.death.values[i]]
        con = ConnectionPatch(xyA=xyA, xyB=xyB,
                              coordsA="data", coordsB="data",
                              axesA=ax1, axesB=ax2,
                              color="black", ls="dotted", alpha=0.5)
        cons.append(con)

        # birth line
        xyA = data[pers.birth_index.values[i], :]
        # corresponding pers value will be it's birth and death
        xyB = [pers.birth.values[i], pers.death.values[i] - pers.pers[i]]
        con = ConnectionPatch(xyA=xyA, xyB=xyB,
                              coordsA="data", coordsB="data",
                              axesA=ax1, axesB=ax2,
                              color="black", ls="dotted", alpha=0.5)
        cons.append(con)

        # vertical connectors
        xyA = [pers.birth.values[i], pers.death.values[i] - pers.pers[i]]
        # corresponding pers value will be it's birth and death
        xyB = [pers.birth.values[i], pers.death.values[i]]
        con = ConnectionPatch(xyA=xyA, xyB=xyB,
                              coordsA="data", coordsB="data",
                              axesA=ax2, axesB=ax2,
                              color="black", ls="dotted", alpha=0.5)
        cons.append(con)

    [ax2.add_artist(con) for con in cons]

    plt.suptitle(f"{comp_strategy}")

    if savepath=="" or savename=="":
        plt.show()
        plt.close()
    else:
        plt.savefig(os.path.join(savepath, savename), dpi=300, bbox_inches='tight')
        plt.close()