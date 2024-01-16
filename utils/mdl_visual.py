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

