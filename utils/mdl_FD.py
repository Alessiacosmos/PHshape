"""
@File           : mdl_FD.py
@Author         : Gefei Kong
@Time:          : 02.05.2023 14:19
------------------------------------------------------------------------------------------------------------------------
@Description    : as below
functions related to Fourier descriptor
reference code: 大津阈值+特征脸+傅里叶描述子 - 知乎 https://zhuanlan.zhihu.com/p/576219472 (专栏已删除)
"""

import numpy as np


def get_fd(shape_pts:np.ndarray):
    """
    finds and returns the Fourier-Descriptor of the shape point set
    idea description:
        1. https://zhuanlan.zhihu.com/p/576219472
        2. https://blog.csdn.net/didi_ya/article/details/114256766
        3. https://blog.csdn.net/Lemon_jay/article/details/89349006
    :param shape_pts: shape=[K, 2], a series of [x,y] coordinates of a 2D-shape
    :return
        fourier_result: shape=(K,), the Fourier descriptor (FD) of the shape
    """
    pts_complex = np.empty(shape_pts.shape[0], dtype=complex)
    pts_complex.real = shape_pts[:,0]
    pts_complex.imag = shape_pts[:,1]
    fourier_result = np.fft.fft(pts_complex) # complex result x+jy
    return fourier_result


def trunc_fft(fd:np.ndarray, top_num:int) -> np.ndarray:
    """
    trunc the top-P Fourier descriptor, to simplify the shape
    codes are from: https://zhuanlan.zhihu.com/p/576219472
    :param fd:       shape=(K, ), complex format (x+jy), the Fourier descriptor
    :param top_num:  the trunc num, top_num = P
    :return:
           fftLow:   the trunc FD at the low-frequency part
                     (because the low-frequency part represents the global shape info.,
                      as explained at https://blog.csdn.net/didi_ya/article/details/114256766:
                      轮廓全局由低频决定，轮廓细节由高频决定)
    """
    fftShift = np.fft.fftshift(fd)  # 中心化，将低频分量移动到频域中心
    center = int(len(fftShift) / 2)
    low, high = center - int(top_num / 2), center + int(top_num / 2)
    fftshiftLow = fftShift[low:high]
    fftLow = np.fft.ifftshift(fftshiftLow)  # 逆中心化
    return fftLow


def norm_fd(f:np.ndarray) -> np.ndarray:
    """
    normalize FD
    reason: https://zhuanlan.zhihu.com/p/576219472
        事物轮廓曲线的初始点、 尺寸及方向会影响傅里叶描述子的大小。
        当识别事物存在尺度改变、 旋转等运动时， 要通过归一化的方法处理描述子，进而使傅里叶描述子具有旋转、平移和尺度变换不变性的特性，
        这就是归一化的傅里叶描述子。
        即： 对傅立叶描述子进行简单的归一化操作后，
            即可使描述子具有平移、旋转、尺度不变性，即不受轮廓在图像中的位置、角度及轮廓的缩放等影响，是一个鲁棒性较好的图像特征。
    :param f: original FD
    :return:
           fd: normalized FD
    """
    f = np.sqrt(np.square(f.real) + np.square(f.imag))
    f[0] = 0
    fd = np.zeros(len(f) - 1)
    fd[1:] = f[2:] / f[1]

    return fd


def recon_by_fdLow(fdLow:np.ndarray, scale:float=1) -> np.ndarray:
    """
    reconstruct shape by low-frequency part FD
    :param fdLow:  shape=(p, ), 1-D fourier descriptor at low-freequency part (complex format (x+jy))
    :param scale:  the max(x-value, y-value) scale, to scale the fdLow to the right [x,y] scale
    :return:
    """
    ifft = np.fft.ifft(fdLow)  # 傅里叶逆变换 (P,)
    contRebuild = np.stack((ifft.real, ifft.imag), axis=-1)  # 复数转为数组 (P, 2)
    if contRebuild.min() < 0:
        contRebuild -= contRebuild.min()
    contRebuild *= scale / contRebuild.max()
    return contRebuild


def simp_shape_fft(shape_pts:np.ndarray, top_num:int):

    # normalize data
    # if shape_pts[:,0].min != 0: # 该数据没有经过归一化
    shape_min = np.min(shape_pts, axis=0)
    shape_pts = shape_pts - shape_min

    # 1. get Fourier descriptor of the shape
    shape_fd = get_fd(shape_pts)
    print("shape_fd max min: ", np.max(shape_fd, axis=0), np.min(shape_fd, axis=0))
    # 2. trunc Fd to achieve the simplification, top_num decides the vertex number of the simplified shape
    shape_fdLow = trunc_fft(shape_fd, top_num)
    print("shape_fdLow max min: ", np.max(shape_fdLow, axis=0), np.min(shape_fdLow, axis=0))
    # 3. inverse fft to achieve the rebuilding of the shape
    scale = np.max(shape_pts)# (shape_pts[:, 0])
    print("shape_pts max min: ", np.max(shape_pts, axis=0), np.min(shape_pts, axis=0), f"scale = {scale}")
    shape_simp = recon_by_fdLow(shape_fdLow, scale=scale)
    print("shape_simp max min: ", np.max(shape_simp, axis=0), np.min(shape_simp, axis=0))
    # re-normalize
    shape_simp += shape_min
    return shape_simp