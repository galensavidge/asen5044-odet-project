"""Plotting functions for nonlinear and linear systems."""

import matplotlib.pyplot as plt
import matplotlib.axes
import numpy as np
from typing import List


def states(x_k: np.ndarray,
           t: np.ndarray,
           axs: List[matplotlib.axes.Axes],
           legend_label: str = ''):
    """Plot states on 4 subplots.

    Args:
        x_k: 4xT array of states at each time step, [X,Xdot,Y,Ydot]
        t: array of length T, times
        axs: list of 4 matplotlib Axes objects to plot on
        legend_label: optional string to use for legend
    """

    # separate states
    X = x_k[:, 0]
    Xdot = x_k[:, 1]
    Y = x_k[:, 2]
    Ydot = x_k[:, 3]

    # plot
    axs[0].plot(t, X, label=legend_label)
    axs[1].plot(t, Xdot, label=legend_label)
    axs[2].plot(t, Y, label=legend_label)
    axs[3].plot(t, Ydot, label=legend_label)

    # add labels
    for ii, ax in enumerate(axs):
        if ii == len(axs) - 1:
            ax.set(xlim=[t[0], t[-1]], xlabel='Time [s]')
        else:
            ax.set(xlim=[t[0], t[-1]])

        ax.grid(visible=True)

        if legend_label:
            ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")

    axs[0].set(ylabel='X [km]')
    axs[1].set(ylabel='Xdot [km/s]')
    axs[2].set(ylabel='Y [km]')
    axs[3].set(ylabel='Ydot [km/s]')


# def measurements(y_k: List,
#                  t: np.ndarray,
#                  axs: List[matplotlib.axes.Axes],
#                  legend_label: str = '',line_style = '-'):
#     """Plot measurements on 3 subplots.

#     Args:
#         y_k: 2d array of measurements at each time step
#         t: array of length T, times
#         axs: list of matplotlib Axes objects to plot on
#         legend_label: optional string to use for legend
#         line_style: optional string to use for plot line style
#     """

#     # seperate measurements into arrays for each element
#     # 2d arrays: first dim is time, second is to store multiple measurement
#     values
#     rho = np.full((np.size(t),1),np.nan)
#     rho_dot = np.full((np.size(t),1),np.nan)
#     phi = np.full((np.size(t),1),np.nan)
#     max_num_meas = 1
#     for t_idx,y in enumerate(y_k):

#         # check how many measurements are in this time step
#         if len(y) != 0 and len(y) % 3 != 0:
#             raise ValueError('Measurement vector does not have multiples of
#             three elements.')
#         num_meas = int(len(y) / 3)

#         # if this is the most so far, adjust the meas arrays sizes
#         if num_meas > max_num_meas:
#             max_num_meas = num_meas
#             rho = np.hstack((rho,np.full((np.size(t),1),np.nan)))
#             rho_dot = np.hstack((rho_dot,np.full((np.size(t),1),np.nan)))
#             phi = np.hstack((phi,np.full((np.size(t),1),np.nan)))

#         # add elements to meas arrays
#         for idx in range(num_meas):
#             print(f'{num_meas=},{idx=},{y=}')
#             rho[t_idx,idx] = y[idx*3]
#             rho_dot[t_idx,idx] = y[idx*3+1]
#             phi[t_idx,idx] = y[idx*3+2]

#     # plot values
#     for idx in range(max_num_meas):
#         axs[0].scatter(t,rho[:,idx],label=legend_label)
#         axs[1].scatter(t,rho_dot[:,idx],label=legend_label)
#         axs[2].scatter(t,phi[:,idx],label=legend_label)

#     # add labels
#     for ax in axs:
#         ax.set(xlim=[t[0], t[-1]], xlabel='Time [s]')
#         ax.grid(visible=True)

#     if legend_label:
#         axs[0].legend(bbox_to_anchor=(1.04, 0.5), loc="center left")

#     axs[0].set(ylabel='rho [km]')
#     axs[1].set(ylabel='rho_dot [km/s]')
#     axs[2].set(ylabel='phi [rad]')


def measurements_withids(y_k: List,
                         t: np.ndarray,
                         axs: List[matplotlib.axes.Axes],
                         legend_label: str = '',
                         marker_style='.',
                         color=None):
    """Plot measurements and station IDs on 4 subplots.

    Args:
        y_k: 3d array of measurements at each time step
        t: array of length T, times
        axs: list of 4 matplotlib Axes objects to plot on
        legend_label: optional string to use for legend
        marker_style: optional string to use for plot marker style
        color: optional string for the color of ALL the points
    """

    # separate measurments into arrays per station id
    # 3d y array: 1st dim is station id, second is time step, third is meas
    # vector
    # 2d t array: records if station had measurment at that time step
    y_id = np.full((12, np.size(t), 3), np.nan)
    t_id = np.full((12, np.size(t)), np.nan)
    for t_idx, y in enumerate(y_k):

        # for each station, save values to correct station id and time
        for meas in y:
            station_id = int(meas[3])
            t_id[station_id - 1, t_idx] = station_id
            y_id[station_id - 1, t_idx, :] = meas[0:3]

    # give each station id a color
    if color:
        id_colors = [color for i in range(12)]
    else:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        tab_colors = prop_cycle.by_key()['color']
        id_colors = tab_colors + ['maroon', 'indigo']

    # plot per station id
    ms = 10
    ll = legend_label
    for st_id in range(12):
        if st_id > 0:
            ll = ''
        axs[0].scatter(t,
                       y_id[st_id, :, 0],
                       label=ll,
                       color=id_colors[st_id],
                       s=ms,
                       marker=marker_style)
        axs[1].scatter(t,
                       y_id[st_id, :, 1],
                       label=ll,
                       color=id_colors[st_id],
                       s=ms,
                       marker=marker_style)
        axs[2].scatter(t,
                       y_id[st_id, :, 2],
                       label=ll,
                       color=id_colors[st_id],
                       s=ms,
                       marker=marker_style)
        axs[3].scatter(t,
                       t_id[st_id, :],
                       label=ll,
                       color=id_colors[st_id],
                       s=ms,
                       marker=marker_style)

    # add labels
    for ii, ax in enumerate(axs):
        if ii == len(axs) - 1:
            ax.set(xlim=[t[0], t[-1]], xlabel='Time [s]')
        else:
            ax.set(xlim=[t[0], t[-1]])

        ax.grid(visible=True)

    if legend_label:
        axs[0].legend(bbox_to_anchor=(1.04, 0.5), loc="center left")

    axs[0].set(ylabel=r'\rho\; [km]')
    axs[1].set(ylabel='rho_dot [km/s]')
    axs[2].set(ylabel='phi [rad]')
    axs[3].set(ylabel='Visible Station ID')
