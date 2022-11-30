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
    X = x_k[0, :]
    Xdot = x_k[1, :]
    Y = x_k[2, :]
    Ydot = x_k[3, :]

    # plot
    axs[0].plot(t, X, label=legend_label)
    axs[1].plot(t, Xdot, label=legend_label)
    axs[2].plot(t, Y, label=legend_label)
    axs[3].plot(t, Ydot, label=legend_label)

    # add labels
    for ax in axs:
        ax.set(xlim=[t[0], t[-1]], xlabel='Time [s]')
        ax.grid()

        if legend_label:
            ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")

    axs[0].set(ylabel='X [km]')
    axs[1].set(ylabel='Xdot [km/s]')
    axs[2].set(ylabel='Y [km]')
    axs[3].set(ylabel='Ydot [km/s]')


def measurements(y_k: np.ndarray,
                 t: np.ndarray,
                 axs: List[matplotlib.axes.Axes],
                 legend_label: str = ''):
    """Plot measurements and station IDs on 4 subplots.

    Args:
        y_k: 2d array of measurements at each time step
        t: array of length T, times
        axs: list of 4 matplotlib Axes objects to plot on
        legend_label: optional string to use for legend
    """

    # separate measurments into arrays per station id
    # 3d array: 1st dim is station id, second is time step, third is meas
    # vector
    # 2d array: records if station had measurment at that time step
    y_id = np.full((12, np.size(t), 3), np.nan)
    t_id = np.full((12, np.size(t)), np.nan)
    for t_idx, y in enumerate(y_k):

        # for each station, save values to correct station id and time
        for meas in y:
            station_id = int(meas[3])
            t_id[station_id - 1, t_idx] = station_id
            y_id[station_id - 1, t_idx, :] = meas[0:3]

    # give each station id a color
    prop_cycle = plt.rcParams['axes.prop_cycle']
    tab_colors = prop_cycle.by_key()['color']
    id_colors = tab_colors + ['maroon', 'indigo']

    # plot per station id
    for st_id in range(12):
        axs[0].plot(t,
                    y_id[st_id, :, 0],
                    label=legend_label,
                    color=id_colors[st_id])
        axs[1].plot(t,
                    y_id[st_id, :, 1],
                    label=legend_label,
                    color=id_colors[st_id])
        axs[2].plot(t,
                    y_id[st_id, :, 2],
                    label=legend_label,
                    color=id_colors[st_id])
        axs[3].plot(t,
                    t_id[st_id, :],
                    label=legend_label,
                    color=id_colors[st_id])

    # add labels
    for ax in axs:
        ax.set(xlim=[t[0], t[-1]], xlabel='Time [s]')
        ax.grid()

    if legend_label:
        axs[0].legend(bbox_to_anchor=(1.04, 0.5), loc="center left")

    axs[0].set(ylabel='rho [km]')
    axs[1].set(ylabel='rho_dot [km/s]')
    axs[2].set(ylabel='phi [rad]')
    axs[3].set(ylabel='Visible Station ID')


# TODO:
# measurement plotting - use markers?
