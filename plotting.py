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


def plot_nees_test(ax: matplotlib.axes.Axes, nees: np.ndarray,
                   time: np.ndarray, r1: float, r2: float):
    """Plot NEES test results.

    Args:
        axs: matplotlib Axes object to plot on
        time: array of length T, times
        nees: array of normalized squared state error values averaged over all
            the simulations
        r1: upper error bound according to alpha
        r2: lower error bound according to alpha
    """

    ax.scatter(time, nees)
    ax.plot(time, r1, 'k--', label='r_1 bound')
    ax.plot(time, r2, 'k--', label='r_2 bound')
    ax.set(xlabel='Time [s]',
           ylabel='NEES Statistic',
           title='NEES Estimation Results')
    ax.set(xlim=[time[0], time[-1]])
    ax.legend()


def plot_nis_test(ax: matplotlib.axes.Axes, nis: np.ndarray, time: np.ndarray,
                  r1: float, r2: float):
    """Plot NIS test results.

    Args:
        axs: matplotlib Axes object to plot on
        time: array of length T, times
        nis: array of normalized squared measurement residual values averaged
            over all the simulations
        r1: upper error bound according to alpha
        r2: lower error bound according to alpha
    """

    ax.scatter(time, nis)
    ax.plot(time, r1, 'k--', label='r_1 bound')
    ax.plot(time, r2, 'k--', label='r_2 bound')
    ax.set(xlabel='Time [s]',
           ylabel='NIS Statistic',
           title='NIS Estimation Results')
    ax.set(xlim=[time[0], time[-1]])
    ax.legend()


def plot_2sig_err(axs: List[matplotlib.axes.Axes],
                  err_k: np.ndarray,
                  time: np.ndarray,
                  err_cov_k: np.ndarray,
                  legend_label: str = ''):
    """Plot state error and 2sigma bounds.

    Args:
        axs: list of 4 matplotlib Axes object to plot on
        err_k: Tx4 array of state errors at each time step
        time: array of length T, times
        err_cov_k: Tx4x4 array of state error covariance matrices
        legend_label: optional string to use for legend
    """

    # get 2 sigma errors
    n = np.size(err_k[0])
    sig_k = np.zeros_like(err_k)
    for t_idk, err_cov in enumerate(err_cov_k):
        for s_idx in range(n):
            sig_k[t_idk, s_idx] = (err_cov[s_idx, s_idx]**0.5) * 2

    # plot
    state_labels = ['dX [km]', 'dXdot [km/s]', 'dY [km]', 'dYdot[km]']
    state_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    for idx, ax in enumerate(axs):
        ax.plot(time,
                err_k[:, idx],
                color=state_colors[idx],
                label='Error' + legend_label)
        ax.plot(time,
                sig_k[:, idx],
                '--',
                color=state_colors[idx],
                label='2 Sigma' + legend_label)
        ax.plot(time,
                -sig_k[:, idx],
                '--',
                color=state_colors[idx],
                label='_nolegend_')

        ax.set(xlim=[time[0], time[-1]],
               xlabel='Time [s]',
               ylabel=state_labels[idx])
        ylims = np.mean(sig_k[30:,idx])
        # ax.set(ylim=[-1.2*ylims,1.2*ylims])
        ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
        ax.grid()
