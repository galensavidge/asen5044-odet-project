"""Plotting functions for nonlinear and linear systems."""

import matplotlib.pyplot as plt
import matplotlib.axes
import numpy as np

def states(x_k: np.ndarray,t: np.ndarray,axs: list[matplotlib.axes.Axes],legend_label:str = ''):
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
    axs[0].plot(t,X,label=legend_label)
    axs[1].plot(t,Xdot,label=legend_label)
    axs[2].plot(t,Y,label=legend_label)
    axs[3].plot(t,Ydot,label=legend_label)

    # add labels
    for ax in axs:
        ax.set(xlim=[t[0],t[-1]],xlabel='Time [s]')

        if legend_label:
            ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")

    axs[0].set(ylabel='X [km]')
    axs[1].set(ylabel='Xdot [km/s]')
    axs[2].set(ylabel='Y [km]')
    axs[3].set(ylabel='Ydot [km/s]')

def measurements(y_k: np.ndarray,t: np.ndarray,axs: list[matplotlib.axes.Axes],legend_label:str = ''):
    """Plot measurements and station IDs on 4 subplots.
    
    Args:
        y_k: 2d array of measurements at each time step
        t: array of length T, times
        axs: list of 4 matplotlib Axes objects to plot on
        legend_label: optional string to use for legend
    """
    
    # separate measurments into arrays per station id
    # give each station id a color
    # plot per station id


# TODO:
# measurement plotting