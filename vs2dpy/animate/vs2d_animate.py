import pandas as pd
import numpy as np
from matplotlib import animation
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm


def vs2d_animation(fold='', name='vid', n=None, x=None, z=None,
                   h=None, gwl=None, gwt_loc=[], fps=4, hourly=False):
    
    ts = list(h.keys())
    if hourly:
        ts = ts[0:-1:24]    

    fig, ax = plt.subplots(3, 1, figsize=(14, 8), gridspec_kw={
        'height_ratios': [1, 3, 1]})

    p = n.copy()
    p[p < 0] = 0
    e = n.copy()
    e[e >= 0] = 0
    cnt = 0

    if len(gwt_loc) == 0:
        gwt_loc = [gwl.columns[3], gwl.columns[int(len(gwl.columns)/2)],
                   gwl.columns[-1]]

    # subplot 0
    vl = ax[0].axvline(ymin=0, ymax=1, linestyle='--', color='C3', alpha=0.8)

    ax[0].plot(p.index, p, label='Precipitation [m]', linewidth=0.7)
    ax[0].plot(e.index, e, label='Pot Evaporation [m]', linewidth=0.7)
    ax[0].grid()
    ax[0].legend(loc=2, ncol=2)
    ax[0].set_xlim(ts[0], ts[-1])

    # subplot 1
    levels = MaxNLocator(nbins=21).tick_values(-2, 0)
    cmap = plt.colormaps['viridis_r']
    mp = ax[1].pcolormesh(x, np.flip(z), h[ts[0]], cmap=cmap,
                          norm=BoundaryNorm(levels, ncolors=cmap.N, clip=True))
    gl, = ax[1].plot([], [], color='white', label='Groundwater Table')
    sc1 = ax[1].scatter([], [], color='C7', zorder=10)
    sc2 = ax[1].scatter([], [], color='C8', zorder=10)
    sc3 = ax[1].scatter([], [], color='C9', zorder=10)

    ax[1].set_ylabel('z [m]')
    ax[1].set_xlabel('x [m]')
    ax[1].legend(handles=[gl], loc=3)

    cb = plt.colorbar(mp, ax=ax[1], fraction=0.05, pad=0.01)
    cb.set_label(r'$\psi$ [m]')
    cb.ax.invert_yaxis()

    # subplot 2
    sc1_gw = ax[2].scatter([], [], s=10, color='C7', zorder=10)
    sc2_gw = ax[2].scatter([], [], s=10, color='C8', zorder=10)
    sc3_gw = ax[2].scatter([], [], s=10, color='C9', zorder=10)

    vl_gw = ax[2].axvline(ymin=0, ymax=1, linestyle='--',
                          color='C3', alpha=0.8)

    for j, loc in enumerate(gwt_loc):
        ax[2].plot(ts, gwl.loc[ts, loc].values,
                   label=f'GWT at x={loc}m', color=f'C{j+7}', linewidth=0.7)

    ax[2].set_xlim(ts[0], ts[-1])
    ax[2].legend(loc=2, ncol=3)
    ax[2].set_ylabel('h [m]')
    ax[2].grid()

    fig.tight_layout()

    def animate(i):

        vl.set_xdata(ts[i])

        ax[1].set_title(r'Pressure head $\psi$' +
                        f' [m] for a polder at day {ts[i]}', fontsize=16)
        mp.set_array(h[ts[i]].ravel())
        gl.set_data(gwl.columns, gwl.loc[ts[i]])

        for sc, sc_gw, loc in zip([sc1, sc2, sc3], [sc1_gw, sc2_gw, sc3_gw], gwt_loc):
            sc.set_offsets((loc, gwl.loc[ts[i], loc]))
            sc_gw.set_offsets((ts[i], gwl.loc[ts[i], loc]))

        vl_gw.set_xdata(ts[i])

    bar = tqdm(total=len(ts))
    anim = animation.FuncAnimation(fig, animate, frames=len(ts))  # len(ind)
    anim.save(f'{fold}{name}.mp4', fps=fps, dpi=150)
