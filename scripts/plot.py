import numpy as np
from triqs.gf import *
from scripts.obs import *
from triqs_tprf.lattice_utils import k_space_path
from scipy.interpolate import griddata
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from fractions import Fraction
#mpl.rcParams['figure.dpi']=100

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "figure.titlesize": 20,
})

import seaborn as sns
color_list = sns.color_palette('colorblind') + sns.color_palette("Set2") + sns.color_palette("Set3")


def plot_chimin(data, peak_on_the='right'):

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])

    ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]

    ax[0].set_xlabel('$q_z/\pi$')
    ax[0].set_ylabel(r'$\chi^{-1}_m(\pi,\pi,q_z)$')

    ax[1].set_xlabel('$T$')
    ax[1].set_ylabel(r'$\overline{q}_z/\pi$')

    if peak_on_the == 'right':
        bbox_to_anchor=(0.1, 0.08, 1, 1)
        loc_leg = 'upper right'
    else:
        bbox_to_anchor=(0.63, 0.08, 1, 1)
        loc_leg = 'upper left'

    axins = inset_axes(
        ax[0],
        width="30%",
        height="40%",
        bbox_transform=ax[0].transAxes,
        bbox_to_anchor=bbox_to_anchor,
        loc="lower left"
    )

    axins.set_facecolor((0.95, 0.95, 0.95, 0.7))

    q_grid = np.linspace(data['q_path'][0], data['q_path'][-1], len(data['invchi'][0]))
    qz_grid = q_grid[:,-1]

    Qz = qz_grid[np.argmin(data['invchi'], axis=1)]
    Qz_ref = data['Q'][:,-1]
    Qz_fit = data['Q_fitted'][:,-1]

    for i, T in enumerate(data['T']):

        plot_every = 4
        if i % plot_every == 0:

            ax[0].plot(qz_grid, data['invchi'][i], 'o-', markersize=2, label=f'T={T:.4g}', color=color_list[i//plot_every], zorder=-i)

            qz_grid_fitted = np.linspace(Qz_fit[i]-data['xi_range'][i][-1], Qz_fit[i]+data['xi_range'][i][-1], 20)

            OZ_curve = [1/OZ(qz, *data['OZ_fit'][i]) for qz in qz_grid_fitted]

            ax[0].plot(qz_grid_fitted, OZ_curve, ':', color='black', linewidth=1, zorder=-i)
            ax[0].plot(Qz_fit[i], 1/OZ(Qz_fit[i], *data['OZ_fit'][i]), 'x', markersize=4, color='black', zorder=-i)
            if isinstance(data['bz_fine'], dict):
                ax[0].plot(Qz_ref[i], data['invchi_min'][i], '*', markersize=2, color='indianred', zorder=-i)

            if i == 0:
                label1 = 'from OZ fit'
                label2 = 'L-BFGS-B'
            else:
                label1 = None
                label2 = None

            qz_grid_fitted = np.linspace(Qz_fit[i] - data['xi_range'][i][-1], Qz_fit[i] + data['xi_range'][i][-1], 20)

            axins.plot(qz_grid, data['invchi'][i], 'o-', markersize=5, color=color_list[i//plot_every], zorder=0)

            axins.plot(qz_grid_fitted, OZ_curve, ':', color='black', linewidth=1.2, zorder=1)

            axins.scatter(Qz_fit[i], 1/OZ(Qz_fit[i], *data['OZ_fit'][i]), marker='x', s=30, linewidths=1.8, color='black', label=label1, zorder=2)

            if isinstance(data['bz_fine'], dict):
                axins.scatter(Qz_ref[i], data['invchi_min'][i], marker='*', s=20, linewidths=1.8, color='indianred', label=label2, zorder=3)

    axins.grid(True)
    axins.tick_params(labelsize=15)
    axins.legend(loc='upper right', fontsize=12)

    xlim = max(1e-2, 2*data['xi_range'][0][-1])
    mask = np.abs(qz_grid - Qz_ref[0]) < xlim
    invchi_max = max(data['invchi'][0][mask])
    axins.set_ylim(0, invchi_max*1.1)
    axins.set_xlim(max(Qz_ref[0]-xlim, 0.5), min(Qz_ref[0]+xlim, 1))

    ax[1].plot(data['T'], Qz, 'o-', markersize=4, color='steelblue', label='from $q$ grid')

    if isinstance(data['bz_fine'], dict):
        ax[1].plot(data['T'], Qz_ref, '--', color='indianred', label='L-BFGS-B')

    ax[1].plot(data['T'], Qz_fit, '--', color='black', linewidth=0.8, label='from OZ fit')

    for a in ax:
        a.grid()
        
    ax[0].legend(loc=loc_leg)
    ax[1].legend()

    ax[1].set_xlim(left=0.)
    ax[0].set_xlim(0.5,1)
    ax[0].set_ylim(bottom=0.)

    fig.tight_layout()
    return fig

def plot_scaling(data, x_exp=(1,1,1), fit=False, origin=True, right="OZ"):
    
    T = data['T']
    dim = data['lat']['dim']

    def x_axis_label(exp):
        if exp == 1:
            return 'T'
        elif exp == 0.5:
            return r'$\sqrt{T}$'

        frac = Fraction(exp).limit_denominator(100)

        if frac.denominator == 1:
            return rf'$T^{{{frac.numerator}}}$'

        return rf'$T^{{{frac.numerator}/{frac.denominator}}}$'
    
    fig = plt.figure(figsize=(12, 6))

    outer_gs = gridspec.GridSpec(
        1, 3 if right == "OZ" else 3,
        figure=fig,
        width_ratios=[1, 1, 1.1]
    )

    # --- Left: chi ---
    ax_chi = fig.add_subplot(outer_gs[0])
    ax_chi.set_xlabel(x_axis_label(x_exp[0]))
    ax_chi.set_ylabel(r'$\chi^{-1}_m(\overline{\mathbf{q}},T,n)$')
    ax_chi.set_title('Susceptibility')
    ax_chi.grid()

    fp_chi_all = [None] * len(data['n'])
    for i, n in enumerate(data['n']):
        pos_idx = np.where(data['invchi_min'][i] > 0)
        if fit and len(pos_idx[0]) > 2:
            fp_chi = np.polyfit(T[pos_idx][:15] ** x_exp[0],
                            data['invchi_min'][i][pos_idx][:15], 1)
            fp_chi_all[i] = fp_chi
            x0 = T[pos_idx][0] ** x_exp[0]
            ax_chi.axline((x0, x0 * fp_chi[0] + fp_chi[1]),
                          slope=fp_chi[0], linestyle='--',
                          color='black', linewidth=0.9)

        ax_chi.plot(T ** x_exp[0], data['invchi_min'][i], 'o-',
                     markersize=4, color=color_list[i], label=f'$n = {n:.5g}$')
        
    ax_chi.legend()
    if fit and len(pos_idx[0]) > 2:
        ax_chi.set_xlim(right=1.05*T[pos_idx][-1]**x_exp[0])
    if origin:
        ax_chi.set_xlim(left=0.)
    else:
        if fit and len(pos_idx[0]) > 2:
            ax_chi.set_xlim(left=0.95*T[pos_idx][0]**x_exp[0])

    # --- Right: either OZ OR Q ---
    if right == "OZ":
        right_rows = 1
    elif right == "Q":
        right_rows = dim
    else:
        raise ValueError("right must be 'OZ' or 'Q'")

    right_gs = gridspec.GridSpecFromSubplotSpec(
        right_rows, 1, subplot_spec=outer_gs[2], hspace=0.1
    )
    right_axes = [fig.add_subplot(right_gs[i]) for i in range(right_rows)]

    row = 0

    if right == "OZ":
        ax_OZ = right_axes[row]
        ax_OZ.set_ylabel(r'$\mathcal{A}(T)$')
        ax_OZ.set_xlabel(x_axis_label(x_exp[2]))
        ax_OZ.set_title(r'OZ weight')
        ax_OZ.grid()

        fp_OZ_all = [None] * len(data['n'])
        for i, n in enumerate(data['n']):
            pos_idx = np.where(data['OZ_weight'][i] > 0)
            if fit and len(pos_idx[0]) > 2:
                fp_OZ = np.polyfit(T[pos_idx][:15] ** x_exp[2],
                                data['OZ_weight'][i][pos_idx][:15], 1)
                fp_OZ_all[i] = fp_OZ
                x0 = T[pos_idx][0] ** x_exp[2]
                ax_OZ.axline((x0, x0 * fp_OZ[0] + fp_OZ[1]),
                             slope=fp_OZ[0], linestyle='--',
                             color='black', linewidth=0.9)

            ax_OZ.plot(T ** x_exp[2], data['OZ_weight'][i],
                       'o-', markersize=4, color=color_list[i],
                       label=f'$n = {n}$')

        if fit and len(pos_idx[0]) > 2:
            ax_OZ.set_xlim(right=1.05*T[pos_idx][-1]**x_exp[2])
        if origin:
            ax_OZ.set_xlim(left=0.)
        else:
            if fit and len(pos_idx[0]) > 2:
                ax_OZ.set_xlim(left=0.95*T[pos_idx][0]**x_exp[2])

    elif right == "Q":
        Q_label = [r'$\overline{q}_x/\pi$', r'$\overline{q}_y/\pi$', r'$\overline{q}_z/\pi$']

        for d in range(dim):
            ax = right_axes[row]
            row += 1

            ax.set_ylabel(Q_label[d])
            ax.grid()

            for i, n in enumerate(data['n']):
                ax.plot(T, data['Q'][i, :, d],
                        'o-', markersize=4,
                        color=color_list[i])

        right_axes[0].set_title(r'$\overline{\mathbf{q}}$ vector')
        right_axes[-1].set_xlabel('T')

    for ax in right_axes[:-1]:
        ax.tick_params(axis='x', labelbottom=False)
    
    # --- Middle: xi ---
    ax_xi = fig.add_subplot(outer_gs[1])
    ax_xi.set_xlabel(x_axis_label(x_exp[1]))
    ax_xi.set_ylabel(r'$\xi^{-1}_m(T,n)$')
    ax_xi.set_title('Correlation length')
    ax_xi.grid()

    for i, n in enumerate(data['n']):
        pos_idx = np.where(data['invxi_min'][i] > 0)
        if fit and fp_chi_all[i] is not None and fp_OZ_all[i] is not None:
            fp_chi, fp_OZ = fp_chi_all[i], fp_OZ_all[i]
            x_fit = np.linspace(0., 1.1 * T[pos_idx][-1], 1000)
            chi_times_OZ = (fp_chi[0] * x_fit**x_exp[0] + fp_chi[1]) * (fp_OZ[0]  * x_fit**x_exp[2] + fp_OZ[1])
            xi_fit = np.sqrt(np.abs(chi_times_OZ)) * np.sign(chi_times_OZ)
            ax_xi.plot(x_fit**x_exp[1], xi_fit, linestyle='--',
                       color='black', linewidth=0.9)

        ax_xi.plot(T ** x_exp[1], data['invxi_min'][i], 'o-',
                   markersize=4, color=color_list[i])
    
    if fit and len(pos_idx[0]) > 2:
        ax_xi.set_xlim(right=1.05*T[pos_idx][-1]**x_exp[1])
    if origin:
        ax_xi.set_xlim(left=0.)
    else:
        if fit and len(pos_idx[0]) > 2:
            ax_xi.set_xlim(left=0.95*T[pos_idx][0]**x_exp[1])
    
    if origin:
        for ax in [ax_chi, ax_xi] + [right_axes[0]] if right == "OZ" else []:
            ax.set_ylim(bottom=0.)

    fig.tight_layout()
    return fig

def plot_diagram(data_list, var_list, var_plotlabel, inset_xrange=None, subplots='commens', c0=0):

    fig = plt.figure(figsize=(12, 6))
    ax = [fig.add_subplot(1,2,1),
          fig.add_subplot(2,2,2),
          fig.add_subplot(2,2,4)]

    for a in ax:
        a.grid()
        a.set_xlim(data_list[0]['n'][0], data_list[0]['n'][-1])
        a.invert_xaxis()

    ax[0].set_ylabel(r'$T_N$')

    if subplots == 'commens':
        ax[1].set_ylabel(r'$Q_z(T_N)/\pi$')
        ax[2].set_ylabel(r'$\mu(T_N)$')
    elif subplots == 'scaling':
        ax[1].set_ylabel(r'$\gamma$')
        ax[2].set_ylabel(r'$\mathcal{A}(T_N)$')
        ax[2].set_ylim(0, 0.2)

    ax[1].tick_params(axis='x', labelbottom=False)
    ax[0].set_xlabel(r'$n$')
    ax[2].set_xlabel(r'$n$')

    # --- inset axes on ax[0] ---
    axins = None
    if inset_xrange is not None:
        x1, x2 = inset_xrange
        axins = inset_axes(
            ax[0],
            width="45%",
            height="45%",
            bbox_to_anchor=(0.13, 0.13, 0.85, 0.85),
            bbox_transform=ax[0].transAxes,
            loc='upper right'
        )
        axins.grid()
        axins.set_xlim(max(x1, x2), min(x1, x2))   # inverted like the main axes
        axins.set_facecolor((0.95, 0.95, 0.95, 0.7))

    for i, data in enumerate(data_list):
        n_list = data['n']

        Tc = -abs(data['c']/data['a'])**(1/data['b']) * np.sign(data['c']/data['a'])

        if subplots == 'commens':
            y1 = data['Qc'][:, -1]
            y2 = data['mu_c']
        elif subplots == 'scaling':
            y1 = data['b']
            y2 = -abs(data['cOZ']/data['aOZ'])**(1/data['bOZ']) * np.sign(data['cOZ']/data['aOZ'])

        label = rf"${var_plotlabel}={var_list[i]}$"

        color = color_list[c0+i]

        ax[0].plot(n_list, Tc, 'o-', label=label, markersize=4, color=color)
        ax[1].plot(n_list, y1, 'o-', markersize=4, color=color)
        ax[2].plot(n_list, y2, 'o-', markersize=4, color=color)

        if axins is not None:
            axins.plot(n_list, Tc, 'o-', markersize=4, color=color)

        nc = np.interp(0., Tc, n_list)
        for a in ax:
            a.axvline(nc, linestyle='--', color=color, alpha=0.6)

        if axins is not None:
            axins.axvline(nc, linestyle='--', color=color, alpha=0.6)

    ax[0].legend(loc='lower left')
    ax[0].set_ylim(0, 0.4)
    if axins is not None:
        xlo, xhi = sorted(inset_xrange)

        ymin, ymax = np.inf, -np.inf

        for data in data_list:
            n = data['n']
            Tc = -abs(data['c']/data['a'])**(1/data['b']) * np.sign(data['c']/data['a'])

            mask = (n >= xlo) & (n <= xhi)
            idx = np.where(mask)[0]

            if idx.size:
                i0 = max(idx[0] - 1, 0)
                i1 = min(idx[-1] + 1, len(n) - 1)

                ymin = min(ymin, Tc[i0:i1+1].min())
                ymax = max(ymax, Tc[i0:i1+1].max())

        axins.set_ylim(max(ymin,0), ymax)

    fig.tight_layout()

    return fig