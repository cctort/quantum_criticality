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
import matplotlib.patches as patches
from matplotlib import ticker
from matplotlib.animation import FuncAnimation

import seaborn as sns
color_list = sns.color_palette('colorblind') + sns.color_palette("Set2") + sns.color_palette("Set3")

def update_mpl_params(bigger_labels=0):
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 16+bigger_labels,
        "axes.labelsize": 18+bigger_labels,
        "axes.titlesize": 18+bigger_labels,
        "xtick.labelsize": 16+bigger_labels,
        "ytick.labelsize": 16+bigger_labels,
        "legend.fontsize": 12,
        "figure.titlesize": 20+3*bigger_labels//2,
    })

def fmt_sci(val):
    """Formats floats into clean LaTeX math notation without surrounding $ signs."""
    if val == 0:
        return "0"

    # Split e-notation and strip trailing zeros/decimals
    mantissa, exp = f"{float(val):.6e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    exp_int = int(exp)

    if exp_int == 0:
        return mantissa
    if mantissa == "1":
        return f"10^{{{exp_int}}}"

    return rf"{mantissa} \times 10^{{{exp_int}}}"

def plot_chimin(data, peak_on_the='right', bigger_labels=0, figsize=(12, 6), plot_every=4):

    update_mpl_params(bigger_labels)

    fig = plt.figure(figsize=figsize)
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

        if i % plot_every == 0:

            ax[0].plot(qz_grid, data['invchi'][i], 'o-', markersize=2, label=f'T={T:.6g}', color=color_list[i//plot_every], zorder=-i)

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
    axins.legend(loc='upper right', fontsize=10)

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
    fig.set_dpi(300)
    return fig

def plot_scaling(data, x_exp=(1,1,1), fit=False, origin=True, right="OZ", c0=0, bigger_labels=0, figsize=(12, 6)):

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
    
    update_mpl_params(bigger_labels)

    fig = plt.figure(figsize=figsize)

    outer_gs = gridspec.GridSpec(
        1, 3,
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
        pos_idx = np.where(data['invchi_min'][i][:6] > 0)
        if fit and len(pos_idx[0]) > 2:
            fp_chi = np.polyfit(T[pos_idx] ** x_exp[0],
                            data['invchi_min'][i][pos_idx], 1)
            fp_chi_all[i] = fp_chi
            x0 = T[pos_idx][0] ** x_exp[0]
            ax_chi.axline((x0, x0 * fp_chi[0] + fp_chi[1]),
                          slope=fp_chi[0], linestyle='--',
                          color='black', linewidth=0.9, zorder=-1)

        ax_chi.plot(T ** x_exp[0], data['invchi_min'][i], 'o-',
                     markersize=4, color=color_list[i+c0], label=f'$n = {n:.6g}$')
        
    ax_chi.legend()
    if fit and len(pos_idx[0]) > 2:
        ax_chi.set_xlim(right=1.05*T[-1]**x_exp[0])
    if origin:
        ax_chi.set_xlim(left=0.)
    else:
        if fit and len(pos_idx[0]) > 2:
            ax_chi.set_xlim(left=0.95*T[0]**x_exp[0])

    # --- Right: OZ, Q, or mu ---
    if right == "OZ":
        right_rows = 1
    elif right == "Q":
        right_rows = dim
    elif right == "mu":
        right_rows = 1
    else:
        raise ValueError("right must be 'OZ', 'Q', or 'mu'")

    right_gs = gridspec.GridSpecFromSubplotSpec(
        right_rows, 1, subplot_spec=outer_gs[2], hspace=0.1
    )
    right_axes = [fig.add_subplot(right_gs[i]) for i in range(right_rows)]

    row = 0

    fp_OZ_all = [None] * len(data['n'])  # needed later for xi fit, only populated if right == "OZ"

    if right == "OZ":
        ax_OZ = right_axes[row]
        ax_OZ.set_ylabel(r'$\mathcal{A}(T)$')
        ax_OZ.set_xlabel(x_axis_label(x_exp[2]))
        ax_OZ.set_title(r'OZ weight')
        ax_OZ.grid()

        for i, n in enumerate(data['n']):
            pos_idx = np.where(data['OZ_weight'][i][:6] > 0)
            if fit and len(pos_idx[0]) > 2:
                fp_OZ = np.polyfit(T[pos_idx] ** x_exp[2],
                                data['OZ_weight'][i][pos_idx], 1)
                fp_OZ_all[i] = fp_OZ
                x0 = T[pos_idx][0] ** x_exp[2]
                ax_OZ.axline((x0, x0 * fp_OZ[0] + fp_OZ[1]),
                             slope=fp_OZ[0], linestyle='--',
                             color='black', linewidth=0.9)

            ax_OZ.plot(T ** x_exp[2], data['OZ_weight'][i],
                       'o-', markersize=4, color=color_list[i+c0],
                       label=f'$n = {n}$')

        if fit and len(pos_idx[0]) > 2:
            ax_OZ.set_xlim(right=1.05*T[-1]**x_exp[2])
        if origin:
            ax_OZ.set_xlim(left=0.)
        else:
            if fit and len(pos_idx[0]) > 2:
                ax_OZ.set_xlim(left=0.95*T[0]**x_exp[2])

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
                        color=color_list[i+c0])

        right_axes[0].set_title(r'$\overline{\mathbf{q}}$ vector')
        right_axes[-1].set_xlabel('T')

    elif right == "mu":
        ax_mu = right_axes[row]
        ax_mu.set_ylabel(r'$\mu(T,n)$')
        ax_mu.set_xlabel(x_axis_label(x_exp[2]))
        ax_mu.set_title('Chemical potential')
        ax_mu.grid()

        for i, n in enumerate(data['n']):
            idx = np.where(np.isfinite(data['mu'][i]))
            if fit and len(idx[0]) > 2:
                fp_mu = np.polyfit(T[idx][:15] ** x_exp[2],
                                data['mu'][i][idx][:15], 1)
                x0 = T[idx][0] ** x_exp[2]
                ax_mu.axline((x0, x0 * fp_mu[0] + fp_mu[1]),
                             slope=fp_mu[0], linestyle='--',
                             color='black', linewidth=0.9)

            ax_mu.plot(T ** x_exp[2], data['mu'][i],
                       'o-', markersize=4, color=color_list[i+c0],
                       label=f'$n = {n}$')

        if fit and len(idx[0]) > 2:
            ax_mu.set_xlim(right=1.05*T[idx][-1]**x_exp[2])
        if not origin and fit and len(idx[0]) > 2:
            ax_mu.set_xlim(left=0.95*T[idx][0]**x_exp[2])

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
        if fit and right == "OZ" and fp_chi_all[i] is not None and fp_OZ_all[i] is not None:
            fp_chi, fp_OZ = fp_chi_all[i], fp_OZ_all[i]
            x_fit = np.linspace(0., 1.1 * T[pos_idx][-1], 1000)
            chi_times_OZ = (fp_chi[0] * x_fit**x_exp[0] + fp_chi[1]) * (fp_OZ[0]  * x_fit**x_exp[2] + fp_OZ[1])
            xi_fit = np.sqrt(np.abs(chi_times_OZ)) * np.sign(chi_times_OZ)
            ax_xi.plot(x_fit**x_exp[1], xi_fit, linestyle='--',
                       color='black', linewidth=0.9, zorder=-1)

        ax_xi.plot(T ** x_exp[1], data['invxi_min'][i], 'o-',
                   markersize=4, color=color_list[i+c0])
    
    if fit and len(pos_idx[0]) > 2:
        ax_xi.set_xlim(right=1.05*T[pos_idx][-1]**x_exp[1])
    if origin:
        ax_xi.set_xlim(left=0.)
    else:
        if fit and len(pos_idx[0]) > 2:
            ax_xi.set_xlim(left=0.95*T[pos_idx][0]**x_exp[1])
    
    if origin:
        axes_to_zero = [ax_chi, ax_xi]
        if right == "OZ":
            axes_to_zero.append(right_axes[0])
        for ax in axes_to_zero:
            ax.set_ylim(bottom=0.)

    fig.tight_layout()
    fig.set_dpi(300)
    return fig

def plot_diagram(data_list, var_list, var_plotlabel, inset_xrange=None, inset_yrange=None, subplots='commens', c0=0, bigger_labels=0, figsize=(12, 6)):

    update_mpl_params(bigger_labels)
    fig = plt.figure(figsize=figsize)
    ax = [fig.add_subplot(1, 2, 1),
          fig.add_subplot(2, 2, 2),
          fig.add_subplot(2, 2, 4)]

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
            y2 = np.abs(data['aOZ'] * np.maximum(Tc, 0.)**data['bOZ'] + data['cOZ'])

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
    if subplots == 'commens':
        ax[1].set_ylim(top=1.)
    elif subplots == 'scaling':
        ax[1].set_ylim(bottom=0.)
        ax[2].set_ylim(0., 8.)
        
    if axins is not None:
        xlo, xhi = sorted(inset_xrange)

        # Use explicit inset_yrange if passed, otherwise compute dynamic Y-limits
        if inset_yrange is not None:
            inset_ymin, inset_ymax = inset_yrange
        else:
            ymin, ymax = np.inf, -np.inf

            for data in data_list:
                n = data['n']
                Tc = -abs(data['c']/data['a'])**(1/data['b']) * np.sign(data['c']/data['a'])

                mask = (n >= xlo) & (n <= xhi)
                idx = np.where(mask)[0]

                if idx.size:
                    i0 = max(idx[0] - 1, 0)
                    i1 = min(idx[-1] + 1, len(n) - 1)

                    valid_tc = Tc[i0:i1+1][np.isfinite(Tc[i0:i1+1])]
                    if valid_tc.size > 0:
                        ymin = min(ymin, valid_tc.min())
                        ymax = max(ymax, valid_tc.max())

            if np.isfinite(ymin) and np.isfinite(ymax) and ymin != ymax:
                inset_ymin, inset_ymax = max(ymin, 0), ymax
            else:
                inset_ymin, inset_ymax = 0.0, 0.4

        axins.set_ylim(inset_ymin, inset_ymax)

        # --- Static 2D Highlight Box on ax[0] ---
        highlight_box = patches.Rectangle(
            (xlo, inset_ymin),
            xhi - xlo,
            inset_ymax - inset_ymin,
            linewidth=1,
            edgecolor='gray',
            facecolor='gray',
            alpha=0.2,
            linestyle='--',
            zorder=1
        )
        ax[0].add_patch(highlight_box)

    fig.tight_layout()
    fig.set_dpi(300)

    return fig

def smoothstep(t):
    """Maps t in [0, 1] to a smooth S-curve (cubic acceleration/deceleration)."""
    return t * t * (3 - 2 * t)

def create_inset_transition(data_list, var_list, var_plotlabel, 
                             range_start, range_end, 
                             yrange_start=None, yrange_end=None, 
                             frames=30, **kwargs):
    
    # 1. Initialize figure with starting range & high DPI
    fig = plot_diagram(data_list, var_list, var_plotlabel, 
                       inset_xrange=range_start, inset_yrange=yrange_start, **kwargs)
    
    ax0 = fig.axes[0]
    axins = fig.axes[-1]  

    # --- Setup Y-Range Sequence ---
    if yrange_start is None or yrange_end is None:
        full_xlo = min(min(range_start), min(range_end))
        full_xhi = max(max(range_start), max(range_end))
        
        ymin, ymax = np.inf, -np.inf
        for data in data_list:
            n = np.asarray(data['n'])
            Tc = -abs(data['c']/data['a'])**(1/data['b']) * np.sign(data['c']/data['a'])
            
            mask = (n >= full_xlo) & (n <= full_xhi)
            idx = np.where(mask)[0]
            if idx.size > 0:
                i0 = max(idx[0] - 1, 0)
                i1 = min(idx[-1] + 1, len(n) - 1)
                valid_tc = Tc[i0:i1+1][np.isfinite(Tc[i0:i1+1])]
                if valid_tc.size > 0:
                    ymin = min(ymin, valid_tc.min())
                    ymax = max(ymax, valid_tc.max())
        
        default_y = (max(ymin, 0), ymax) if np.isfinite(ymin) and np.isfinite(ymax) else (0.0, 0.4)
        yrange_start = yrange_start or default_y
        yrange_end = yrange_end or default_y

    # --- Non-Linear Interpolation (Smoothstep) ---
    t_linear = np.linspace(0, 1, frames)
    t_eased = smoothstep(t_linear)

    x1_seq = range_start[0] + (range_end[0] - range_start[0]) * t_eased
    x2_seq = range_start[1] + (range_end[1] - range_start[1]) * t_eased
    
    y1_seq = yrange_start[0] + (yrange_end[0] - yrange_start[0]) * t_eased
    y2_seq = yrange_start[1] + (yrange_end[1] - yrange_start[1]) * t_eased

    # --- 2D Highlight Box on ax[0] ---
    if ax0.patches:
        highlight_box = ax0.patches[-1]
    else:
        initial_xmin = min(range_start)
        initial_width = abs(range_start[1] - range_start[0])
        initial_ymin = min(yrange_start)
        initial_height = abs(yrange_start[1] - yrange_start[0])

        highlight_box = patches.Rectangle(
            (initial_xmin, initial_ymin),
            initial_width,
            initial_height,
            linewidth=1, edgecolor='gray', facecolor='gray', alpha=0.2, linestyle='--', zorder=1
        )
        ax0.add_patch(highlight_box)

    # --- Sparse Labeled Major Ticks + Unlabeled Minor Ticks ---
    # Strictly limits text labels to at most 3 per axis
    axins.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
    axins.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))

    # Shows tick marks between labels without adding extra text
    axins.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    axins.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    # --- Animation Frame Update ---
    def update(frame):
        x1, x2 = x1_seq[frame], x2_seq[frame]
        y1, y2 = y1_seq[frame], y2_seq[frame]
        
        # 1. Update limits
        axins.set_xlim(max(x1, x2), min(x1, x2))
        axins.set_ylim(min(y1, y2), max(y1, y2))
        
        # 2. Update highlight box on ax[0]
        curr_xmin = min(x1, x2)
        curr_width = abs(x2 - x1)
        curr_ymin = min(y1, y2)
        curr_height = abs(y2 - y1)
        
        highlight_box.set_x(curr_xmin)
        highlight_box.set_y(curr_ymin)
        highlight_box.set_width(curr_width)
        highlight_box.set_height(curr_height)

        return [axins, highlight_box]

    anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=False)
    return anim