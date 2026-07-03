import numpy as np
from triqs.gf import *
from scripts.obs import *
from triqs_tprf.lattice_utils import k_space_path
from scipy.interpolate import griddata
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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


def plot_chimin(data):

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])

    ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]

    ax[0].set_xlabel('$q_z/\pi$')
    ax[0].set_ylabel(r'$\chi^{-1}_m(\pi,\pi,q_z)$')

    ax[1].set_xlabel('$T$')
    ax[1].set_ylabel('$Q_z/\pi$')

    axins = inset_axes(
        ax[0],
        width="30%",
        height="40%",
        bbox_transform=ax[0].transAxes,
        #bbox_to_anchor=(0.1, 0.08, 1, 1),
        bbox_to_anchor=(0.63, 0.08, 1, 1),
        loc="lower left"
    )

    axins.set_facecolor((0.98, 0.98, 0.98, 0.7))

    q_grid = np.linspace(data['q_path'][0], data['q_path'][-1], len(data['invchi'][0]))
    qz_grid = q_grid[:,-1]

    Qz = qz_grid[np.argmin(data['invchi'], axis=1)]
    Qz_ref = data['Q'][:,-1]
    Qz_fit = data['Q_fitted'][:,-1]

    for i, T in enumerate(data['T']):

        plot_every = 4
        if i % plot_every == 0:

            ax[0].plot(
                qz_grid,
                data['invchi'][i],
                'o-',
                markersize=2,
                label=f'T={T:.4g}',
                color=color_list[i//plot_every],
                zorder=-i
            )

            qz_grid_fitted = np.linspace(
                Qz_fit[i]-data['xi_range'][-1],
                Qz_fit[i]+data['xi_range'][-1],
                data['xi_pts']
            )

            OZ_curve = [1/OZ(qz, *data['OZ_fit'][i]) for qz in qz_grid_fitted]

            ax[0].plot(
                qz_grid_fitted,
                OZ_curve,
                '--',
                color='black',
                linewidth=1,
                zorder=-i
            )

            ax[0].plot(
                Qz_fit[i],
                1/OZ(Qz_fit[i], *data['OZ_fit'][i]),
                'x',
                markersize=4,
                color='black',
                zorder=-i
            )

            ax[0].plot(
                Qz_ref[i],
                data['invchi_min'][i],
                '*',
                markersize=2,
                color='red',
                zorder=-i
            )

    qz_grid_fitted = np.linspace(
        Qz_fit[0] - data['xi_range'][-1],
        Qz_fit[0] + data['xi_range'][-1],
        data['xi_pts']
    )

    mask = np.abs(qz_grid - Qz_ref[0]) < 4e-2
    OZ_curve = [1/OZ(qz, *data['OZ_fit'][0]) for qz in qz_grid_fitted]

    axins.plot(
        qz_grid[mask],
        data['invchi'][0][mask],
        'o-',
        markersize=5,
        color=color_list[0],
        zorder=0
    )

    axins.plot(
        qz_grid_fitted,
        OZ_curve,
        '--',
        color='black',
        linewidth=1.2,
        zorder=1
    )

    axins.scatter(
        Qz_fit[0],
        1/OZ(Qz_fit[0], *data['OZ_fit'][0]),
        marker='x',
        s=30,
        linewidths=1.8,
        color='black',
        label='from OZ fit',
        zorder=2
    )

    axins.scatter(
        Qz_ref[0],
        data['invchi_min'][0],
        marker='*',
        s=20,
        linewidths=1.8,
        color='red',
        label='exact',
        zorder=3
    )

    axins.grid(True)
    axins.tick_params(labelsize=15)
    axins.legend(loc='upper right', fontsize=12)

    idx_max = np.argmax(data['invchi'][0][mask])
    axins.set_ylim(0, (data['invchi'][0][mask][idx_max]+max(np.delete(data['invchi'][0][mask], idx_max)))/2)

    ax[1].plot(
        data['T'],
        Qz,
        'o-',
        markersize=4,
        color='steelblue',
        label='from $q$ grid'
    )

    ax[1].plot(
        data['T'],
        Qz_ref,
        '--',
        color='red',
        label='exact'
    )

    ax[1].plot(
        data['T'],
        Qz_fit,
        '--',
        color='black',
        linewidth=0.8,
        label='from OZ fit'
    )

    for a in ax:
        a.grid()
        
    ax[0].legend(loc='upper left')
    ax[1].legend()

    ax[1].set_xlim(left=0.)
    ax[0].set_xlim(0.5,1)
    axins.set_xlim(max(Qz_ref[0]-3e-2,0.5), min(Qz_ref[0]+3e-2,1))
    ax[0].set_ylim(bottom=0.)

    fig.tight_layout()
    return fig

def plot_scaling(data, x_exp=(1, 1, 1),
                 fit=False, origin=True, right="OZ"):
    
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

def plot_diagram(points, z, z_label='Values', level=None, level_label=None,
                 point_label=('',''), title='', label='', figure=None,
                 shape=(1,1,1), scatter=True, logscale=False, deriv_wrt=None,
                 log_deriv=False, deriv_offset=None):

    row, col, pos = shape
    fig = plt.figure(figsize=(13, 6)) if figure is None else figure
    ax = fig.add_subplot(row, col, pos)

    ax.set_xlabel(point_label[0])
    ax.set_ylabel(point_label[1])
    ax.set_title(label)
    ax.grid()

    x = np.array([p[point_label[0]] for p in points])
    y = np.array([p[point_label[1]] for p in points])
    z = np.array(z)

    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))

    xi = np.unique(x)
    yi = np.unique(y)
    Xi, Yi = np.meshgrid(xi, yi)

    # -------------------------
    # NO DERIVATIVE
    # -------------------------
    if deriv_wrt is None:
        Zi = griddata((x, y), z, (Xi, Yi), method='linear')
        z_scatter = z.copy()

    # -------------------------
    # DERIVATIVE
    # -------------------------
    else:
        Zi = np.full_like(Xi, np.nan, dtype=float)
        z_scatter = np.full(len(z), np.nan)

        if deriv_wrt == point_label[0]:
            for i, yi_val in enumerate(yi):
                mask = (y == yi_val)
                if mask.sum() > 1:
                    x_row = x[mask]
                    z_row = z[mask]

                    sort_idx = np.argsort(x_row)
                    x_sorted = x_row[sort_idx]
                    z_sorted = z_row[sort_idx]

                    dz = np.gradient(z_sorted, x_sorted)

                    if log_deriv:
                        valid = z_sorted != 0
                        dz_log = np.full_like(dz, np.nan)
                        dz_log[valid] = (x_sorted[valid] / z_sorted[valid]) * dz[valid]
                        dz = dz_log

                    for j, xv in enumerate(x_sorted):
                        col_idx = np.where(xi == xv)[0][0]
                        Zi[i, col_idx] = dz[j]

                    inv = np.empty_like(sort_idx)
                    inv[sort_idx] = np.arange(len(sort_idx))
                    z_scatter[mask] = dz[inv]

        elif deriv_wrt == point_label[1]:

            if deriv_offset is None:
                Tc_list = [0.] * len(xi)
            elif np.isscalar(deriv_offset):
                Tc_list = [float(deriv_offset)] * len(xi)
            else:
                Tc_list = list(deriv_offset)
                if len(Tc_list) != len(xi):
                    raise ValueError("deriv_offset length mismatch")

            for i, (xi_val, Tc) in enumerate(zip(xi, Tc_list)):
                mask = (x == xi_val)
                if mask.sum() > 1:
                    y_col = y[mask]
                    z_col = z[mask]

                    sort_idx = np.argsort(y_col)
                    y_sorted = y_col[sort_idx]
                    z_sorted = z_col[sort_idx]

                    dz = np.gradient(z_sorted, y_sorted)

                    if log_deriv:
                        shifted = y_sorted - Tc
                        valid = z_sorted != 0
                        dz_log = np.full_like(dz, np.nan)
                        dz_log[valid] = (shifted[valid] / z_sorted[valid]) * dz[valid]
                        dz = dz_log

                    for j, yv in enumerate(y_sorted):
                        row_idx = np.where(yi == yv)[0][0]
                        Zi[row_idx, i] = dz[j]

                    inv = np.empty_like(sort_idx)
                    inv[sort_idx] = np.arange(len(sort_idx))
                    z_scatter[mask] = dz[inv]

        else:
            raise ValueError("deriv_wrt must match one of point_label")

    # -------------------------
    # MASK + COLOR NORMALIZATION
    # -------------------------
    Zi_masked = np.ma.masked_invalid(Zi)
    vals = Zi_masked.compressed()

    if vals.size == 0:
        norm = mcolors.Normalize(vmin=-1, vmax=1)
    elif logscale:
        pos_vals = vals[vals > 0]
        norm = mcolors.LogNorm(vmin=pos_vals.min(), vmax=pos_vals.max()) if pos_vals.size > 0 else mcolors.Normalize()
    elif np.any(vals < 0) and np.any(vals > 0):
        vmax = np.max(np.abs(vals))
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0., vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=np.min(vals), vmax=np.max(vals))

    # -------------------------
    # CONTOUR PLOT
    # -------------------------
    contour = ax.contourf(Xi, Yi, Zi_masked, levels=20,
                          cmap='RdYlBu_r', alpha=0.6, norm=norm)
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label(z_label)

    ax.contour(Xi, Yi, Zi_masked, levels=20,
               colors='k', linewidths=0.5, alpha=0.6)

    if level is not None:
        line = ax.contour(Xi, Yi, Zi_masked,
                          levels=[level], colors='black',
                          linewidths=2., linestyles=':')
        if level_label:
            ax.clabel(line, inline=1, fontsize=10, fmt=level_label)

    # -------------------------
    # SCATTER
    # -------------------------
    if scatter:
        valid = ~np.isnan(z_scatter)
        ax.scatter(x[valid], y[valid],
                   c=z_scatter[valid], s=30,
                   cmap='RdYlBu_r', edgecolor='black',
                   norm=norm, zorder=5)
        ax.scatter(x[~valid], y[~valid],
                   c='gray', s=30, edgecolor='black',
                   zorder=5, alpha=0.5)

    fig.suptitle(f'Phase diagram - {title}')
    return fig