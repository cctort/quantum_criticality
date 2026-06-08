import numpy as np
from triqs.gf import *
from scripts.obs import *
from triqs_tprf.lattice_utils import k_space_path
from scipy.interpolate import griddata
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
mpl.rcParams['figure.dpi']=100

import seaborn as sns
color_list = sns.color_palette('colorblind') + sns.color_palette("Set2") + sns.color_palette("Set3")


def prun_dmft(dmft, U, title='', label='', figure=None):
   
    G_iw = dmft.G_iw
    S_iw = dmft.S_iw
    W = 4*dmft.lattice.dim*dmft.lattice.t

    # Padé Analytic Continuation (A(w) = -1/pi * Im G(w+i0+))
    g_w = GfReFreq(indices=[0], window=(-U-W, U+W), n_points=1000)
    
    g_w.set_from_pade(G_iw, n_points=1000) 
    A_w = -1.0/np.pi * g_w.imag

    # Frequency mesh (real and imag)
    iw = np.array([float(freq.imag) for freq in S_iw.mesh])
    w = np.array([float(freq) for freq in A_w.mesh])

    if figure is None:
        
        fig = plt.figure(figsize=(12, 5))
        axs = [None]*3

        axs[0] = fig.add_subplot(2, 2, 1)
        axs[0].set_xlim(0, 8)
        axs[0].set_xlabel(r'$i\omega_n$')
        axs[0].set_ylabel(r'Re $\Sigma(i\omega_n)$')
        axs[0].set_title('Self-energy (real part)')

        axs[1] = fig.add_subplot(2, 2, 3)
        axs[1].set_xlim(0, 8)
        axs[1].set_xlabel(r'$i\omega_n$')
        axs[1].set_ylabel(r'Im $\Sigma(i\omega_n)$')
        axs[1].set_title('Self-energy (imaginary part)')

        axs[2] = fig.add_subplot(1, 2, 2)
        axs[2].set_xlabel(r'$\omega$')
        axs[2].set_ylabel(r'$A(\omega)$')
        axs[2].set_title(f'Spectral Density (from Padé continuation)')
        axs[2].set_xlim(-W-U, W+U)

        for i in range(3):
            axs[i].grid()

    else:

        fig, axs = figure

    num_plots = len(axs[0].lines)
    color = color_list[num_plots % len(color_list)]

    axs[0].plot(iw, S_iw.real.data[:,0,0], 'o--', markersize=4, color=color, label=label)
    axs[1].plot(iw, S_iw.imag.data[:,0,0], 'o--', markersize=4, color=color, label=label)
    axs[2].plot(w, A_w.data[:,0,0], lw=2, color=color)
    axs[2].axvline(U/2, linestyle='--', alpha=0.5, label=rf'$\pm U/2$ ({label})', color=color)
    axs[2].axvline(-U/2, linestyle='--', alpha=0.5, color=color)
    axs[2].fill_between(w, 0, A_w.data[:,0,0], alpha=0.2, color=color)
    axs[2].set_ylim(bottom=0)
    for i in range(3):
        axs[i].legend()

    fig.suptitle(rf'Self-energy - {title}')
    fig.tight_layout()

    Z, gamma, tau = get_Z(S_iw.data[:,0,0], dmft.beta, dmft.n_iw)

    print(f'quasi-particle weight : Z = {Z:.3e}')
    print(f'quasi-particle scattering rate : gamma = {gamma:.3e}')
    print(f'quasi-particle lifetime : tau = {tau:.3e}')

    return fig, axs

def prun_chi_path(dmft, k_path, title='', label='', figure=None):

    if figure is None:
        
        fig = plt.figure(figsize=(12, 5))
        axs = [None]*3

        axs[0] = fig.add_subplot(2, 2, 1)
        axs[0].set_ylabel(r'$\chi_0^{-1}(\mathbf{k},0)$')
        axs[0].set_title('non-interacting susceptibility')

        axs[1] = fig.add_subplot(2, 2, 3)
        axs[1].set_ylabel(r'$\chi_{RPA}^{-1}(\mathbf{k},0)$')
        axs[1].set_title('RPA susceptibility')

        axs[2] = fig.add_subplot(1, 2, 2)
        axs[2].set_title('Brillouin Zone Path')
        axs[2].set_xlabel(r'$k_x$')
        axs[2].set_ylabel(r'$k_y$')
        axs[2].set_aspect('equal')

        for i in range(3):
            axs[i].grid()

    else:

        fig, axs = figure

    fig.suptitle(rf'Susceptibility - {title}')
    fig.tight_layout()

    num_plots = len(axs[0].lines)
    color = color_list[num_plots % len(color_list)]

    k_labels = [p[0] for p in k_path]
    k_points = [p[1] for p in k_path]
    path = [(k_points[i], k_points[i+1]) for i in range(len(k_points)-1)]

    k_vecs, k_plot, k_ticks = k_space_path(path, num=32, bz=dmft.H_r.bz)
    k = np.linspace(-np.pi, np.pi, num=100)
    kx, ky = np.meshgrid(k, k)

    eps_k_grid = np.vectorize(lambda kx, ky : dmft.eps_k((kx, ky, 0)).real[0,0])(kx, ky)
    invchi0_iwk_path = np.vectorize(lambda k : 1/dmft.chi0_iwk(0,k).real[0,0,0,0], signature='(n)->()')
    invchi_iwk_path = np.vectorize(lambda k : 1/dmft.chi_rpa_iwk(0,k).real[0,0,0,0], signature='(n)->()')

    for i in range(2):
        axs[i].set_xticks(k_ticks, labels=k_labels)
    
    axs[0].plot(k_plot, invchi0_iwk_path(k_vecs), '--', color=color, label=label)
    axs[1].plot(k_plot, invchi_iwk_path(k_vecs), '--', color=color, label=label)

    if figure is None: # We assume that multiple calls refer to the same path in the same BZ

        path_x = np.array([p[0]*np.pi for p in k_points])
        path_y = np.array([p[1]*np.pi for p in k_points])
        
        mu = dmft.convergence['mu'][-1]
        U = dmft.U

        cont = axs[2].contour(kx, ky, eps_k_grid,levels=50,cmap='RdBu')
        axs[2].contour(kx, ky, eps_k_grid,levels=[mu-U/2],colors='black',linestyles='dotted',linewidths=2)
        cbar = fig.colorbar(cont, ax=axs[2])
        cbar.set_label(r'$\epsilon(\mathbf{k})$')
        axs[2].plot(path_x, path_y, 'o', color='indianred', markersize=5)
        for i in range(len(k_points)-1):
            axs[2].annotate('', xy=(path_x[i+1], path_y[i+1]), xytext=(path_x[i], path_y[i]), arrowprops=dict(arrowstyle='->',linewidth=2,color='black'))

        #axs[2].quiver(path_x[:-1]+path_x[1:], path_y[:-1]+path_y[1:], np.diff(path_x), np.diff(path_y), angles='xy', scale_units='xy', scale=3, color=color)
        for txt, x, y in zip(k_labels, path_x, path_y):
            axs[2].annotate(txt, (x, y), textcoords="offset points", xytext=(5,5), ha='center', fontweight='bold')

    return fig, axs


def prun_chi(chi_iwk, title='', label='', figure=None, shape=(1,1,1)):

    row, col, pos = shape

    if figure is None:
        fig = plt.figure(figsize=(12, 5))
        axs = [None]*row*col

        for i in range(row*col):
            axs[i] = fig.add_subplot(row, col, i+1)
            axs[i].set_xlabel(r'$k_x$')
            axs[i].set_ylabel(r'$k_y$')
            axs[i].grid()
            axs[i].set_aspect('equal')

    else:

        fig, axs = figure

    k = np.linspace(0., 2*np.pi, num=100)
    kx, ky = np.meshgrid(k, k)
    invchi_rpa_grid = np.vectorize(lambda kx, ky : 1/chi_iwk(0, (kx, ky, 0)).real[0,0,0,0])(kx, ky)

    cont = axs[pos-1].contour(kx/np.pi, ky/np.pi, invchi_rpa_grid, levels=50, cmap='RdBu')
    axs[pos-1].contour(kx/np.pi, ky/np.pi, invchi_rpa_grid, levels=[0], colors='black', linestyles='dotted', linewidths=2)
    cbar = fig.colorbar(cont, ax=axs[pos-1])
    cbar.set_label(r'$\chi_{RPA}^{-1}(\mathbf{k})$')
    axs[pos-1].set_title(label)

    fig.suptitle(rf'Susceptibility - {title}')
    if pos == row*col:
        fig.tight_layout()

    return fig, axs


def prun_conv(dmft, U, density, title='', label='', figure=None):
    
    convergence = dmft.convergence

    if figure is None:
        
        fig = plt.figure(figsize=(12, 5))
        axs = [None]*6

        axs[0] = fig.add_subplot(2, 3, 1)
        axs[0].set_xlabel('Iterations')
        axs[0].set_ylabel(r'$\Sigma$ residuals')
        axs[0].set_yscale('log')

        axs[1] = fig.add_subplot(2, 3, 4)
        axs[1].set_xlabel('Iterations')
        axs[1].set_ylabel(r'$\alpha$')

        axs[2] = fig.add_subplot(2, 3, 2)
        axs[2].set_xlabel('Iterations')
        axs[2].set_ylabel(r'$n-n_{goal}$')

        axs[3] = fig.add_subplot(2, 3, 5)
        axs[3].set_xlabel('Iterations')
        axs[3].set_ylabel(r'$\mu/U$')

        axs[4] = fig.add_subplot(2, 3, 3)
        axs[4].set_xlabel('Iterations')
        axs[4].set_ylabel(r'$n_0-n_{goal}$')

        axs[5] = fig.add_subplot(2, 3, 6)
        axs[5].set_xlabel('Iterations')
        axs[5].set_ylabel(r'$\mu_0/U$')

        for i in range(6):
            axs[i].grid()
    
    else:

        fig, axs = figure

    num_plots = len(axs[0].lines)
    color = color_list[num_plots % len(color_list)]

    iterations = np.arange(1,len(convergence['diff'])+1)
    axs[0].plot(iterations, convergence['diff'], '.--', markersize=4, color=color, label=label, alpha=1.)
    axs[0].legend()

    axs[1].plot(iterations, convergence['alpha'], '.--', markersize=4, color=color, label=label, alpha=1.)
    axs[1].legend()

    axs[2].plot(iterations, np.array(convergence['n'])-density, '.--', markersize=4, color=color, label=label, alpha=1.)
    axs[2].legend()

    axs[3].plot(iterations, np.array(convergence['mu'])/U, '.--', markersize=4, color=color, label=label, alpha=1.)
    axs[3].legend()

    axs[4].plot(iterations, np.array(convergence['n0'])-density, '.--', markersize=4, color=color, label=label, alpha=1.)
    axs[4].legend()

    axs[5].plot(iterations, np.array(convergence['mu0'])/U, '.--', markersize=4, color=color, label=label, alpha=1.)
    axs[5].legend() 

    fig.suptitle(rf'Convergence - {title}')
    fig.tight_layout()

    return fig, axs


def pcut_dmft(results, var_label, title='', label='', figure=None):

    if figure is None:

        fig = plt.figure(figsize=(12, 5))
        axs = [None]*3

        axs[0] = fig.add_subplot(1, 2, 1)
        axs[0].set_xlabel(var_label)
        axs[0].set_ylabel('$Z$')
        axs[0].set_title('Quasi-particle weight')
        axs[0].grid()

        axs[1] = fig.add_subplot(2, 2, 2)
        axs[1].set_xlabel(var_label)
        axs[1].set_ylabel(r'$\gamma$')
        axs[1].set_title('Quasi-particle scattering rate')
        axs[1].grid()

        axs[2] = fig.add_subplot(2, 2, 4)
        axs[2].set_xlabel(var_label)
        axs[2].set_ylabel(r'$\tau$')
        axs[2].set_title('Quasi-particle lifetime')
        axs[2].grid()

    else:

        fig, axs = figure

    num_plots = len(axs[0].lines)
    color = color_list[num_plots % len(color_list)]
    
    sorted_keys = sorted((key for key in results if key != 'inputs'), key=lambda k: dict(k)[var_label])
    var = [dict(key)[var_label] for key in sorted_keys]

    Z_list = []
    gamma_list = []
    tau_list = []
    for par_key in sorted_keys:
        
        Z, gamma, tau = get_Z(results[par_key]['S_iw'], 1/dict(par_key)['T'], results['inputs']['n_iw'])
        Z_list.append(Z)
        gamma_list.append(gamma)
        tau_list.append(tau)

    axs[0].plot(var, Z_list, 'o--', markersize=4, color=color, label=label)
    axs[1].plot(var, gamma_list, 'o--', markersize=4, color=color, label=label)
    axs[2].plot(var, tau_list, 'o--', markersize=4, color=color, label=label)
    for ax in axs:
        ax.legend()

    fig.suptitle(f'Observables - {title}')
    fig.tight_layout()

    return fig, axs

def pcut_chi(results, var_label, plot_Q=(1,1,1), fit=False, x_exp=(1,1),
             title='', label='', figure=None, alpha=1., color=None, styles=('o-','o-')):

    has_xi = results['get_xi']
    if not isinstance(x_exp, tuple):
        x_exp = (x_exp,)

    if figure is None:
        dim = len(results['Q'][0])
        n_Q = sum(plot_Q)
        fig = plt.figure(figsize=(12, 6))

        outer_gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[2, 1])

        # --- Left panel: chi (and optionally xi) as nested subplots ---
        n_left = 2 if has_xi else 1
        left_gs = gridspec.GridSpecFromSubplotSpec(1, n_left, subplot_spec=outer_gs[0], wspace=0.3)

        ax_chi = fig.add_subplot(left_gs[0])

        if x_exp[0] == 1:
            x_label = var_label
        elif x_exp[0] == 0.5:
            x_label = rf'$\sqrt{{{var_label}}}$'
        else:
            x_label = rf'${var_label}^{x_exp[0]:.2g}$'

        ax_chi.set_xlabel(x_label)
        ax_chi.set_ylabel(r'$\chi^{-1}(\mathbf{Q})$')
        ax_chi.set_title('Susceptibility')
        ax_chi.grid()

        ax_xi = None
        if has_xi:

            if x_exp[1] == 1:
                x_label = var_label
            elif x_exp[1] == 0.5:
                x_label = rf'$\sqrt{{{var_label}}}$'
            else:
                x_label = rf'${var_label}^{x_exp[1]:.2g}$'

            ax_xi = fig.add_subplot(left_gs[1])
            ax_xi.set_title('Correlation length')
            ax_xi.set_xlabel(x_label)
            ax_xi.set_ylabel(r'$\xi^{-1}$')
            ax_xi.grid()

        # --- Right panel: Q components as nested subplots ---
        right_gs = gridspec.GridSpecFromSubplotSpec(n_Q, 1, subplot_spec=outer_gs[1], hspace=0.15)

        Q_label = [r'$Q_x/\pi-1$', r'$Q_y/\pi-1$', r'$Q_z/\pi-1$']
        ax_Q = []
        i = 0
        for d in range(dim):
            if plot_Q[d] == 1:
                ax = fig.add_subplot(right_gs[i])
                ax.set_ylabel(Q_label[d])
                ax.grid()
                ax_Q.append(ax)
                i += 1
        ax_Q[-1].set_xlabel(var_label)
        if len(ax_Q) > 1:
            for ax in ax_Q[:-1]:
                ax.tick_params(axis='x', labelbottom=False)

        if ax_Q:
            ax_Q[0].set_title(r'$\mathbf{Q}$ vector')

        axs = [ax_chi, ax_xi] + ax_Q

    else:
        fig, axs = figure
        ax_chi = axs[0]
        ax_xi  = axs[1]

    # Data arrays
    var_arr = np.array([par[var_label] for par in results['par_list']])
    Q = np.array(list(zip(*results['Q'])))
    invchi = np.array(results['invchi'])

    if color is None:
        used_colors = {line.get_color() for line in axs[0].lines}
        color = next(c for c in color_list if c not in used_colors)

    fit_par_chi = None
    if fit:
        pos_idx = np.where(invchi > 0)
        if len(pos_idx[0]) > 2:
            fit_par_chi = np.polyfit(var_arr[pos_idx][:15]**x_exp[0], invchi[pos_idx][:15], 1)
            x0 = var_arr[pos_idx][0]**x_exp[0]
            ax_chi.axline((x0, x0*fit_par_chi[0] + fit_par_chi[1]), slope=fit_par_chi[0], linestyle='--', color=color, alpha=alpha)
        else:
            fit_par_chi = [np.nan]*2

    # Plot chi
    ax_chi.plot(var_arr**x_exp[0], invchi, styles[0], markersize=4, color=color, label=label, alpha=alpha)

    # Plot xi if available
    fit_par_xi = None
    if has_xi:
        invxi = np.array(results['invxi'])

        if fit:
            pos_idx = np.where(invxi > 0)
            if len(pos_idx[0]) > 2:
                fit_par_xi = np.polyfit(var_arr[pos_idx][:15]**x_exp[1], invxi[pos_idx][:15], 1)
                x0 = var_arr[pos_idx][0]**x_exp[1]
                ax_xi.axline((x0, x0*fit_par_xi[0] + fit_par_xi[1]), slope=fit_par_xi[0], linestyle='--', color=color, alpha=alpha)
        else:
            fit_par_xi = [np.nan]*2
            
        ax_xi.plot(var_arr**x_exp[1], invxi, styles[0], markersize=4, color=color, alpha=alpha)

    # Plot Q components
    dim = len(results['Q'][0])
    i = 0
    for d in range(dim):
        if plot_Q[d] == 1:
            i += 1
            axs[i+1].plot(var_arr, Q[d]-1, styles[1], markersize=4, color=color, alpha=alpha)

    if label != '':
        ax_chi.legend()
    
    #if has_xi:
    #    chi_xlim = tuple(np.sign(x)*abs(x)**x_exp[1] for x in axs[-1].get_xlim())
    #    ax_xi.set_xlim(chi_xlim)

    fig.suptitle(title)
    fig.tight_layout()

    return (fig, axs), (fit_par_chi, fit_par_xi)

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