import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter


def plot_energy_profile(E_bonds, E_local_avg, L_total, title="Energy Profile"):
    """
    Plot the bond energy profile and local average.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(E_bonds, 'o-', markersize=3, label='Bond Energy')
    plt.plot(E_local_avg.index, E_local_avg.values, 'r--', linewidth=1, label='Local Avg')
    plt.title(f"{title} (L={L_total})")
    plt.xlabel('Bond Index')
    plt.ylabel('Energy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_luttinger_fit(x_data, y_data, kappa, p, title="Luttinger Parameter Extraction", save_path=None):
    """
    Plot the linear fit of ln|EA| vs. ln(rL).
    Optimized for publication quality:
    - Linear axes for log-transformed data.
    - No scientific notation.
    - Large fonts and bold lines.
    """
    # --- 绘图设置：强制白底黑字 ---
    fig = plt.figure(figsize=(6, 4.5), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')

    if x_data is not None and len(x_data) > 0:
        # 1. 转换数据为自然对数
        ln_x = np.log(x_data)
        ln_y = np.log(y_data)

        # 2. 绘制数据点：空心蓝色圆圈
        plt.scatter(ln_x, ln_y, color='blue', facecolors='none', s=60, linewidths=1.5, label='Data', zorder=2)

        # 3. 绘制拟合线：Y = p[0]*X + p[1]
        # p[0] 是斜率 (-kappa), p[1] 是截距 (lnC)
        fit_ln_y = p[0] * ln_x + p[1]
        plt.plot(ln_x, fit_ln_y, 'r-', linewidth=2, label=f'Fit: $\\kappa={kappa:.4f}$', zorder=1)

        # --- 坐标轴格式调整 ---
        # 强制使用普通数值格式，不使用科学计数法
        for axis in [ax.xaxis, ax.yaxis]:
            formatter = ScalarFormatter()
            formatter.set_scientific(False)
            axis.set_major_formatter(formatter)

        # 标题和标签
        plt.title(title, color='black', fontsize=18)
        plt.xlabel(r'$\ln(r_L)$', color='black', fontsize=16)
        plt.ylabel(r'$\ln|E_A|$', color='black', fontsize=16)

        # 边框加粗
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(1.5)

        # 刻度样式
        plt.tick_params(axis='both', which='both', colors='black', direction='in',
                        labelsize=14, width=1.5, length=6, top=True, right=True)

        # 图例
        legend = plt.legend(loc='best', frameon=True, edgecolor='black', fancybox=False, facecolor='white', fontsize=14)
        legend.get_frame().set_linewidth(1.5)
        for text in legend.get_texts():
            text.set_color("black")

        plt.grid(True, alpha=0.3)
    else:
        plt.title("Insufficient data for fit", color='black', fontsize=18)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_central_charge(res, title_prefix="", phi_val=None, save_path=None):
    """
    Plot the central charge fit results.
    Optimized for publication quality.
    """
    L = res['L']
    c_avg = res['c_avg']
    bc_type = res.get('bc_type', 'open')

    fig = plt.figure(figsize=(6, 4.5), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')

    # 5. 拟合与绘制
    # 奇数点 (蓝色)
    if 'odd' in res:
        plt.scatter(res['odd']['x_fit'], res['odd']['S_fit'], color='blue', facecolors='none', s=60, linewidths=1.5)
        x_line = np.linspace(min(res['odd']['x_fit']), max(res['odd']['x_fit']), 100)
        plt.plot(x_line, res['odd']['slope'] * x_line + res['odd']['intercept'], 'b-', linewidth=1.5, label=f'c ~ {res["odd"]["c"]:.3f}')

    # 偶数点 (红色)
    if 'even' in res:
        plt.scatter(res['even']['x_fit'], res['even']['S_fit'], color='red', facecolors='none', s=60, linewidths=1.5)
        x_line = np.linspace(min(res['even']['x_fit']), max(res['even']['x_fit']), 100)
        plt.plot(x_line, res['even']['slope'] * x_line + res['even']['intercept'], 'r-', linewidth=1.5, label=f'c ~ {res["even"]["c"]:.3f}')

    # --- 字体、边框及坐标轴格式调整 ---
    title = f"$L={L}$, $c \\sim {c_avg:.3f}$"
    if title_prefix:
        title = f"{title_prefix} {title}"
    plt.title(title, color='black', fontsize=18)
    
    coeff_str = "1/3" if bc_type == 'periodic' else "1/6"
    plt.xlabel(f"$({coeff_str}) \\ln[(L/\\pi) \\sin(x \\pi / L)]$", color='black', fontsize=16)
    plt.ylabel(r"$S_L(x)$", color='black', fontsize=16)

    # 坐标轴边框
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)

    # 刻度设置
    plt.tick_params(axis='x', colors='black', direction='in', which='both', top=True, labelsize=14, width=1.5, length=6)
    plt.tick_params(axis='y', colors='black', direction='in', which='both', right=True, labelsize=14, width=1.5, length=6)

    # 设置 Y 轴刻度格式为两位小数
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # 图例设置
    legend = plt.legend(loc='lower right', frameon=True, edgecolor='black', fancybox=False, facecolor='white', fontsize=14)
    legend.get_frame().set_linewidth(1.5)
    for text in legend.get_texts():
        text.set_color("black")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_dimerization(dimerization, title_prefix="", save_path=None):
    """
    Plot the dimerization order parameter.
    """
    fig = plt.figure(figsize=(6, 4.5), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')

    plt.plot(dimerization, 'go-', markersize=4, linewidth=1.5, label='Dimerization')
    
    title = "Dimerization"
    if title_prefix:
        title = f"{title_prefix} {title}"
    plt.title(title, color='black', fontsize=18)
    plt.xlabel('Bond Index', color='black', fontsize=16)
    plt.ylabel(r'$O_i^{dim}$', color='black', fontsize=16)

    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)

    plt.tick_params(axis='both', which='both', colors='black', direction='in', labelsize=14, width=1.5, length=6, top=True, right=True)
    
    # 强制不使用科学计数法，并确保数值接近 0 时不出现奇怪的标注
    y_formatter = ScalarFormatter(useOffset=False)
    y_formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(y_formatter)
    
    # 设置 y 轴范围：如果数据极小（接近 0），则固定范围，防止放大噪声
    data_range = np.max(dimerization) - np.min(dimerization)
    data_max_abs = np.max(np.abs(dimerization))
    
    if data_max_abs < 1e-4:
        ax.set_ylim(-0.01, 0.01)
    else:
        # 给予 10% 的上下余量
        padding = max(0.01, 0.1 * data_range)
        ax.set_ylim(np.min(dimerization) - padding, np.max(dimerization) + padding)

    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_analysis_results(results, phi_val=None, save_prefix=None, show_field_info=True):
    """
    Plot analysis results based on boundary conditions:
    - OBC: Luttinger fit and Central Charge fit.
    - PBC: Central Charge fit and Dimerization.
    """
    bc_type = results.get('bc_type', 'open')
    # Use explicit title from results only if show_field_info is True
    title_prefix = results.get('title', "") if show_field_info else ""
    
    print(f"Generating plots for {bc_type} task...")

    # Cleanup old/incorrect files to avoid "three images" confusion
    if save_prefix:
        if bc_type == 'open':
            dimer_file = f"{save_prefix}_dimer.png"
            if os.path.exists(dimer_file):
                try: os.remove(dimer_file)
                except: pass
        else: # periodic
            luttinger_file = f"{save_prefix}_luttinger.png"
            if os.path.exists(luttinger_file):
                try: os.remove(luttinger_file)
                except: pass

    if bc_type == 'open':
        # 1. Plot Luttinger fit (OBC only)
        luttinger_save = f"{save_prefix}_luttinger.png" if save_prefix else None
        print(f"  -> Plotting Luttinger fit to {luttinger_save}")
        plot_luttinger_fit(
            results.get('x_luttinger'),
            results.get('y_luttinger'),
            results.get('kappa'),
            results.get('p_luttinger'),
            title=f"{title_prefix} Luttinger Fit" if title_prefix else "Luttinger Parameter Extraction",
            save_path=luttinger_save
        )

        # 2. Plot Central Charge fit
        cc_save = f"{save_prefix}_cc.png" if save_prefix else None
        print(f"  -> Plotting Central Charge fit to {cc_save}")
        plot_central_charge(
            results['cc_res'],
            title_prefix=title_prefix,
            phi_val=phi_val,
            save_path=cc_save
        )

    else:  # periodic
        # 1. Plot Central Charge fit
        cc_save = f"{save_prefix}_cc.png" if save_prefix else None
        print(f"  -> Plotting Central Charge fit (PBC) to {cc_save}")
        plot_central_charge(
            results['cc_res'],
            title_prefix=title_prefix,
            phi_val=phi_val,
            save_path=cc_save
        )

        # 2. Plot Dimerization (PBC only)
        if 'dimerization' in results:
            dimer_save = f"{save_prefix}_dimer.png" if save_prefix else None
            print(f"  -> Plotting Dimerization to {dimer_save}")
            plot_dimerization(
                results['dimerization'],
                title_prefix=title_prefix,
                save_path=dimer_save
            )
        else:
            print("  Warning: 'dimerization' data not found in results even though bc_type is periodic.")