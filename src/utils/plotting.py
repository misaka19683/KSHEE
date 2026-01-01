import matplotlib.pyplot as plt
import numpy as np
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
    Plot the linear fit of ln|EA| vs ln(rL).
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

    fig = plt.figure(figsize=(6, 4.5), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')

    # 5. 拟合与绘制
    # 奇数点 (蓝色)
    if 'odd' in res:
        plt.scatter(res['odd']['x_fit'], res['odd']['S_fit'], color='blue', facecolors='none', s=60, linewidths=1.5)
        x_line = np.linspace(min(res['odd']['x_fit']), max(res['odd']['x_fit']), 100)
        plt.plot(x_line, res['odd']['c'] * x_line + res['odd']['intercept'], 'b-', linewidth=1.5, label=f'c ~ {res["odd"]["c"]:.3f}')

    # 偶数点 (红色)
    if 'even' in res:
        plt.scatter(res['even']['x_fit'], res['even']['S_fit'], color='red', facecolors='none', s=60, linewidths=1.5)
        x_line = np.linspace(min(res['even']['x_fit']), max(res['even']['x_fit']), 100)
        plt.plot(x_line, res['even']['c'] * x_line + res['even']['intercept'], 'r-', linewidth=1.5, label=f'c ~ {res["even"]["c"]:.3f}')

    # --- 字体、边框及坐标轴格式调整 ---
    phi_str = f", $\\phi = {phi_val/np.pi:.2f}\\pi$" if phi_val is not None else ""

    plt.title(f"{title_prefix} $L={L}${phi_str}, $c \\sim {c_avg:.3f}$", color='black', fontsize=18)
    plt.xlabel(r"$(1/6) \ln[\sin(x \pi / L)]$", color='black', fontsize=16)
    plt.ylabel(r"$S_L(x)$", color='black', fontsize=16)

    # 坐标轴边框
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)

    # 刻度设置
    plt.tick_params(axis='x', colors='black', direction='in', which='both', top=True, labelsize=14, width=1.5, length=6)
    plt.tick_params(axis='y', colors='black', direction='in', which='both', right=True, labelsize=14, width=1.5, length=6)

    # 设置 Y 轴刻度格式为两位小数
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # 图例设置
    legend = plt.legend(loc='lower right', frameon=True, edgecolor='black', fancybox=False, facecolor='white', fontsize=14)
    legend.get_frame().set_linewidth(1.5)
    for text in legend.get_texts():
        text.set_color("black")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_analysis_results(results, phi_val=None, save_prefix=None):
    """
    Plot all analysis results:
    1. Luttinger fit
    2. Central Charge fit
    """
    # 1. Plot Luttinger fit
    luttinger_save = f"{save_prefix}_luttinger.png" if save_prefix else None
    plot_luttinger_fit(
        results['x_luttinger'],
        results['y_luttinger'],
        results['kappa'],
        results['p_luttinger'],
        title="Luttinger Parameter Extraction",
        # title=f"Luttinger Fit {results['title']}",
        save_path=luttinger_save
    )

    # 2. Plot Central Charge fit
    cc_save = f"{save_prefix}_cc.png" if save_prefix else None
    plot_central_charge(
        results['cc_res'],
        # title_prefix=results['title'],
        phi_val=phi_val,
        save_path=cc_save
    )