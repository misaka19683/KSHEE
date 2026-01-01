import numpy as np
import pandas as pd
from tenpy import MPS


def calc_nn_bond_energies(psi:MPS, model):
    """
    Calculate nearest-neighbor bond energies for the Kitaev-Gamma chain.
    """
    L = psi.L
    E_bonds = np.zeros(L - 1)
    
    # Get parameters from model options
    j = model.options.get('j', 0.)
    k = model.options.get('k', 0.)
    gamma = model.options.get('gamma', 0.)
    gamma_prime = model.options.get('gamma_prime', 0.)
    
    Ex = model.options.get('Ex', 0.)
    Ey = model.options.get('Ey', 0.)
    Ez = model.options.get('Ez', 0.)
    
    lambda_in_dp = model.options.get('lambda_in_dp', 0.)
    lambda_in_dn = model.options.get('lambda_in_dn', 0.)
    lambda_out = model.options.get('lambda_out', 0.)
    lambda_gamma_out = model.options.get('lambda_gamma_out', 0.)
    
    L_in_plus = lambda_in_dp + lambda_in_dn
    L_in_minus = lambda_in_dp - lambda_in_dn

    for i in range(L - 1):
        j_site = i + 1
        bond_type = i % 2
        E_bond = 0.0
        
        # Heisenberg J
        for op in ['Sx', 'Sy', 'Sz']:
            E_bond += j * np.real(psi.expectation_value_term([(op, i), (op, j_site)]))

        if bond_type == 0: # Type 1
            # Kitaev K and Gamma
            E_bond += k * np.real(psi.expectation_value_term([('Sx', i), ('Sx', j_site)]))
            v = psi.expectation_value_term([('Sy', i), ('Sz', j_site)]) + psi.expectation_value_term([('Sz', i), ('Sy', j_site)])
            E_bond += gamma * np.real(v)
            # Gamma'
            for o1, o2 in [('Sx', 'Sy'), ('Sy', 'Sx'), ('Sx', 'Sz'), ('Sz', 'Sx')]:
                E_bond += gamma_prime * np.real(psi.expectation_value_term([(o1, i), (o2, j_site)]))
            
            # Electric Field effects
            if abs(Ex) > 1e-12:
                for o1, o2, s in [('Sz', 'Sx', 1), ('Sx', 'Sz', -1), ('Sx', 'Sy', 1), ('Sy', 'Sx', -1)]:
                    E_bond += s * Ex * lambda_out * np.real(psi.expectation_value_term([(o1, i), (o2, j_site)]))
                for o1, o2 in [('Sx', 'Sy'), ('Sy', 'Sx'), ('Sx', 'Sz'), ('Sz', 'Sx')]:
                    E_bond += Ex * lambda_gamma_out * np.real(psi.expectation_value_term([(o1, i), (o2, j_site)]))
            if abs(Ey) > 1e-12:
                v = psi.expectation_value_term([('Sy', i), ('Sz', j_site)]) - psi.expectation_value_term([('Sz', i), ('Sy', j_site)])
                E_bond += Ey * L_in_plus * np.real(v)
            if abs(Ez) > 1e-12:
                v = psi.expectation_value_term([('Sy', i), ('Sz', j_site)]) - psi.expectation_value_term([('Sz', i), ('Sy', j_site)])
                E_bond += Ez * L_in_minus * np.real(v)
        else: # Type 2
            # Kitaev K and Gamma
            E_bond += k * np.real(psi.expectation_value_term([('Sy', i), ('Sy', j_site)]))
            v = psi.expectation_value_term([('Sz', i), ('Sx', j_site)]) + psi.expectation_value_term([('Sx', i), ('Sz', j_site)])
            E_bond += gamma * np.real(v)
            # Gamma'
            for o1, o2 in [('Sy', 'Sz'), ('Sz', 'Sy'), ('Sx', 'Sy'), ('Sy', 'Sx')]:
                E_bond += gamma_prime * np.real(psi.expectation_value_term([(o1, i), (o2, j_site)]))
            
            # Electric Field effects
            if abs(Ex) > 1e-12:
                v = psi.expectation_value_term([('Sz', i), ('Sx', j_site)]) - psi.expectation_value_term([('Sx', i), ('Sz', j_site)])
                E_bond += Ex * L_in_minus * np.real(v)
            if abs(Ey) > 1e-12:
                for o1, o2, s in [('Sx', 'Sy', 1), ('Sy', 'Sx', -1), ('Sy', 'Sz', 1), ('Sz', 'Sy', -1)]:
                    E_bond += s * Ey * lambda_out * np.real(psi.expectation_value_term([(o1, i), (o2, j_site)]))
                for o1, o2 in [('Sy', 'Sz'), ('Sz', 'Sy'), ('Sy', 'Sx'), ('Sx', 'Sy')]:
                    E_bond += Ey * lambda_gamma_out * np.real(psi.expectation_value_term([(o1, i), (o2, j_site)]))
            if abs(Ez) > 1e-12:
                v = psi.expectation_value_term([('Sz', i), ('Sx', j_site)]) - psi.expectation_value_term([('Sx', i), ('Sz', j_site)])
                E_bond += Ez * L_in_plus * np.real(v)
        
        E_bonds[i] = E_bond
    return E_bonds

def calc_total_bond_energies(psi: MPS, model):
    """
    Calculate bond energies including Next-Nearest Neighbor (NNN) terms.
    The energy of the NNN interaction (i, i+2) is added to the bond E_bonds[i].
    """
    # First, calculate the NN part using the existing logic
    E_bonds = calc_nn_bond_energies(psi, model)
    L = psi.L

    # Get NNN parameters
    k2 = model.options.get('k2', 0.)
    j2 = model.options.get('j2', 0.)

    # If parameters are effectively zero, return NN result directly
    if abs(k2) < 1e-12 and abs(j2) < 1e-12:
        return E_bonds

    # Iterate again to add NNN terms
    # We add energy(i, i+2) to E_bonds[i]
    for i in range(L - 2):
        k_site = i + 2
        E_nnn = 0.0

        # K2 S^z_i S^z_{i+2}
        if abs(k2) > 1e-12:
            E_nnn += k2 * np.real(psi.expectation_value_term([('Sz', i), ('Sz', k_site)]))

        # J2 S_i . S_{i+2}
        if abs(j2) > 1e-12:
            for op in ['Sx', 'Sy', 'Sz']:
                E_nnn += j2 * np.real(psi.expectation_value_term([(op, i), (op, k_site)]))

        # Add to the bond at site i (which connects i and i+1)
        E_bonds[i] += E_nnn

    return E_bonds

def extract_alternating_energy(E_bonds):
    """
    Extract the alternating component of the energy density.
    """
    E_series = pd.Series(E_bonds)
    E_local_avg = (E_series.shift(1) + E_series.shift(-1)) / 2.0
    E_A = (E_series - E_local_avg).abs().dropna()
    return E_A, E_local_avg.dropna()


def fit_luttinger_parameter(E_A, L_total, eps=1e-12, min_points=6, use_robust=False):
    """
    Fit the Luttinger parameter kappa from the alternating energy density.

    This function implements a more robust fitting strategy:
    1. Filters out points where |E_A| is too small (<= eps) or r_L is invalid.
    2. Supports both standard Least Squares and Robust Regression (Huber weights).

    Parameters
    ----------
    E_A : pandas.Series
        Alternating component of energy density.
    L_total : int
        Total number of sites.
    eps : float
        Threshold for |E_A| to avoid log(0) or numerical noise.
    min_points : int
        Minimum number of points required for fitting.
    use_robust : bool
        If True, use Iteratively Reweighted Least Squares (IRLS) with Huber weights.

    Returns
    -------
    kappa : float
        The estimated Luttinger parameter.
    p : list
        [slope, intercept] of the fit.
    x_data : array
        The valid r_L data points used for fitting.
    y_data : array
        The valid |E_A| data points used for fitting.
    """
    # Calculate conformal distance r_L
    # Note: Using index + 1 to match the convention in the notebook snippet
    # 这个部分极其重要，这个数值一旦错误，图像就会变成一个圈，或者发生奇怪的弯曲
    r_phys = E_A.index.to_numpy(dtype=float) + 1
    r_L = (L_total / np.pi) * np.sin(np.pi * r_phys / L_total)

    y = E_A.to_numpy(dtype=float)

    # Filter valid points (bulk points with significant amplitude)
    ok = np.isfinite(r_L) & np.isfinite(y) & (r_L > 0) & (y > eps)
    x_data = r_L[ok]
    y_data = y[ok]

    kappa = np.nan
    slope = np.nan
    intercept = np.nan

    if len(x_data) >= min_points:
        log_x = np.log(x_data)
        log_y = np.log(y_data)

        if not use_robust:
            slope, intercept = np.polyfit(log_x, log_y, 1)
        else:
            # Simple Huber-IRLS (Iteratively Reweighted Least Squares)
            # This is more robust against outliers
            X = np.vstack([log_x, np.ones_like(log_x)]).T
            w = np.ones_like(log_x)

            for _ in range(15):  # Iterate to converge weights
                W = np.sqrt(w)
                # Weighted least squares
                beta, *_ = np.linalg.lstsq(X * W[:, None], log_y * W, rcond=None)
                resid = log_y - (X @ beta)

                # Robust scale estimate (MAD - Median Absolute Deviation)
                s = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-15
                c = 1.345 * s

                # Update Huber weights
                w = np.where(np.abs(resid) <= c, 1.0, c / np.abs(resid))

            slope, intercept = beta[0], beta[1]

        kappa = -slope

    return kappa, [slope, intercept], x_data, y_data

def fit_luttinger_parameter_middle_half(E_A, L_total, eps=1e-12, min_points=6, use_robust=False):
    """
    与原 fit_luttinger_parameter 返回格式完全一致：
    return kappa, [slope, intercept], x_data, y_data

    其中 x_data, y_data 被替换为“原函数有效点的中间一半”，并在该子集上重新拟合。
    """
    # 1) 先用原函数得到有效点（以及原拟合结果，但这里主要复用 x/y）
    kappa0, p0, x_data, y_data = fit_luttinger_parameter(
        E_A, L_total, eps=eps, min_points=min_points, use_robust=use_robust
    )

    x = np.asarray(x_data, dtype=float)
    y = np.asarray(y_data, dtype=float)

    # 2) 取中间一半（按当前顺序）
    n = len(x)
    if n == 0:
        return np.nan, [np.nan, np.nan], x, y

    keep = n // 2  # 保留一半（向下取整）
    start = (n - keep) // 2
    end = start + keep

    x_mid = x[start:end]
    y_mid = y[start:end]

    # 3) 在中间一半上重新拟合；若点数不够，按原格式返回 NaN
    if len(x_mid) < min_points:
        return np.nan, [np.nan, np.nan], x_mid, y_mid

    log_x = np.log(x_mid)
    log_y = np.log(y_mid)

    if not use_robust:
        slope, intercept = np.polyfit(log_x, log_y, 1)
    else:
        X = np.vstack([log_x, np.ones_like(log_x)]).T
        w = np.ones_like(log_x)
        beta = np.array([0.0, 0.0])

        for _ in range(15):
            W = np.sqrt(w)
            beta, *_ = np.linalg.lstsq(X * W[:, None], log_y * W, rcond=None)
            resid = log_y - (X @ beta)

            s = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-15
            c = 1.345 * s
            w = np.where(np.abs(resid) <= c, 1.0, c / np.abs(resid))

        slope, intercept = float(beta[0]), float(beta[1])

    kappa = -float(slope)
    return kappa, [float(slope), float(intercept)], x_mid, y_mid


def calculate_central_charge(psi: MPS, L: int):
    """
    Calculate the central charge from entanglement entropy.
    S(x) = (c/6) * ln[sin(pi*x/L)] + constant (for OBC)

    Parameters
    ----------
    psi : tenpy.MPS
        The ground state MPS.
    L : int
        Number of sites.

    Returns
    -------
    res : dict
        A dictionary containing:
        - 'c_avg': Averaged central charge from odd and even sites.
        - 'odd', 'even': Dicts with 'c', 'intercept', 'x_fit', 'S_fit' for each parity.
        - 'S': All entanglement entropy values.
        - 'x_plot': All (1/6)*ln(sin) values.
    """
    # 1. 获取纠缠熵
    S = psi.entanglement_entropy()
    x_real = np.arange(1, L)

    # 2. 定义拟合区间 (L/8 to 7L/8)
    mask_range = (x_real >= L // 8) & (x_real <= 7 * L // 8)

    # 3. 计算横坐标
    sin_val = np.sin(np.pi * x_real / L)
    sin_val = np.maximum(sin_val, 1e-10)
    x_plot = (1.0 / 6.0) * np.log(sin_val)

    # 4. 分离奇偶点并拟合
    # 奇数点 (MPS 索引 0, 2, ... 对应物理点 1, 3, ...)
    mask_odd = mask_range & ((x_real % 2) != 0)
    x_odd = x_plot[mask_odd]
    S_odd = S[mask_odd]

    # 偶数点 (MPS 索引 1, 3, ... 对应物理点 2, 4, ...)
    mask_even = mask_range & ((x_real % 2) == 0)
    x_even = x_plot[mask_even]
    S_even = S[mask_even]

    res = {'S': S, 'x_plot': x_plot, 'L': L}

    if len(x_odd) > 2:
        coeffs_odd = np.polyfit(x_odd, S_odd, 1)
        res['odd'] = {'c': coeffs_odd[0], 'intercept': coeffs_odd[1], 'x_fit': x_odd, 'S_fit': S_odd}

    if len(x_even) > 2:
        coeffs_even = np.polyfit(x_even, S_even, 1)
        res['even'] = {'c': coeffs_even[0], 'intercept': coeffs_even[1], 'x_fit': x_even, 'S_fit': S_even}

    c_odd = res.get('odd', {}).get('c', np.nan)
    c_even = res.get('even', {}).get('c', np.nan)

    if not np.isnan(c_odd) and not np.isnan(c_even):
        res['c_avg'] = (c_odd + c_even) / 2.0
    elif not np.isnan(c_odd):
        res['c_avg'] = c_odd
    elif not np.isnan(c_even):
        res['c_avg'] = c_even
    else:
        res['c_avg'] = np.nan

    return res


def perform_full_analysis(psi: MPS, model, title=""):
    """
    Perform a full analysis of the MPS state:
    1. Bond energies and alternating component.
    2. Luttinger parameter fitting.
    3. Central charge calculation.
    """
    L = model.lat.N_sites

    # 1. Bond Energies
    E_bonds = calc_nn_bond_energies(psi, model)
    E_A, E_local_avg = extract_alternating_energy(E_bonds)

    # 2. Luttinger parameter fitting
    kappa, p_luttinger, x_luttinger, y_luttinger = fit_luttinger_parameter(E_A, L)

    # 3. Central charge calculation
    cc_res = calculate_central_charge(psi, L)

    results = {
        'title': title,
        'L': L,
        'E_bonds': E_bonds,
        'E_A': E_A,
        'E_local_avg': E_local_avg,
        'kappa': kappa,
        'p_luttinger': p_luttinger,
        'x_luttinger': x_luttinger,
        'y_luttinger': y_luttinger,
        'cc_res': cc_res
    }

    return results


