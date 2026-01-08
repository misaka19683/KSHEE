import numpy as np
import pandas as pd
from tenpy import MPS

def _get_bc_type(model) -> str:
    """Return 'open' or 'periodic' based on model.lat.boundary_conditions or model.lat.bc."""
    lat = getattr(model, 'lat', None)
    if lat is None: return "open"
    for attr in ["boundary_conditions", "bc"]:
        val = getattr(lat, attr, [])
        vals = [val] if isinstance(val, (str, bool)) else val
        if any(v is True or (isinstance(v, str) and "periodic" in v.lower()) for v in vals):
            return "periodic"
    return "open"

def calc_nn_bond_energies(psi:MPS, model):
    """
    Calculate nearest-neighbor bond energies for the Kitaev-Gamma chain.

    For OBC: returns length (L-1) bonds: (0,1)...(L-2,L-1)
    For PBC: returns length L bonds, including the wrap bond: (L-1,0)
    """
    L = psi.L
    bc_type = _get_bc_type(model)

    n_bonds = L if bc_type == "periodic" else L - 1
    E_bonds = np.zeros(n_bonds, dtype=float)
    
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

    for i in range(n_bonds):
        j_site = (i + 1) % L
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

    Convention:
    - We add energy(i, i+2) to the bond index i.
    - For PBC, indices wrap around.
    """
    # First, calculate the NN part using the existing logic
    E_bonds = calc_nn_bond_energies(psi, model)
    L = psi.L
    bc_type = _get_bc_type(model)

    # Get NNN parameters
    k2 = model.options.get('k2', 0.)
    j2 = model.options.get('j2', 0.)

    # If parameters are effectively zero, return NN result directly
    if abs(k2) < 1e-12 and abs(j2) < 1e-12:
        return E_bonds

    n_bonds = L if bc_type == "periodic" else L - 1
    n_terms = L if bc_type == "periodic" else L - 2

    # Iterate again to add NNN terms
    # We add energy(i, i+2) to E_bonds[i]
    for i in range(n_terms):
        k_site = (i + 2) % L
        E_nnn = 0.0

        # K2 S^z_i S^z_{i+2}
        if abs(k2) > 1e-12:
            E_nnn += k2 * np.real(psi.expectation_value_term([('Sz', i), ('Sz', k_site)]))

        # J2 S_i . S_{i+2}
        if abs(j2) > 1e-12:
            for op in ['Sx', 'Sy', 'Sz']:
                E_nnn += j2 * np.real(psi.expectation_value_term([(op, i), (op, k_site)]))

        # add to the bond index i (which exists for both OBC/PBC in this convention)
        E_bonds[i % n_bonds] += E_nnn

    return E_bonds

def extract_alternating_energy(E_bonds, bc_type="open"):
    """
    Extract the alternating component of the energy density.

    For OBC: use pandas shift and drop boundaries.
    For PBC: use circular neighbors (np.roll), no dropna.
    """
    E_bonds = np.asarray(E_bonds, dtype=float)

    if bc_type == "periodic":
        left = np.roll(E_bonds, 1)
        right = np.roll(E_bonds, -1)
        E_local_avg = 0.5 * (left + right)
        E_A = np.abs(E_bonds - E_local_avg)
        # Return pandas Series to keep downstream code unchanged-ish
        idx = np.arange(len(E_bonds))
        return pd.Series(E_A, index=idx), pd.Series(E_local_avg, index=idx)

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
        If True, use Iteratively Reweighted The Least Squares (IRLS) with Huber weights.

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
            # Simple Huber-IRLS (Iteratively Reweighted The Least Squares)
            # This is more robust against outliers
            X = np.vstack([log_x, np.ones_like(log_x)]).T
            w = np.ones_like(log_x)

            beta = np.array([0.0, 0.0])
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

def calculate_central_charge(psi: MPS, L: int, bc_type="open"):
    """
    Calculate the central charge from entanglement entropy.
    S(x) = (c/6) * ln[sin(pi*x/L)] + constant (for OBC)
    S(x) = (c/3) * ln[sin(pi*x/L)] + constant (for PBC)

    Parameters
    ----------
    psi : tenpy.MPS
        The ground state MPS.
    L : int
        Number of sites.
    bc_type : str, optional
        Boundary condition type, either "open" or "periodic". Default is "open".

    Returns
    -------
    res : dict
        A dictionary containing:
        - 'c_avg': Averaged central charge from odd and even sites.
        - 'odd', 'even': Dicts with 'c', 'intercept', 'x_fit', 'S_fit' for each parity.
        - 'S': All entanglement entropy values.
        - 'x_plot': All ln((L/pi)*sin(pi*x/L)) values.
    """
    S = psi.entanglement_entropy()
    x_real = np.arange(1, L)

    mask_range = (x_real >= L // 8) & (x_real <= 7 * L // 8)

    sin_val = np.sin(np.pi * x_real / L)
    sin_val = np.maximum(sin_val, 1e-12)
    x_plot = np.log((L / np.pi) * sin_val)

    lam = 1.0 if bc_type == "periodic" else 0.5

    mask_odd = mask_range & ((x_real % 2) != 0)
    x_odd = x_plot[mask_odd]
    S_odd = S[mask_odd]

    mask_even = mask_range & ((x_real % 2) == 0)
    x_even = x_plot[mask_even]
    S_even = S[mask_even]

    res = {'S': S, 'x_plot': x_plot, 'L': L, 'bc_type': bc_type}

    if len(x_odd) > 2:
        slope, intercept = np.polyfit(x_odd, S_odd, 1)
        c_odd = slope * 3.0 / lam
        res['odd'] = {'c': c_odd, 'slope': slope, 'intercept': intercept, 'x_fit': x_odd, 'S_fit': S_odd}

    if len(x_even) > 2:
        slope, intercept = np.polyfit(x_even, S_even, 1)
        c_even = slope * 3.0 / lam
        res['even'] = {'c': c_even, 'slope': slope, 'intercept': intercept, 'x_fit': x_even, 'S_fit': S_even}

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


def calculate_dimerization(psi: MPS, model):
    """
    Calculate dimerization order parameter.
    First calculate rotated bond energies B'_i, then O_i = B'_{i+1} - B'_i.
    """
    L = psi.L
    bc_type = _get_bc_type(model)
    n_bonds = L if bc_type == "periodic" else L - 1
    
    b_prime = np.zeros(n_bonds)
    for i in range(n_bonds):
        j = (i + 1) % L
        # i is a 0-based index.
        # i=0, 2, ... (even i) corresponds to Type 1 bond (1-2, 3-4...)
        # i=1, 3, ... (odd i) corresponds to Type 2 bond (2-3, 4-5...)
        
        if i % 2 == 0:
            # Type 1 bond: -SxSx + SySz + SzSy
            val = -psi.expectation_value_term([('Sx', i), ('Sx', j)]) \
                  + psi.expectation_value_term([('Sy', i), ('Sz', j)]) \
                  + psi.expectation_value_term([('Sz', i), ('Sy', j)])
        else:
            # Type 2 bond: -SySy + SzSx + SxSz
            val = -psi.expectation_value_term([('Sy', i), ('Sy', j)]) \
                  + psi.expectation_value_term([('Sz', i), ('Sx', j)]) \
                  + psi.expectation_value_term([('Sx', i), ('Sz', j)])
        b_prime[i] = np.real(val)
    
    dimer = np.zeros(n_bonds)
    for i in range(n_bonds):
        dimer[i] = b_prime[(i + 1) % n_bonds] - b_prime[i]
        
    return dimer


def perform_full_analysis(psi: MPS, model, title=""):
    """
    Full analysis:
    - OBC: calculate Luttinger parameter (kappa) and central charge.
    - PBC: calculate central charge and dimerization.
    """
    L = model.lat.N_sites
    bc_type = _get_bc_type(model)

    # 1. Bond Energies
    E_bonds = calc_nn_bond_energies(psi, model)
    E_A, E_local_avg = extract_alternating_energy(E_bonds, bc_type=bc_type)

    results = {
        'title': title,
        'L': L,
        'bc_type': bc_type,
        'E_bonds': E_bonds,
        'E_A': E_A,
        'E_local_avg': E_local_avg,
    }

    # 2. Luttinger parameter fitting (Only for OBC)
    if bc_type == "open":
        kappa, p_luttinger, x_luttinger, y_luttinger = fit_luttinger_parameter(E_A, L)
        results.update({
            'kappa': kappa,
            'p_luttinger': p_luttinger,
            'x_luttinger': x_luttinger,
            'y_luttinger': y_luttinger,
        })
    else:
        results.update({
            'kappa': np.nan,
            'p_luttinger': None,
            'x_luttinger': None,
            'y_luttinger': None,
        })

    # 3. Central charge (Both OBC and PBC)
    cc_res = calculate_central_charge(psi, L, bc_type=bc_type)
    results['cc_res'] = cc_res

    # 4. Dimerization (Only for PBC)
    if bc_type == "periodic":
        dimer_res = calculate_dimerization(psi, model)
        results['dimerization'] = dimer_res

    return results


