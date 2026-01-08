import numpy as np
import logging
from src.models.kitaev_gamma import KitaevGammaChain
from src.simulation.dmrg_runner import run_dmrg
from src.physics.measurements import perform_full_analysis
from src.utils.plotting import plot_analysis_results

def run_analysis(L, params, title="", plot=False):
    print(f"\n--- Running analysis for L={L} {title} ---")
    # 1. Initialize model
    model_params = params.copy()
    model_params['L'] = L
    model = KitaevGammaChain(model_params)

    # 2. Run DMRG
    psi, info = run_dmrg(model)
    print(f"Final Energy: {info['E']:.8f}")

    # 3. Perform full analysis (Measurements + Fitting)
    results = perform_full_analysis(psi, model, title=title)
    print(f"Extracted Kappa: {results['kappa']:.6f}")
    print(f"Extracted Central Charge: {results['cc_res']['c_avg']:.6f}")

    # 4. Plotting
    if plot:
        phi_val = params.get('phi', None)
        plot_analysis_results(results, phi_val=phi_val)

    return results['kappa']

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    
    # Base parameters
    theta = np.pi * 0.44
    phi = np.pi * 0.85
    base_params = {
        'gamma': np.sin(theta) * np.sin(phi),
        'k': np.sin(theta) * np.cos(phi),
        'j': np.cos(theta),
        'gamma_prime': 0.1,
        'k2': 0.04,
        'j2': 0.02,
        'lambda_in_dp': 1.5,
        'lambda_in_dn': 0.5,
        'lambda_out': 0.3,
        'lambda_gamma_out': 0.9,
    }
    
    # Case 1: No electric field
    params_no_field = base_params.copy()
    params_no_field.update({'Ex': 0.0, 'Ey': 0.0, 'Ez': 0.0})
    kappa_no_field = run_analysis(48, params_no_field, title="(No Field)")
    
    # Case 2: (1,1,1) electric field
    params_field = base_params.copy()
    field_val = 1.0 / np.sqrt(3)
    params_field.update({'Ex': field_val, 'Ey': field_val, 'Ez': field_val})
    kappa_field = run_analysis(48, params_field, title="(E=[1,1,1])")
    
    print("\n" + "="*30)
    print(f"Comparison Result:")
    print(f"Kappa (No Field):    {kappa_no_field:.6f}")
    print(f"Kappa (With Field):  {kappa_field:.6f}")
    print("="*30)
