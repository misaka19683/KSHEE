import numpy as np
import os
import pandas as pd
from src.models.kitaev_gamma import KitaevGammaChain
from src.simulation.dmrg_runner import run_dmrg
from src.physics.measurements import perform_full_analysis
from src.utils.plotting import plot_analysis_results


def save_task_data(results, task_name):
    """保存计算结果"""
    os.makedirs("results/data", exist_ok=True)
    summary_path = f"results/data/{task_name}_summary.csv"
    bc_type = results.get('bc_type', 'open')
    data = {'task': [task_name], 'central_charge': [results['cc_res']['c_avg']]}
    if bc_type == 'open':
        data['kappa'] = [results.get('kappa')]
    pd.DataFrame(data).to_csv(summary_path, index=False)
    energy_path = f"results/data/{task_name}_energy.csv"
    pd.DataFrame({'E_bonds': results['E_bonds']}).to_csv(energy_path, index=False)
    print(f"Results for {task_name} saved.")


def main():
    # 参数设置
    theta = np.pi * 0.44
    phi = np.pi * 0.85

    base_params = {
        'L': 96,
        'bc_MPS': 'finite',
        'lattice_params': {'bc': 'open'},
        'gamma': np.sin(theta) * np.sin(phi), 'k': np.sin(theta) * np.cos(phi), 'j': np.cos(theta),
        'gamma_prime': 0.1, 'k2': 0.02, 'j2': 0.2,
        'lambda_in_dp': -0.1, 'lambda_in_dn': -0.05,
        'lambda_out': 0.04, 'lambda_gamma_out': 0.05,
    }

    dmrg_params = {
        'mixer': True,
        'max_E_err': 1.e-10,
        'trunc_params': {'chi_max': 1000, 'svd_min': 1.e-10},
    }

    # 执行 j = -4 的任务
    task_name_jm4 = "task_j_m4"
    print(f"Starting task: {task_name_jm4}")

    model_params_jm4 = base_params.copy()
    field_val_jm4 = -4 / np.sqrt(3)
    model_params_jm4.update({'Ex': field_val_jm4, 'Ey': field_val_jm4, 'Ez': field_val_jm4})

    model_jm4 = KitaevGammaChain(model_params_jm4)
    psi_jm4, E_total_jm4 = run_dmrg(model_jm4, dmrg_params)

    results_jm4 = perform_full_analysis(psi_jm4, model_jm4, title="")
    save_task_data(results_jm4, task_name_jm4)

    # 绘图
    os.makedirs("results/figures", exist_ok=True)
    plot_analysis_results(results_jm4, phi_val=phi, save_prefix=f"results/figures/{task_name_jm4}")
    print("Task completed successfully.")


if __name__ == "__main__":
    main()