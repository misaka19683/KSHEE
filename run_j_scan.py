import numpy as np
import os
import sys
import pandas as pd
import subprocess
import argparse
import matplotlib

# =================================================================
# 1. 环境配置与线程设置
# =================================================================

# 默认路径配置
DEFAULT_RESULTS_DIR = "results/j_scan_data"
DEFAULT_LOGS_DIR = "logs/j_scan"

def set_threads(n_threads):
    t_str = str(n_threads)
    envs = ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", 
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]
    for env in envs:
        os.environ[env] = t_str
    os.environ["MKL_DYNAMIC"] = "FALSE"

# 自动设置线程并启用 Agg 后端
n_cores = 64 if "--task" in sys.argv else (os.cpu_count() or 1)
set_threads(n_cores)
matplotlib.use('Agg')

from src.models.kitaev_gamma import KitaevGammaChain
from src.simulation.dmrg_runner import run_dmrg
from src.physics.measurements import perform_full_analysis
from src.utils.plotting import plot_analysis_results

# =================================================================
# 2. 物理参数与任务列表定义
# =================================================================

theta = np.pi * 0.44
phi = np.pi * 0.85
L = 96 

base_params = {
    'L': L,
    'bc_MPS': 'finite',
    'lattice_params': {'bc': 'open'},
    'gamma': np.sin(theta) * np.sin(phi), 
    'k': np.sin(theta) * np.cos(phi), 
    'j': np.cos(theta),
    'gamma_prime': 0.1, 
    'k2': 0.02, 
    'j2': 0.2,
    'lambda_in_dp': -0.1, 
    'lambda_in_dn': -0.05,
    'lambda_out': 0.04, 
    'lambda_gamma_out': 0.05,
}

dmrg_params = {
    'mixer': True,
    'max_E_err': 1.e-10,
    'trunc_params': {'chi_max': 1000, 'svd_min': 1.e-10},
}

# 生成 j 从 -10 到 10 的任务
tasks = []
j_values = range(-10, 11)
already_done = [-1, 0, 1, 2, 3, 4, 5]

for j in j_values:
    j_str = str(j).replace('-', 'm')
    field_val = j / np.sqrt(3)
    tasks.append({
        "name": f"task_j_{j_str}",
        "Ex": field_val,
        "Ey": field_val,
        "Ez": field_val,
        "j_val": j,
        "is_new": j not in already_done
    })

# =================================================================
# 3. 核心执行逻辑
# =================================================================

def save_task_data(results, task_name, task_dir):
    summary_path = os.path.join(task_dir, f"{task_name}_summary.csv")
    # OBC 保存 kappa 和 central charge
    data = {
        'task': [task_name],
        'central_charge': [results['cc_res']['c_avg']],
        'kappa': [results.get('kappa')]
    }
    pd.DataFrame(data).to_csv(summary_path, index=False)
    
    # 保存键能量数据
    pd.DataFrame({'E_bonds': results['E_bonds']}).to_csv(
        os.path.join(task_dir, f"{task_name}_energy.csv"), index=False
    )
    
    print(f"Results for {task_name} saved to {task_dir}.")

def run_single_task(task, results_dir=DEFAULT_RESULTS_DIR):
    task_name = task["name"]
    task_dir = os.path.join(results_dir, task_name)
    os.makedirs(task_dir, exist_ok=True)
    
    print(f"[{os.getpid()}] 正在执行任务: {task_name} (j={task['j_val']})")
    
    model_params = base_params.copy()
    model_params.update({'Ex': task['Ex'], 'Ey': task['Ey'], 'Ez': task['Ez']})
    
    model = KitaevGammaChain(model_params)
    psi, info = run_dmrg(model, dmrg_params)
    results = perform_full_analysis(psi, model, title=f"E=[1,1,1]*{task['j_val']}/sqrt(3)")
    
    save_task_data(results, task_name, task_dir)
    save_prefix = os.path.join(task_dir, task_name)
    plot_analysis_results(results, phi_val=phi, save_prefix=save_prefix)
    
    print(f"[{os.getpid()}] 任务 {task_name} 完成。")
    return f"{task_name} OK"

# =================================================================
# 4. SLURM 提交逻辑
# =================================================================

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH -J {job_name}
#SBATCH -p xhhctdnormal
#SBATCH -N 1
#SBATCH -c 64
#SBATCH --mem=200G
#SBATCH -t 96:00:00
#SBATCH -o {log_dir}/%j_{task_name}.out
#SBATCH -e {log_dir}/%j_{task_name}.err

cd {work_dir} || exit   
module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate toolchain_env

uv run {script_name} --task {task_name} --output {results_dir}
"""

def submit_to_slurm(results_dir, log_dir, dry_run=False, only_new=True):
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    
    target_tasks = [t for t in tasks if not only_new or t["is_new"]]
    
    print(f"准备提交 {len(target_tasks)} 个任务至 SLURM... (结果目录: {results_dir}, 日志目录: {log_dir})")
    
    work_dir = os.getcwd()
    script_name = os.path.basename(__file__)
    
    for i, task in enumerate(target_tasks):
        task_name = task["name"]
        script_content = SLURM_TEMPLATE.format(
            job_name=f"j_{task['j_val']}", 
            task_name=task_name,
            log_dir=log_dir,
            results_dir=results_dir,
            work_dir=work_dir,
            script_name=script_name
        )
        if dry_run:
            print(f"[{i+1}/{len(target_tasks)}] [Dry Run] 准备提交: {task_name} (j={task['j_val']})")
            continue
            
        process = subprocess.Popen(['sbatch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(input=script_content)
        
        if process.returncode == 0:
            print(f"[{i+1}/{len(target_tasks)}] 成功提交: {task_name} (Job ID: {stdout.strip().split()[-1]})")
        else:
            print(f"[{i+1}/{len(target_tasks)}] 提交失败: {task_name}\n原因: {stderr.strip()}")

# =================================================================
# 5. 主程序入口
# =================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kitaev-Gamma j-scan Task Runner")
    parser.add_argument("--submit", action="store_true", help="提交任务到 SLURM 队列")
    parser.add_argument("--all", action="store_true", help="与 --submit 配合使用，提交所有任务（包括已计算过的）")
    parser.add_argument("--dry-run", action="store_true", help="与 --submit 配合使用，仅查看不提交")
    parser.add_argument("--task", type=str, help="运行指定的单个任务")
    parser.add_argument("--output", "-o", type=str, default=DEFAULT_RESULTS_DIR, help=f"结果输出根目录")
    parser.add_argument("--logs", type=str, default=DEFAULT_LOGS_DIR, help=f"日志输出目录")
    
    args = parser.parse_args()

    if args.submit:
        submit_to_slurm(results_dir=args.output, log_dir=args.logs, dry_run=args.dry_run, only_new=not args.all)
    elif args.task:
        target_task = next((t for t in tasks if t["name"] == args.task), None)
        if target_task:
            run_single_task(target_task, results_dir=args.output)
        else:
            print(f"错误: 未找到任务 '{args.task}'")
            sys.exit(1)
    else:
        # 默认：本地顺序执行新任务
        target_tasks = [t for t in tasks if t["is_new"]]
        print(f"开始顺序执行 {len(target_tasks)} 个新任务... (输出目录: {args.output})")
        for task in target_tasks:
            run_single_task(task, results_dir=args.output)
        print("\n所有选定任务已完成。")
