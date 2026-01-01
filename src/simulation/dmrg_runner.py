from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS

def run_dmrg(model, dmrg_params=None, initial_state=None):
    """
    Run DMRG for a given model and parameters.
    """
    if dmrg_params is None:
        dmrg_params = {
            'mixer': True,
            'max_E_err': 1.e-10,
            'trunc_params': {'chi_max': 1000, 'svd_min': 1.e-10},
        }

    if initial_state is None:
        # Default to Neel state
        product_state = ["up", "down"] * (model.lat.N_sites // 2)
        initial_state = MPS.from_product_state(
            model.lat.mps_sites(),
            product_state,
            bc=model.lat.bc_MPS,
            unit_cell_width=len(model.lat.unit_cell)
        )

    info = dmrg.run(initial_state, model, dmrg_params)
    return initial_state, info
