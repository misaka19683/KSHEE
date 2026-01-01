from tenpy.models.model import CouplingMPOModel
from tenpy.networks.site import SpinHalfSite
from tenpy.models.lattice import Ladder

class KitaevGammaChain(CouplingMPOModel):
    """
    Kitaev-Gamma chain model with electric field effects and NNN interactions.
    """
    def init_lattice(self, model_params):
        site = SpinHalfSite(conserve=None)
        total_sites = model_params.get('L', 96)
        L_ladder = total_sites // 2

        lat_params = model_params.get('lattice_params', {})
        bc = lat_params.get('bc', 'open')
        bc_MPS = model_params.get('bc_MPS', 'finite')

        lat = Ladder(L=L_ladder, sites=[site, site], bc=bc, bc_MPS=bc_MPS)

        # 下面是使用 Lattice 的定义格子的方式
        # from tenpy.models.lattice import Lattice
        # lat = Lattice(
        #     Ls=[total_sites],
        #     unit_cell=[site, site],              # ✅ 两个 site 的 unit cell
        #     basis=np.array([[2.0]]),               # unit cell 的平移向量长度=2
        #     positions=np.array([[0.0], [1.0]]),    # unit cell 内两点位置：0 和 1
        #     pairs={
        #         "nearest_neighbors": [
        #             (0, 1, np.array([0])),         # A(n) - B(n)
        #             (1, 0, np.array([1])),         # B(n) - A(n+1)
        #         ]
        #     },
        #     bc=["open"],   # 或 ["periodic"]
        #     bc_MPS="finite"
        # )
        return lat

    def init_terms(self, model_params):
        # --- 1. Basic parameters ---
        j = model_params.get('j', 0.)
        k = model_params.get('k', 0.)
        gamma = model_params.get('gamma', 0.)
        gamma_prime = model_params.get('gamma_prime', 0.)
        k2 = model_params.get('k2', 0.)
        j2 = model_params.get('j2', 0.)

        # --- 2. Electric field parameters ---
        Ex = model_params.get('Ex', 0.)
        Ey = model_params.get('Ey', 0.)
        Ez = model_params.get('Ez', 0.)

        lambda_in_dp = model_params.get('lambda_in_dp', 0.)
        lambda_in_dn = model_params.get('lambda_in_dn', 0.)
        lambda_out = model_params.get('lambda_out', 0.)
        lambda_gamma_out = model_params.get('lambda_gamma_out', 0.)

        # Precompute combination parameters
        L_in_plus = lambda_in_dp + lambda_in_dn
        L_in_minus = lambda_in_dp - lambda_in_dn

        # =======================================================
        # Part A: Magnetic Interactions
        # =======================================================

        # --- Bond Type 1: (Site 1 -> Site 2, u=0 -> u=1, dx=0) ---
        # K S^x S^x
        self.add_coupling(k, 0, 'Sx', 1, 'Sx', 0)
        # J (S \cdot S)
        for op in ['Sx', 'Sy', 'Sz']:
            self.add_coupling(j, 0, op, 1, op, 0)
        # Gamma (S^y S^z + S^z S^y)
        self.add_coupling(gamma, 0, 'Sy', 1, 'Sz', 0)
        self.add_coupling(gamma, 0, 'Sz', 1, 'Sy', 0)
        # Gamma' (S^x S^y + S^y S^x + S^x S^z + S^z S^x)
        for o1, o2 in [('Sx', 'Sy'), ('Sy', 'Sx'), ('Sx', 'Sz'), ('Sz', 'Sx')]:
            self.add_coupling(gamma_prime, 0, o1, 1, o2, 0)

        # --- Bond Type 2: (Site 2 -> Site 3, u=1 -> u=0, dx=1) ---
        # K S^y S^y
        self.add_coupling(k, 1, 'Sy', 0, 'Sy', 1)
        # J (S \cdot S)
        for op in ['Sx', 'Sy', 'Sz']:
            self.add_coupling(j, 1, op, 0, op, 1)
        # Gamma (S^z S^x + S^x S^z)
        self.add_coupling(gamma, 1, 'Sz', 0, 'Sx', 1)
        self.add_coupling(gamma, 1, 'Sx', 0, 'Sz', 1)
        # Gamma' (S^y S^z + S^z S^y + S^y S^x + S^x S^y)
        for o1, o2 in [('Sy', 'Sz'), ('Sz', 'Sy'), ('Sx', 'Sy'), ('Sy', 'Sx')]:
            self.add_coupling(gamma_prime, 1, o1, 0, o2, 1)

        # --- Next-Nearest Neighbor (u -> u, dx=1) ---
        for u in [0, 1]:
            # K2 S^z S^z
            self.add_coupling(k2, u, 'Sz', u, 'Sz', 1)
            # J2 (S \cdot S)
            for op in ['Sx', 'Sy', 'Sz']:
                self.add_coupling(j2, u, op, u, op, 1)

        # =======================================================
        # Part B: Electric Field Induced Terms
        # =======================================================

        # --- 1. Ex field contribution ---
        if abs(Ex) > 1e-12:
            # Bond 1
            # lambda_out * (S^z S^x - S^x S^z + S^x S^y - S^y S^x)
            self.add_coupling(Ex * lambda_out, 0, 'Sz', 1, 'Sx', 0)
            self.add_coupling(-Ex * lambda_out, 0, 'Sx', 1, 'Sz', 0)
            self.add_coupling(Ex * lambda_out, 0, 'Sx', 1, 'Sy', 0)
            self.add_coupling(-Ex * lambda_out, 0, 'Sy', 1, 'Sx', 0)
            # lambda_gamma_out * (S^x S^y + S^y S^x + S^x S^z + S^z S^x)
            for o1, o2 in [('Sx', 'Sy'), ('Sy', 'Sx'), ('Sx', 'Sz'), ('Sz', 'Sx')]:
                self.add_coupling(Ex * lambda_gamma_out, 0, o1, 1, o2, 0)

            # Bond 2
            self.add_coupling(Ex * L_in_minus, 1, 'Sz', 0, 'Sx', 1)
            self.add_coupling(-Ex * L_in_minus, 1, 'Sx', 0, 'Sz', 1)

        # --- 2. Ey field contribution ---
        if abs(Ey) > 1e-12:
            # Bond 1
            # (D+ + D-) * (S^y S^z - S^z S^y)
            self.add_coupling(Ey * L_in_plus, 0, 'Sy', 1, 'Sz', 0)
            self.add_coupling(-Ey * L_in_plus, 0, 'Sz', 1, 'Sy', 0)
            # Bond 2
            # lambda_out * (S^x S^y - S^y S^x + S^y S^z - S^z S^y)
            self.add_coupling(Ey * lambda_out, 1, 'Sx', 0, 'Sy', 1)
            self.add_coupling(-Ey * lambda_out, 1, 'Sy', 0, 'Sx', 1)
            self.add_coupling(Ey * lambda_out, 1, 'Sy', 0, 'Sz', 1)
            self.add_coupling(-Ey * lambda_out, 1, 'Sz', 0, 'Sy', 1)
            # lambda_gamma_out * (S^y S^z + S^z S^y + S^y S^x + S^x S^y)
            for o1, o2 in [('Sy', 'Sz'), ('Sz', 'Sy'), ('Sy', 'Sx'), ('Sx', 'Sy')]:
                self.add_coupling(Ey * lambda_gamma_out, 1, o1, 0, o2, 1)

        # --- 3. Ez field contribution ---
        if abs(Ez) > 1e-12:
            # Bond 1 (u=0 -> u=1): (D+ - D-) * (S^y S^z - S^z S^y)
            self.add_coupling(Ez * L_in_minus, 0, 'Sy', 1, 'Sz', 0)
            self.add_coupling(-Ez * L_in_minus, 0, 'Sz', 1, 'Sy', 0)
            # Bond 2 (u=1 -> u=0): (D+ + D-) * (S^z S^x - S^x S^z)
            self.add_coupling(Ez * L_in_plus, 1, 'Sz', 0, 'Sx', 1)
            self.add_coupling(-Ez * L_in_plus, 1, 'Sx', 0, 'Sz', 1)

# # another way to class

# class KitaevGammaChainBase(CouplingMPOModel):
#     """
#     Kitaev-Gamma Chain 基类.
#     仅包含最近邻 (Nearest-Neighbor) 的 K, J, Gamma, Gamma' 相互作用。
#     不包含次近邻 (NNN) 和电场项。
#     """
#     def init_lattice(self, model_params):
#         site = SpinHalfSite(conserve=None)
#         total_sites = model_params.get('L', 96)
#         L_ladder = total_sites // 2
#         lat_params = model_params.get('lattice_params', {})
#         bc = lat_params.get('bc', 'open')
#         bc_MPS = model_params.get('bc_MPS', 'finite')
#         lat = Ladder(L=L_ladder, sites=[site, site], bc=bc, bc_MPS=bc_MPS)
#         return lat
#
#     def init_terms(self, model_params):
#         # --- 读取基础参数 ---
#         j = model_params.get('j', 0.)
#         k = model_params.get('k', 0.)
#         gamma = model_params.get('gamma', 0.)
#         gamma_prime = model_params.get('gamma_prime', 0.)
#
#         # =======================================================
#         # Part A: 原有磁性相互作用 (仅 NN)
#         # =======================================================
#
#         # --- Bond Type 1: (Site 1 -> Site 2, u=0 -> u=1, dx=0) ---
#         # K S^x S^x
#         self.add_coupling(k, 0, 'Sx', 1, 'Sx', 0)
#         # J (S \cdot S)
#         for op in ['Sx', 'Sy', 'Sz']:
#             self.add_coupling(j, 0, op, 1, op, 0)
#         # Gamma (S^y S^z + S^z S^y)
#         self.add_coupling(gamma, 0, 'Sy', 1, 'Sz', 0)
#         self.add_coupling(gamma, 0, 'Sz', 1, 'Sy', 0)
#         # Gamma'
#         self.add_coupling(gamma_prime, 0, 'Sx', 1, 'Sy', 0)
#         self.add_coupling(gamma_prime, 0, 'Sy', 1, 'Sx', 0)
#         self.add_coupling(gamma_prime, 0, 'Sx', 1, 'Sz', 0)
#         self.add_coupling(gamma_prime, 0, 'Sz', 1, 'Sx', 0)
#
#         # --- Bond Type 2: (Site 2 -> Site 3, u=1 -> u=0, dx=1) ---
#         # K S^y S^y
#         self.add_coupling(k, 1, 'Sy', 0, 'Sy', 1)
#         # J (S \cdot S)
#         for op in ['Sx', 'Sy', 'Sz']:
#             self.add_coupling(j, 1, op, 0, op, 1)
#         # Gamma (S^z S^x + S^x S^z)
#         self.add_coupling(gamma, 1, 'Sz', 0, 'Sx', 1)
#         self.add_coupling(gamma, 1, 'Sx', 0, 'Sz', 1)
#         # Gamma'
#         self.add_coupling(gamma_prime, 1, 'Sy', 0, 'Sz', 1)
#         self.add_coupling(gamma_prime, 1, 'Sz', 0, 'Sy', 1)
#         self.add_coupling(gamma_prime, 1, 'Sx', 0, 'Sy', 1)
#         self.add_coupling(gamma_prime, 1, 'Sy', 0, 'Sx', 1)
#
#
# def add_nnn_terms(model, model_params):
#     """
#     向模型中添加次近邻 (NNN) 相互作用项 (K2, J2)
#     """
#     k2 = model_params.get('k2', 0.)
#     j2 = model_params.get('j2', 0.)
#
#     # --- Next-Nearest Neighbor (u -> u, dx=1) ---
#     for u in [0, 1]:
#         # K2 S^z S^z
#         model.add_coupling(k2, u, 'Sz', u, 'Sz', 1)
#         # J2 (S \cdot S)
#         for op in ['Sx', 'Sy', 'Sz']:
#             model.add_coupling(j2, u, op, u, op, 1)
#
#
# def add_electric_field_terms(model, model_params):
#     """
#     向模型中添加电场诱导项 (Ex, Ey, Ez)
#     """
#     # --- 读取电场相关参数 ---
#     Ex = model_params.get('Ex', 0.)
#     Ey = model_params.get('Ey', 0.)
#     Ez = model_params.get('Ez', 0.)
#
#     lambda_in_dp = model_params.get('lambda_in_dp', 0.)
#     lambda_in_dn = model_params.get('lambda_in_dn', 0.)
#     lambda_out = model_params.get('lambda_out', 0.)
#     lambda_gamma_out = model_params.get('lambda_gamma_out', 0.)
#
#     # 预计算组合参数 (D+ +/- D-)
#     L_in_plus = lambda_in_dp + lambda_in_dn
#     L_in_minus = lambda_in_dp - lambda_in_dn
#
#     # --- 1. Ex 场贡献 ---
#     if abs(Ex) > 1e-12:
#         # Bond 1 (u=0 -> u=1): lambda_out & gamma_out
#         # lambda_out * (S^z S^x - S^x S^z + S^x S^y - S^y S^x)
#         model.add_coupling(Ex * lambda_out, 0, 'Sz', 1, 'Sx', 0)
#         model.add_coupling(-Ex * lambda_out, 0, 'Sx', 1, 'Sz', 0)
#         model.add_coupling(Ex * lambda_out, 0, 'Sx', 1, 'Sy', 0)
#         model.add_coupling(-Ex * lambda_out, 0, 'Sy', 1, 'Sx', 0)
#         # lambda_gamma_out * (S^x S^y + S^y S^x + S^x S^z + S^z S^x)
#         model.add_coupling(Ex * lambda_gamma_out, 0, 'Sx', 1, 'Sy', 0)
#         model.add_coupling(Ex * lambda_gamma_out, 0, 'Sy', 1, 'Sx', 0)
#         model.add_coupling(Ex * lambda_gamma_out, 0, 'Sx', 1, 'Sz', 0)
#         model.add_coupling(Ex * lambda_gamma_out, 0, 'Sz', 1, 'Sx', 0)
#
#         # Bond 2 (u=1 -> u=0): (D+ - D-) * (S^z S^x - S^x S^z)
#         term = Ex * L_in_minus
#         model.add_coupling(term, 1, 'Sz', 0, 'Sx', 1)
#         model.add_coupling(-term, 1, 'Sx', 0, 'Sz', 1)
#
#     # --- 2. Ey 场贡献 ---
#     if abs(Ey) > 1e-12:
#         # Bond 1 (u=0 -> u=1): (D+ + D-) * (S^y S^z - S^z S^y)
#         term = Ey * L_in_plus
#         model.add_coupling(term, 0, 'Sy', 1, 'Sz', 0)
#         model.add_coupling(-term, 0, 'Sz', 1, 'Sy', 0)
#
#         # Bond 2 (u=1 -> u=0): lambda_out & gamma_out
#         # lambda_out * (S^x S^y - S^y S^x + S^y S^z - S^z S^y)
#         model.add_coupling(Ey * lambda_out, 1, 'Sx', 0, 'Sy', 1)
#         model.add_coupling(-Ey * lambda_out, 1, 'Sy', 0, 'Sx', 1)
#         model.add_coupling(Ey * lambda_out, 1, 'Sy', 0, 'Sz', 1)
#         model.add_coupling(-Ey * lambda_out, 1, 'Sz', 0, 'Sy', 1)
#         # lambda_gamma_out * (S^y S^z + S^z S^y + S^y S^x + S^x S^y)
#         model.add_coupling(Ey * lambda_gamma_out, 1, 'Sy', 0, 'Sz', 1)
#         model.add_coupling(Ey * lambda_gamma_out, 1, 'Sz', 0, 'Sy', 1)
#         model.add_coupling(Ey * lambda_gamma_out, 1, 'Sy', 0, 'Sx', 1)
#         model.add_coupling(Ey * lambda_gamma_out, 1, 'Sx', 0, 'Sy', 1)
#
#     # --- 3. Ez 场贡献 ---
#     if abs(Ez) > 1e-12:
#         # Bond 1 (u=0 -> u=1): (D+ - D-) * (S^y S^z - S^z S^y)
#         term = Ez * L_in_minus
#         model.add_coupling(term, 0, 'Sy', 1, 'Sz', 0)
#         model.add_coupling(-term, 0, 'Sz', 1, 'Sy', 0)
#
#         # Bond 2 (u=1 -> u=0): (D+ + D-) * (S^z S^x - S^x S^z)
#         term = Ez * L_in_plus
#         model.add_coupling(term, 1, 'Sz', 0, 'Sx', 1)
#         model.add_coupling(-term, 1, 'Sx', 0, 'Sz', 1)
#
# def main():
#     theta = np.pi * 0.44
#     phi = np.pi * 0.85
#     model_params_inner = {
#         'L': 24,
#         'bc_MPS': 'finite',
#         'lattice_params': {
#             'bc': 'open',
#         },
#         'gamma': np.sin(theta) * np.sin(phi),
#         'k': np.sin(theta) * np.cos(phi),
#         'j': np.cos(theta),
#         'gamma_prime': 0.,
#         'k2': 0.,
#         'j2': 0.,
#
#         # 电场参数
#         'lambda_in_dp': 0.,
#         'lambda_in_dn': 0.,
#         'lambda_out': 0.,
#         'lambda_gamma_out': 0.,
#
#         # 电场分量
#         'Ex': 0.0,
#         'Ey': 0.0,
#         'Ez': 0.0,
#     }
#     model_inner = KitaevGammaChainBase(model_params_inner)
#     print("Running DMRG...")
#     dmrg_params_inner = {
#         'mixer': True,
#         'max_E_err': 1.e-10,
#         'trunc_params': {'chi_max': 200, 'svd_min': 1.e-10},
#     }
#
#     # 构造初始态 (Neel 态)
#     product_state_inner = ["up", "down"] * (model_inner.lat.N_sites // 2)
#     from tenpy import MPS
#     from tenpy.algorithms import dmrg
#     psi_inner = MPS.from_product_state(
#         model_inner.lat.mps_sites(),
#         product_state_inner,
#         bc=model_inner.lat.bc_MPS,
#         unit_cell_width=len(model_inner.lat.unit_cell)
#     )
#
#     # 执行 DMRG
#     info_inner = dmrg.run(psi_inner, model_inner, dmrg_params_inner)
#     E_total_dmrg_inner = info_inner['E']
#     print(f"Final DMRG Energy: {E_total_dmrg_inner:.8f}")
#
#     from tenpy.models.model import NearestNeighborModel
#     nn_model=NearestNeighborModel.from_MPOModel(model_inner)
#     bond_energies=nn_model.bond_energies(psi_inner)
#     print(bond_energies)
#
#     return bond_energies,model_inner

