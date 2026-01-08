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