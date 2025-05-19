import json
import numpy as np
from scipy.optimize import minimize_scalar, root_scalar

class SampleCooler:
    """
    Represents a single sample cooler with NTU model and heat exchanger properties.

    Attributes:
    - name (str): Identifier for the cooler
    - area (float): Heat transfer area in m²
    - U (float): Overall heat transfer coefficient in W/m²·K
    - model_data (dict): Polynomial regression model coefficients for NTU
    - cp (float): Specific heat capacity of the fluid, default 4180 J/kg·K
    """

    def __init__(self, name, area_m2, U_W_per_m2K, model_data, cp=4180):
        self.name = name
        self.area = area_m2
        self.U = U_W_per_m2K
        self.cp = cp
        self.model_data = model_data

    def evaluate_ntu(self, m_dot_h, T_h_in):
        """Evaluates NTU using polynomial surface model given hot flow and inlet temp."""
        terms = {
            "m_dot_h": m_dot_h,
            "T_h_in": T_h_in,
            "m_dot_h^2": m_dot_h**2,
            "m_dot_h T_h_in": m_dot_h * T_h_in,
            "T_h_in^2": T_h_in**2
        }
        NTU = self.model_data["intercept"]
        for name, coef in zip(self.model_data["feature_names"], self.model_data["coefficients"]):
            NTU += coef * terms[name]
        return NTU

    @staticmethod
    def effectiveness(NTU, Cr):
        """Computes effectiveness for a counterflow heat exchanger given NTU and Cr."""
        if Cr == 1.0:
            return NTU / (1 + NTU)
        return (1 - np.exp(-NTU * (1 - Cr))) / (1 - Cr * np.exp(-NTU * (1 - Cr)))

    def predict_approach(self, m_dot_h, T_h_in, T_c_in, m_dot_c):
        """Predicts approach temperature for given operating conditions."""
        C_h = m_dot_h * self.cp
        C_c = m_dot_c * self.cp
        C_min = min(C_h, C_c)
        C_max = max(C_h, C_c)
        Cr = C_min / C_max
        NTU = self.evaluate_ntu(m_dot_h, T_h_in)
        eps = self.effectiveness(NTU, Cr)
        q = eps * C_min * (T_h_in - T_c_in)
        T_h_out = T_h_in - q / C_h
        return T_h_out - T_c_in
    
    def estimate_u(self, m_dot_h, m_dot_c, T_h_in):
        """
        Estimate the effective overall heat transfer coefficient (U) for the cooler
        using backsolved NTU and known heat transfer area.

        Parameters:
        - m_dot_h: Sample flowrate (kg/s)
        - m_dot_c: Cooling water flowrate (kg/s)
        - T_h_in: Sample inlet temperature (°C)

        Returns:
        - Estimated U in W/m²·K
        """
        C_h = m_dot_h * self.cp
        C_c = m_dot_c * self.cp
        C_min = min(C_h, C_c)
        NTU = self.evaluate_ntu(m_dot_h, T_h_in)
        U_est = (NTU * C_min) / self.area
        return U_est


    def solve_flowrate(self, mode, T_h_in, T_c_in, m_dot_c, target):
        """
        Solves for sample flowrate based on a given design target.

        Parameters:
        - mode: 'approach', 'tc_out', or 'heat_duty'
        - T_h_in, T_c_in: Inlet temperatures of hot and cold streams
        - m_dot_c: Cooling water mass flowrate
        - target: Target approach, Tc_out, or heat load

        Returns:
        - m_dot_h (kg/s): Solved hot-side flowrate to meet condition
        """
        cp = self.cp

        def predict_approach(flow):
            return self.predict_approach(flow, T_h_in, T_c_in, m_dot_c)

        def tc_out(flow):
            C_h = flow * cp
            C_c = m_dot_c * cp
            C_min = min(C_h, C_c)
            C_max = max(C_h, C_c)
            Cr = C_min / C_max
            NTU = self.evaluate_ntu(flow, T_h_in)
            eps = self.effectiveness(NTU, Cr)
            q = eps * C_min * (T_h_in - T_c_in)
            return T_c_in + q / C_c

        def q_calc(flow):
            C_h = flow * cp
            C_c = m_dot_c * cp
            C_min = min(C_h, C_c)
            C_max = max(C_h, C_c)
            Cr = C_min / C_max
            NTU = self.evaluate_ntu(flow, T_h_in)
            eps = self.effectiveness(NTU, Cr)
            return eps * C_min * (T_h_in - T_c_in)

        # Choose the appropriate residual function based on the mode
        if mode == "approach":
            residual = lambda f: predict_approach(f) - target
        elif mode == "tc_out":
            residual = lambda f: tc_out(f) - target
        elif mode == "heat_duty":
            residual = lambda f: q_calc(f) - target
        else:
            raise ValueError("Unknown flowrate mode")

        # Try to bracket the root
        flow_samples = np.linspace(0.001, 0.05, 100)
        resids = [residual(f) for f in flow_samples]
        bracket = None
        for i in range(len(resids) - 1):
            if resids[i] * resids[i + 1] < 0:
                bracket = (flow_samples[i], flow_samples[i + 1])
                break

        if bracket:
            try:
                sol = root_scalar(residual, bracket=bracket, method='brentq')
                if sol.converged:
                    return sol.root
            except:
                pass

        # Fallback to minimization if bracketing fails
        sol = minimize_scalar(lambda f: abs(residual(f)), bounds=(0.001, 0.05), method='bounded')
        if sol.success and sol.fun < 0.2:
            return sol.x

        raise ValueError("Failed to solve for flowrate within tolerance.")


class CoolerLibrary:
    """
    Manages a collection of SampleCooler objects loaded from a JSON file.

    Methods:
    - load_all(): Reads JSON and constructs SampleCooler objects
    - find_best_by_flowrate(): Evaluates each cooler and returns sorted matches
    """

    def __init__(self, json_path):
        """Initializes the library from a JSON file of cooler definitions."""
        self.coolers = self.load_all(json_path)

    def load_all(self, path):
        """Loads all coolers from a JSON file into SampleCooler instances."""
        with open(path, 'r') as f:
            data = json.load(f)
        return [SampleCooler(**entry) for entry in data]

    def find_best_by_flowrate(self, mode, design_inputs):
        """
        Finds and sorts coolers based on performance at a design point.

        Parameters:
        - mode: Flowrate solving mode ('approach', 'tc_out', etc.)
        - design_inputs: Dict of thermal inputs (T_h_in, T_c_in, m_dot_c, target)

        Returns:
        - List of tuples: [(cooler_name, solved_flowrate), ...] sorted ascending
        """
        results = []
        for cooler in self.coolers:
            try:
                val = cooler.solve_flowrate(mode, **design_inputs)
                results.append((cooler.name, val))
            except Exception:
                continue
        return sorted(results, key=lambda x: x[1])
