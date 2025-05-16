import numpy as np
from scipy.optimize import root_scalar

cp = 4180  # J/kg·K

def effectiveness_counterflow(NTU, C_r):
    if C_r != 1:
        return (1 - np.exp(-NTU * (1 - C_r))) / (1 - C_r * np.exp(-NTU * (1 - C_r)))
    else:
        return NTU / (1 + NTU)

# 1) Estimate approach temperature
def estimate_approach_temp(m_dot_h, T_h_in, m_dot_c, T_c_in, NTU, cp=4180):
    C_h = m_dot_h * cp
    C_c = m_dot_c * cp
    C_min = min(C_h, C_c)
    C_max = max(C_h, C_c)
    C_r = C_min / C_max

    eff = effectiveness_counterflow(NTU, C_r)
    q = eff * C_min * (T_h_in - T_c_in)
    T_h_out = T_h_in - q / C_h
    approach_temp = T_h_out - T_c_in
    return approach_temp, eff, q

# 2) Estimate required m_dot_h for desired approach
def estimate_required_sample_flow(T_h_in, m_dot_c, T_c_in, approach_temp_target, NTU, cp=4180):
    def residual(m_dot_h):
        if m_dot_h <= 0:
            return 1e6  # avoid zero or negative mass flow
        approach, _, _ = estimate_approach_temp(m_dot_h, T_h_in, m_dot_c, T_c_in, NTU, cp)
        return approach - approach_temp_target

    sol = root_scalar(residual, bracket=[0.001, 1.0], method='brentq')
    return sol.root

# Case 1: Estimate approach temperature
# approach, eff, q = estimate_approach_temp(
#     m_dot_h=0.1, 
#     T_h_in=200, 
#     m_dot_c=10.5, 
#     T_c_in=35, 
#     NTU=3.2
# )
# print(f"Approach temp = {approach:.2f} °C")

# Case 2: Required sample flow for desired approach temp
required_flow = estimate_required_sample_flow(
    T_h_in=204, 
    m_dot_c=.75, 
    T_c_in=35, 
    approach_temp_target=5, 
    NTU=3.57
)
print(f"Required sample flow = {required_flow:.4f} kg/s")

