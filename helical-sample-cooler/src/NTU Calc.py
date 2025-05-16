import numpy as np
from scipy.optimize import root_scalar

def estimate_UA_from_chart(m_dot_h, T_h_in, m_dot_c, T_c_in, approach_temp, cp=4180, A=0.16):
    """
    Estimate UA from a known performance point using the NTU-effectiveness method (counterflow).

    Inputs:
    - m_dot_h: hot side mass flow rate (kg/s)
    - T_h_in: inlet temperature of hot side (°C)
    - m_dot_c: cold side mass flow rate (kg/s)
    - T_c_in: inlet temperature of cold side (°C)
    - approach_temp: T_h_out - T_c_in from chart
    - cp: specific heat capacity (default = 4180 J/kg·K)

    Returns:
    - UA in W/K
    - NTU
    - effectiveness
    """
    T_h_out = T_c_in + approach_temp
    C_h = m_dot_h * cp
    C_c = m_dot_c * cp
    C_min = min(C_h, C_c)
    C_max = max(C_h, C_c)
    C_r = C_min / C_max

    delta_T_h = T_h_in - T_h_out
    q = C_h * delta_T_h

    effectiveness = q / (C_min * (T_h_in - T_c_in))

    # Solve for NTU numerically
    def eff_eq(NTU):
        if C_r < 1.0:
            return (1 - np.exp(-NTU * (1 - C_r))) / (1 - C_r * np.exp(-NTU * (1 - C_r))) - effectiveness
        else:
            return NTU / (1 + NTU) - effectiveness

    sol = root_scalar(eff_eq, bracket=[1e-6, 100], method='brentq')
    NTU = sol.root

    UA = NTU * C_min
    U = UA / A
    return UA, NTU, effectiveness, U


UA, NTU, eff, U = estimate_UA_from_chart(
    m_dot_h=0.019148,
    T_h_in=316,
    m_dot_c=0.75,
    T_c_in=35,
    approach_temp=10.5
)

print(f"Estimated UA = {UA:.2f} W/K")
print(NTU)
print(eff)
print(U)
