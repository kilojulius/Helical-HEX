from CoolerClass import SampleCooler
import json


# Step 1: Load model data from JSON
json_path = r"C:\Users\juliu\Documents\Python Projects\Helical HEX\Helical-HEX\ntu_model_fsr6225.json"
with open(json_path, "r") as f:
    model_data = json.load(f)

# Step 2: Define cooler properties
name = "FSR-6225"
area_m2 = 0.16          # Known area for this cooler
U_W_per_m2K = 520       # Assumed U value or back-solved
cp = 4180               # Water default

# Step 3: Create SampleCooler instance
cooler = SampleCooler(name, area_m2, U_W_per_m2K=None, model_data=model_data, cp=cp)

U_est = cooler.estimate_u(m_dot_h=0.014, m_dot_c=0.757, T_h_in=204)
print(f"Estimated U: {U_est:.1f} W/m²·K")

# Step 4: Use methods on the cooler
m_dot_h = 0.014         # Sample flow rate in kg/s
T_h_in = 204            # Sample inlet temp (°C)
T_c_in = 35             # Cooling water inlet temp (°C)
m_dot_c = 0.757         # Cooling water flow rate in kg/s

# Evaluate NTU
ntu = cooler.evaluate_ntu(m_dot_h, T_h_in)
print(f"NTU: {ntu:.2f}")

# Predict approach temperature
approach = cooler.predict_approach(m_dot_h, T_h_in, T_c_in, m_dot_c)
print(f"Predicted approach temperature: {approach:.2f} °C")

# Optionally solve for required flowrate for a 5°C approach
required_flow = cooler.solve_flowrate("approach", T_h_in=T_h_in, T_c_in=T_c_in, m_dot_c=m_dot_c, target=5.0)
print(f"Required flowrate for 5°C approach: {required_flow:.4f} kg/s")
