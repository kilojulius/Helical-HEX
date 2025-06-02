from CoolerClass import SampleCooler, CoolerLibrary
from CoolerClass import NTUModelBuilder
import json


# # Path to the pressure drop CSV
# pressure_drop_csv = r"C:\Users\i04604393\Downloads\wpd_tsr_4225_pdrop.csv"

# # Fit the pressure drop model
# pressure_drop_model = NTUModelBuilder.fit_pressure_drop_model(pressure_drop_csv)

# # Display the model and R² value
# print("Pressure Drop Model:", pressure_drop_model)
# print(f"R² Value: {pressure_drop_model['r2']:.4f}")

# # Example usage of NTUModelBuilder
# builder = NTUModelBuilder()

# cooler_model_name = "TSR-4225"
# builder.set_cooler_properties(cooler_model=cooler_model_name, area=0.11,sample_pressure_drop_model=pressure_drop_model, cooling_water_pressure_drop=6985*3)

# # Step 1: Transform raw CSV data
# csv_path = r"C:\Users\i04604393\Downloads\wpd_tsr_4225.csv"
# df_long = builder.transform_wpd_csv_clean(csv_path)

# # Step 2: Convert flow rates and prepare data
# df_ready = builder.load_digitized_data(df_long)

# # Step 3: Generate NTU model
# model, model_data, ntu_df = builder.generate_ntu_model(df_ready, T_c_in=35.0, m_dot_c=0.2332)

# # Step 4: Save the model to JSON
# builder.save_model_to_json(model_data, f"ntu_model_{cooler_model_name.replace('-','').lower()}.json")

# # Step 5: Calculate and display the average NTU
# average_ntu = ntu_df["NTU"].mean()
# print(f"Average NTU for the new cooler model: {average_ntu:.2f}")

# # Step 1: Load model data from JSON
# json_path = r"C:\Users\i04604393\OneDrive - Endress+Hauser\Documents\Helical HEX\ntu_model_trw4222.json"
# with open(json_path, "r") as f:
#     model_data = json.load(f)

# # Step 2: Define cooler properties
# name = "TRW-4222"
# area_m2 = 0.11          # Known area for this cooler
# # U_W_per_m2K = 520       # Assumed U value or back-solved
# cp = 4180               # Water default

# # Step 3: Create SampleCooler instance
# cooler = SampleCooler(name, area_m2, U_W_per_m2K=None, model_data=model_data, cp=cp)

# U_est = cooler.estimate_u(m_dot_h=0.014, m_dot_c=0.19, T_h_in=149)
# print(f"Estimated U: {U_est:.1f} W/m²·K")

# # Step 4: Use methods on the cooler
# m_dot_h = 0.022         # Sample flow rate in kg/s
# T_h_in = 149            # Sample inlet temp (°C) 
# T_c_in = 35             # Cooling water inlet temp (°C)
# m_dot_c = 0.19         # Cooling water flow rate in kg/s

# # Evaluate NTU
# ntu = cooler.evaluate_ntu(m_dot_h, T_h_in)
# print(f"NTU: {ntu:.2f}")

# # Predict approach temperature
# approach = cooler.predict_approach(m_dot_h, T_h_in, T_c_in, m_dot_c)
# print(f"Predicted approach temperature: {approach:.2f} °C")

# # Optionally solve for required flowrate for a 5°C approach
# required_flow = cooler.solve_flowrate("approach", T_h_in=T_h_in, T_c_in=T_c_in, m_dot_c=m_dot_c, target=5.0)
# print(f"Required flowrate for 5°C approach: {required_flow:.4f} kg/s")

# # Optionally solve for required flowrate for a specific Tc outlet temperature
# required_flow = cooler.solve_flowrate("tc_out", T_h_in=T_h_in, T_c_in=T_c_in, m_dot_c=m_dot_c, target=47)
# print(f"Required flowrate: {required_flow:.4f} kg/s")



coolers = CoolerLibrary(r"C:\Users\i04604393\OneDrive - Endress+Hauser\Documents\Helical HEX\helical-sample-cooler\JSON Library")
coolers_results = coolers.find_best_by_flowrate(mode="approach",design_inputs= {"T_h_in":302, "T_c_in":35, "m_dot_c":0.19, "target":5})
print(coolers_results)