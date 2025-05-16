
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.optimize import root_scalar
import json

cp = 4180  # J/kg·K


# Helper to convert lb/hr to kg/s
def convert_lbhr_to_kgs(lb_hr):
    return lb_hr * 0.000125998


def load_digitized_data(data):
    if isinstance(data, str):  # File path
        df = pd.read_csv(data)
    else:  # Already a DataFrame
        df = data.copy()

    if "flow_lb_hr" in df.columns:
        df["flow_kg_s"] = df["flow_lb_hr"].apply(lambda lbhr: lbhr * 0.000125998)

    return df


def transform_wpd_csv_clean(csv_path):
    # Load entire CSV
    df_full = pd.read_csv(csv_path, header=None)

    # Step 1: Extract T_h_in_C from row 0 (first row)
    T_h_list = df_full.iloc[0, ::2].astype(float).tolist()  # Every other column (X columns only)

    # Step 2: Drop the first two rows (headers: T_h and then 'X/Y')
    df_data = df_full.drop([0, 1]).reset_index(drop=True)

    # Step 3: Loop through every X/Y pair
    long_data = []
    for i, T_h in enumerate(T_h_list):
        flow_col = i * 2
        appr_col = flow_col + 1
        for j in range(len(df_data)):
            flow = df_data.iloc[j, flow_col]
            appr = df_data.iloc[j, appr_col]
            if pd.notna(flow) and pd.notna(appr):
                long_data.append({
                    "flow_lb_hr": float(flow),
                    "T_h_in_C": T_h,
                    "approach_C": float(appr)
                })

    return pd.DataFrame(long_data)


def effectiveness(NTU, Cr):
    if Cr == 1.0:
        return NTU / (1 + NTU)
    else:
        return (1 - np.exp(-NTU * (1 - Cr))) / (1 - Cr * np.exp(-NTU * (1 - Cr)))
    

def backsolve_ntu(df, T_c_in, m_dot_c):
    results = []
    for _, row in df.iterrows():
        m_dot_h = row['flow_kg_s']
        T_h_in = row['T_h_in_C']
        approach = row['approach_C']
        T_h_out = T_c_in + approach
        C_h = m_dot_h * cp
        C_c = m_dot_c * cp
        C_min = min(C_h, C_c)
        C_max = max(C_h, C_c)
        C_r = C_min / C_max
        q = C_h * (T_h_in - T_h_out)
        epsilon = q / (C_min * (T_h_in - T_c_in))

        def residual(NTU):
            return effectiveness(NTU, C_r) - epsilon

        try:
            sol = root_scalar(residual, bracket=[0.001, 10], method='brentq')
            NTU = sol.root
        except ValueError:
            NTU = None
        results.append({'m_dot_h': m_dot_h, 'T_h_in': T_h_in, 'NTU': NTU})
    return pd.DataFrame(results)

def fit_ntu_surface_model(df):
    X = df[["m_dot_h", "T_h_in"]].values
    y = df["NTU"].values
    model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression())
    model.fit(X, y)
    return model

def export_model_to_json(model, path):
    poly = model.named_steps["polynomialfeatures"]
    lin_reg = model.named_steps["linearregression"]
    model_data = {
        "degree": poly.degree,
        "include_bias": poly.include_bias,
        "feature_names": poly.get_feature_names_out(["m_dot_h", "T_h_in"]).tolist(),
        "coefficients": lin_reg.coef_.tolist(),
        "intercept": lin_reg.intercept_
    }
    with open(path, 'w') as f:
        json.dump(model_data, f, indent=4)

def load_model_from_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def evaluate_ntu_from_json(m_dot_h, T_h_in, model_data):
    terms = {
        "m_dot_h": m_dot_h,
        "T_h_in": T_h_in,
        "m_dot_h^2": m_dot_h**2,
        "m_dot_h T_h_in": m_dot_h * T_h_in,
        "T_h_in^2": T_h_in**2
    }
    NTU = model_data["intercept"]
    for name, coef in zip(model_data["feature_names"], model_data["coefficients"]):
        NTU += coef * terms[name]
    return NTU

def predict_approach(m_dot_h, T_h_in, T_c_in, m_dot_c, model_data):
    C_h = m_dot_h * cp
    C_c = m_dot_c * cp
    C_min = min(C_h, C_c)
    C_max = max(C_h, C_c)
    C_r = C_min / C_max
    NTU = evaluate_ntu_from_json(m_dot_h, T_h_in, model_data)
    eps = effectiveness(NTU, C_r)
    q = eps * C_min * (T_h_in - T_c_in)
    T_h_out = T_h_in - q / C_h
    return T_h_out - T_c_in

def solve_flow_for_approach(target_approach, T_h_in, T_c_in, m_dot_c, model_data):
    def residual(flow):
        return predict_approach(flow, T_h_in, T_c_in, m_dot_c, model_data) - target_approach
    sol = root_scalar(residual, bracket=[0.001, 0.5], method='brentq')
    return sol.root if sol.converged else None



df_long = transform_wpd_csv_clean(r"C:\Users\i04604393\Downloads\wpd_datasets.csv")
df_ready = load_digitized_data(df_long)

print(df_long.tail())
print(df_ready.head())