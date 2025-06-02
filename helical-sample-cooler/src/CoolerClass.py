import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.optimize import root_scalar, minimize_scalar
from sklearn.metrics import r2_score
import os



class NTUModelBuilder:
    """
    A class that builds Number of Transfer Units (NTU) models for heat exchangers from experimental data.
    
    The NTU method is a dimensionless analysis technique for heat exchanger performance that is
    particularly useful when outlet temperatures are unknown. This class handles the full workflow
    from raw experimental data to calibrated performance models.

    Theory:
    -------
    The NTU method is based on three dimensionless parameters:
    1. Number of Transfer Units (NTU) = UA/(ṁcp)min
    2. Heat Capacity Rate Ratio (Cr) = (ṁcp)min/(ṁcp)max
    3. Effectiveness (ε) = Qactual/Qmax
    
    For counterflow heat exchangers:
    ε = [1 - exp(-NTU(1-Cr))] / [1 - Cr·exp(-NTU(1-Cr))]  for Cr < 1
    ε = NTU / (1 + NTU)                                     for Cr = 1

    Methodology:
    -----------
    1. Data Processing:
       - Clean and transform raw experimental data
       - Convert units to SI (kg/s, °C, kPa)
       - Structure data for model fitting

    2. NTU Model Development:
       - Back-calculate NTU values from effectiveness data
       - Fit polynomial surface model for NTU prediction
       - Validate model using R² statistics

    3. Pressure Drop Modeling:
       - Fit quadratic model based on fluid dynamics principles
       - Account for both friction and form losses
       - Validate against experimental data

    Attributes:
    ----------
    cp : float
        Specific heat capacity of the fluid [J/kg·K]
    cooler_model : str
        Identifier for the heat exchanger model
    area : float
        Heat transfer surface area [m²]
    sample_pressure_drop_model : dict
        Coefficients and metadata for pressure drop correlation
    cooling_water_pressure_drop : float
        Static pressure drop on cooling water side [Pa]

    Methods:
    --------
    transform_wpd_csv_clean:
        Clean and transform raw WebPlotDigitizer CSV data
    load_digitized_data:
        Convert flow rates from lb/hr to kg/s
    backsolve_ntu:
        Calculate NTU values from experimental data
    fit_ntu_surface_model:
        Create polynomial regression model for NTU
    generate_ntu_model:
        Combine all steps to generate complete model
    save_model_to_json:
        Export model to JSON format

    References:
    ----------
    1. Shah, R.K., Sekulic, D.P., 2003. Fundamentals of Heat Exchanger Design.
       John Wiley & Sons, Chapter 3: Basic Thermal Design Theory
    2. Incropera, F.P., DeWitt, D.P., 2007. Fundamentals of Heat and Mass Transfer.
       John Wiley & Sons, Chapter 11: Heat Exchangers
    3. Kays, W.M., London, A.L., 1984. Compact Heat Exchangers.
       McGraw-Hill, Chapter 2: Basic Heat Exchanger Theory
    """

    def __init__(self, cp=4180):
        """
        Initializes the NTUModelBuilder with a specific heat capacity.

        Parameters:
        - cp (float): Specific heat capacity in J/kg·K (default: 4180 for water).
        """
        self.cp = cp
        self.cooler_model = None
        self.area = None
        self.sample_pressure_drop_model = None
        self.cooling_water_pressure_drop = None


    def set_cooler_properties(self, cooler_model, area, sample_pressure_drop_model, cooling_water_pressure_drop):
        """
        Sets the cooler properties for the NTU model.

        Parameters:
        - cooler_model (str): Name of the cooler model.
        - area (float): Heat transfer area in m².
        - sample_pressure_drop_model (dict): Regression model for sample-side pressure drop.
        - cooling_water_pressure_drop (float): Static cooling water pressure drop in Pa.
        """
        self.cooler_model = cooler_model
        self.area = area
        self.sample_pressure_drop_model = sample_pressure_drop_model
        self.cooling_water_pressure_drop = cooling_water_pressure_drop


    def transform_wpd_csv_clean(self, csv_path):
        """
        Transforms raw digitized heat exchanger performance data from WebPlotDigitizer into a structured format.
        
        This method handles the specific CSV format from WebPlotDigitizer's multi-series data export,
        where each temperature series is represented by alternating columns of flow rates and approach
        temperatures.

        Data Processing Steps:
        1. Read raw CSV without headers
        2. Extract temperature series from first row
        3. Remove metadata rows
        4. Reshape data into long format
        5. Filter out any missing/invalid data points

        Parameters:
        -----------
        csv_path : str
            Path to the raw CSV file from WebPlotDigitizer export.
            Expected format:
            Row 1: Hot inlet temperatures [°C]
            Row 2: Column headers (ignored)
            Row 3+: Alternating columns of:
                   - Flow rate [lb/hr]
                   - Approach temperature [°C]

        Returns:
        --------
        pd.DataFrame
            Cleaned and structured DataFrame with columns:
            - flow_lb_hr: Process fluid flow rate [lb/hr]
            - T_h_in_C: Hot fluid inlet temperature [°C]
            - approach_C: Approach temperature [°C]

        Notes:
        ------
        1. Missing or non-numeric values are automatically filtered
        2. Original units are preserved for later conversion
        3. Each row represents one experimental data point
        """
        df_full = pd.read_csv(csv_path, header=None)
        T_h_list = df_full.iloc[0, ::2].astype(float).tolist()
        df_data = df_full.drop([0, 1]).reset_index(drop=True)
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

    def load_digitized_data(self, data):
        """
        Converts experimental heat exchanger data from imperial to SI units and structures for analysis.
        
        This method serves as a data preprocessing step, converting flow rates from lb/hr to kg/s
        while preserving temperature data in Celsius. It handles both file paths and DataFrames
        for flexibility in data sources.

        Unit Conversions:
        ----------------
        Flow rate: lb/hr → kg/s
        1 lb/hr = 0.000125998 kg/s

        The conversion factor includes:
        - 1 lb = 0.45359237 kg
        - 1 hr = 3600 s
        
        Parameters:
        -----------
        data : str or pd.DataFrame
            Either:
            - Path to CSV file containing experimental data
            - DataFrame with 'flow_lb_hr' column
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with additional column:
            - flow_kg_s: Process fluid mass flow rate [kg/s]
            Original columns are preserved.

        Notes:
        ------
        1. If input is DataFrame, a copy is made to preserve original
        2. Temperatures remain in Celsius (no conversion needed)
        3. Only performs conversion if 'flow_lb_hr' column exists
        """
        if isinstance(data, str):
            df = pd.read_csv(data)
        else:
            df = data.copy()
        if "flow_lb_hr" in df.columns:
            df["flow_kg_s"] = df["flow_lb_hr"].apply(lambda lbhr: lbhr * 0.000125998)
        return df

    def fit_pressure_drop_model(csv_path):
        """
        Loads pressure drop data from a CSV, converts units, fits a quadratic regression model, and evaluates the fit.

        Parameters:
        - csv_path (str): Path to the CSV file containing pressure drop data.

        Returns:
        - dict: A dictionary containing the regression model coefficients, intercept, and R² value.
        """
        # Load the CSV data
        df = pd.read_csv(csv_path, skiprows=1, header=0)
        
        if not {"X", "Y"}.issubset(df.columns):
            raise ValueError("CSV must contain 'X' and 'Y' columns.")
    
        # Convert units
        df["flow_kg_s"] = df["X"] * 0.000125998  # Convert lb/hr to kg/s
        df["pressure_drop_kPa"] = df["Y"] * 6.89476  # Convert PSI to kPa
        
        # Extract features (X) and target (y)
        X = df[["flow_kg_s"]].values
        y = df["pressure_drop_kPa"].values
        
        # Add polynomial features (degree 2)
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        # Fit a linear regression model
        model = LinearRegression()
        model.fit(X_poly, y)
        
        # Predict and calculate R²
        y_pred = model.predict(X_poly)
        r2 = r2_score(y, y_pred)
        
        # Export the model as a dictionary
        model_data = {
            "coefficients": model.coef_.tolist(),
            "intercept": model.intercept_,
            "r2": r2,
            "feature_names": poly.get_feature_names_out(["flow_kg_s"]).tolist()
        }
        
        return model_data

    def backsolve_ntu(self, df_ready, T_c_in, m_dot_c):
        """
        Back-calculates NTU values from experimental effectiveness data using root-finding methods.
        
        This method implements the inverse solution of the NTU-effectiveness relationship for
        counterflow heat exchangers. For each experimental data point, it:
        1. Calculates actual heat transfer rate
        2. Determines effectiveness
        3. Solves numerically for the NTU value that produces this effectiveness

        Theory:
        -------
        The method uses energy balances and effectiveness definitions:

        1. Energy Balance:
           Q = ṁh·cp·(Th,in - Th,out)
           Q = ṁc·cp·(Tc,out - Tc,in)

        2. Effectiveness:
           ε = Qactual/Qmax
           where Qmax = Cmin·(Th,in - Tc,in)

        3. NTU-Effectiveness Relationship (Counterflow):
           ε = [1 - exp(-NTU(1-Cr))] / [1 - Cr·exp(-NTU(1-Cr))]  for Cr < 1
           ε = NTU / (1 + NTU)                                     for Cr = 1

        Numerical Method:
        ---------------
        Uses Brent's method (scipy.optimize.root_scalar) to solve:
           f(NTU) = ε(NTU) - εexperimental = 0
        
        Parameters:
        -----------
        df_ready : pd.DataFrame
            Prepared dataset containing:
            - flow_kg_s: Hot-side mass flow rate [kg/s]
            - T_h_in_C: Hot-side inlet temperature [°C]
            - approach_C: Temperature approach [°C]
        T_c_in : float
            Cold-side (cooling water) inlet temperature [°C]
        m_dot_c : float
            Cold-side mass flow rate [kg/s]

        Returns:
        --------
        pd.DataFrame
            Original data augmented with:
            - NTU: Calculated Number of Transfer Units [-]
            - Effectiveness: Heat exchanger effectiveness [-]

        Notes:
        ------
        1. NTU search range: [0.001, 10]
        2. Solution may fail for physically impossible conditions
        3. Failed solutions return None for NTU

        References:
        ----------
        1. Shah, R.K., Sekulic, D.P., 2003. Fundamentals of Heat Exchanger Design
           John Wiley & Sons, pp. 128-135
        2. Kays, W.M., London, A.L., 1984. Compact Heat Exchangers
           McGraw-Hill, Chapter 2.3: Basic Relations for Cross Flow
        """
        X = df_ready[["flow_kg_s", "T_h_in_C", "approach_C"]].values
        y = df_ready["NTU"].values

        # Initial guess for NTU
        ntu_initial = 1.0

        # Define the residual function for root finding
        def residual(ntu):
            # Calculate effectiveness using the current NTU guess
            df_ready["NTU"] = ntu
            df_ready["Effectiveness"] = df_ready.apply(
                lambda row: self.effectiveness(row["NTU"], row["Cr"]), axis=1
            )
            # Calculate the residuals as the difference between calculated and experimental effectiveness
            return df_ready["Effectiveness"] - df_ready["ε_experimental"]

        # Perform root finding to solve for NTU
        result = root_scalar(
            residual,
            bracket=[0.001, 10],
            method="brentq",
            xtol=1e-6,
            maxiter=100,
        )

        if not result.converged:
            raise RuntimeError("NTU calculation did not converge")

        ntu_calculated = result.root

        # Append the calculated NTU values to the DataFrame
        df_ready["NTU"] = ntu_calculated

        return df_ready

    def fit_ntu_surface_model(self, ntu_df):
        """
        Fits a polynomial surface model to predict NTU values from operating conditions.
        
        This method creates a 2D polynomial regression model that captures the relationship
        between NTU and both mass flow rate and inlet temperature. The model enables
        interpolation within the experimental data range.

        Model Structure:
        --------------
        NTU = b₀ + b₁·ṁ + b₂·T + b₃·ṁ² + b₄·ṁT + b₅·T²

        where:
        - ṁ: Mass flow rate [kg/s]
        - T: Inlet temperature [°C]
        - bᵢ: Regression coefficients

        Theory:
        -------
        The polynomial form is chosen based on heat transfer theory:
        1. Reynolds number effects (Re ∝ ṁ)
        2. Temperature-dependent property variations
        3. Combined effects through cross-terms

        The quadratic terms account for:
        - Non-linear friction effects
        - Property variation effects
        - Secondary flow effects in helical geometry

        Parameters:
        -----------
        ntu_df : pd.DataFrame
            DataFrame containing:
            - flow_kg_s: Mass flow rate [kg/s]
            - T_h_in_C: Inlet temperature [°C]
            - NTU: Calculated NTU values [-]

        Returns:
        --------
        tuple
            - model: sklearn.pipeline.Pipeline
                Fitted polynomial regression model
            - r2: float
                R² score indicating goodness of fit

        Notes:
        ----
        1. Uses sklearn's Pipeline combining:
           - PolynomialFeatures (degree=2)
           - LinearRegression
        2. No bias term (include_bias=False)
        3. Input features are not scaled (assumed pre-scaled)

        References:
        ----------
        1. Hastie, T., et al., 2009. The Elements of Statistical Learning
           Springer, Chapter 7: Model Assessment and Selection
        2. VDI Heat Atlas, 2010. "Heat Transfer to Helically Coiled Tubes"
           Springer, Chapter G1
        """
        X = ntu_df[["flow_kg_s", "T_h_in_C"]].values
        y = ntu_df["NTU"].values
        model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression())
        model.fit(X, y)
        
        # Predict NTU values and calculate R²
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        
        return model, r2

    def export_model_to_dict(self, model, r2):
        """
        Exports the polynomial regression model to a dictionary.

        Parameters:
        - model (sklearn.pipeline.Pipeline): Fitted polynomial regression model.

        Returns:
        - dict: Model data dictionary containing coefficients and metadata.
        """
        poly = model.named_steps["polynomialfeatures"]
        lin_reg = model.named_steps["linearregression"]
        return {
            "cooler_model": self.cooler_model,
            "area_m2": self.area,
            "sample_pressure_drop_model": self.sample_pressure_drop_model,
            "cooling_water_pressure_drop": self.cooling_water_pressure_drop,
            "degree": poly.degree,
            "include_bias": poly.include_bias,
            "feature_names": poly.get_feature_names_out(["m_dot_h", "T_h_in"]).tolist(),
            "coefficients": lin_reg.coef_.tolist(),
            "intercept": lin_reg.intercept_,
            "r2": r2,
        }


    def generate_ntu_model(self, df_ready, T_c_in=35.0, m_dot_c=0.757):
        """
        Combines all steps to generate an NTU model from the dataset.

        Parameters:
        - df_ready (pd.DataFrame): Prepared dataset with flow rates and temperatures.
        - T_c_in (float): Cold-side inlet temperature in °C (default: 35.0).
        - m_dot_c (float): Cold-side mass flow rate in kg/s (default: 0.757).

        Returns:
        - tuple: (Fitted model, model data dictionary, NTU DataFrame).
        """
        ntu_df = self.backsolve_ntu(df_ready, T_c_in, m_dot_c)
        model, r2 = self.fit_ntu_surface_model(ntu_df)
        model_data = self.export_model_to_dict(model, r2)
        return model, model_data, ntu_df


    def save_model_to_json(self, model_data, path):
        """
        Saves the NTU model data to a JSON file.

        Parameters:
        - model_data (dict): Model data dictionary.
        - path (str): Path to save the JSON file.
        """
        with open(path, "w") as f:
            json.dump(model_data, f, indent=4)


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
        """
        Evaluates the Number of Transfer Units (NTU) using a calibrated polynomial surface model.

        The model uses a polynomial regression to predict NTU based on:
        - Sample mass flow rate
        - Sample inlet temperature

        The relationship accounts for:
        - Flow regime effects through mass flow rate
        - Temperature-dependent property variations
        - Interaction effects through cross-terms

        Args:
            m_dot_h (float): Sample mass flow rate in kg/s
            T_h_in (float): Sample inlet temperature in °C

        Returns:
            float: Predicted NTU value for the given conditions
        """
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
        """
        Calculates heat exchanger effectiveness for counterflow configuration.

        The effectiveness (ε) is the ratio of actual to maximum possible heat transfer:
        ε = Q_actual / Q_max

        For counterflow exchangers:
        - When Cr < 1: ε = [1 - exp(-NTU(1-Cr))] / [1 - Cr·exp(-NTU(1-Cr))]
        - When Cr = 1: ε = NTU / (1 + NTU)

        Args:
            NTU (float): Number of Transfer Units
            Cr (float): Heat capacity rate ratio (C_min/C_max)

        Returns:
            float: Heat exchanger effectiveness (0 to 1)
        """
        if Cr == 1.0:
            return NTU / (1 + NTU)
        return (1 - np.exp(-NTU * (1 - Cr))) / (1 - Cr * np.exp(-NTU * (1 - Cr)))

    def predict_approach(self, m_dot_h, T_h_in, T_c_in, m_dot_c):
        """
        Predicts the temperature approach between sample outlet and cooling water inlet.

        The approach temperature is a key performance metric defined as:
        ΔT_approach = T_h_out - T_c_in

        Method:
        1. Calculate heat capacity rates (C = ṁcp)
        2. Determine C_min/C_max and capacity ratio
        3. Evaluate NTU from polynomial model
        4. Calculate effectiveness
        5. Compute heat transfer and outlet temperatures

        Args:
            m_dot_h (float): Sample mass flow rate [kg/s]
            T_h_in (float): Sample inlet temperature [°C]
            T_c_in (float): Cooling water inlet temperature [°C]
            m_dot_c (float): Cooling water mass flow rate [kg/s]

        Returns:
            float: Predicted approach temperature [°C]
        """
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
        Estimates the effective overall heat transfer coefficient (U) for current conditions.

        The U-value is back-calculated using the NTU definition:
        U = (NTU * C_min) / A

        This method provides insights into heat transfer performance and can be used to:
        - Monitor heat exchanger fouling
        - Validate design assumptions
        - Compare performance across operating conditions

        Args:
            m_dot_h (float): Sample mass flow rate [kg/s]
            m_dot_c (float): Cooling water mass flow rate [kg/s]
            T_h_in (float): Sample inlet temperature [°C]

        Returns:
            float: Estimated overall heat transfer coefficient [W/m²·K]

        Notes:
            - Uses calibrated NTU model for current conditions
            - Assumes constant specific heat capacity
            - Value may vary with flow rates and temperatures
        """
        C_h = m_dot_h * self.cp
        C_c = m_dot_c * self.cp
        C_min = min(C_h, C_c)
        NTU = self.evaluate_ntu(m_dot_h, T_h_in)
        U_est = (NTU * C_min) / self.area
        return U_est


    def solve_flowrate(self, mode, T_h_in, T_c_in, m_dot_c, target):
        """
        Determines required sample flow rate to achieve a specified performance target.

        This method uses root-finding algorithms to solve for the sample flow rate
        that achieves one of three possible target conditions:
        - Approach temperature (T_h_out - T_c_in)
        - Cooling water outlet temperature (T_c_out)
        - Heat duty (Q)

        Solution Method:
        1. Define objective function based on mode
        2. Attempt to bracket the solution
        3. Use Brent's method for root-finding if bracketing succeeds
        4. Fall back to bounded minimization if bracketing fails

        Args:
            mode (str): Solution target type:
                - 'approach': Solve for approach temperature
                - 'tc_out': Solve for cooling water outlet temperature
                - 'heat_duty': Solve for heat transfer rate
            T_h_in (float): Sample inlet temperature [°C]
            T_c_in (float): Cooling water inlet temperature [°C]
            m_dot_c (float): Cooling water mass flow rate [kg/s]
            target (float): Target value (units depend on mode):
                - approach: Temperature difference [°C]
                - tc_out: Absolute temperature [°C]
                - heat_duty: Heat transfer rate [W]

        Returns:
            float: Required sample mass flow rate [kg/s]

        Raises:
            ValueError: If mode is invalid or no solution found within tolerance
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

    def load_all(self, directory_path):
        """
        Loads all coolers from JSON files in a specified directory into SampleCooler instances.

        Parameters:
        - directory_path (str): Path to the directory containing JSON files.

        Returns:
        - list: List of SampleCooler instances.
        """

        coolers = []
        for filename in os.listdir(directory_path):
            if filename.endswith(".json"):
                file_path = os.path.join(directory_path, filename)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    # Filter keys to match SampleCooler attributes
                    allowed_keys = {"cooler_model", "area_m2", "sample_pressure_drop_model", "cooling_water_pressure_drop","feature_names", "coefficients", "intercept"}
                    filtered_data = {key: value for key, value in data.items() if key in allowed_keys}
                    coolers.append(SampleCooler(area_m2=data["area_m2"],
                                               U_W_per_m2K=None,
                                               model_data=data,
                                               cp=4180,
                                               name=data["cooler_model"]))
        return coolers


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
