import math  

class HelicalSampleCooler:
    def __init__(
        self,
        sample_inlet_temp=204,
        inlet_water_temp=35,  #Fixed at 35 °C for Model FSR-6225
        desired_outlet_temp=40,
        m_sample=0.014109,  # kg/s
        m_water=0.75,   # kg/s  #45L/min to kg/s
        cp_sample=4180,  # J/kg-K, default water
        cp_water=4180,   # J/kg-K
        area=0.16         # m^2, givenfor Model FSR-6225
    ):
        self.sample_inlet_temp = sample_inlet_temp
        self.inlet_water_temp = inlet_water_temp
        self.desired_outlet_temp = desired_outlet_temp
        self.m_sample = m_sample
        self.m_water = m_water
        self.cp_sample = cp_sample
        self.cp_water = cp_water
        self.area = area
        self.approach_temp = desired_outlet_temp - inlet_water_temp

    def calculate_lmtd(self, t_hot_in, t_hot_out, t_cold_in, t_cold_out):
        deltaT1 = t_hot_in - t_cold_out
        deltaT2 = t_hot_out - t_cold_in
        if abs(deltaT1 - deltaT2) < 1e-6:
            return deltaT1
        return (deltaT1 - deltaT2) / math.log(deltaT1 / deltaT2)

    def calculate_outlet_sample_temperature(self):
        # Assume desired_outlet_temp is achievable, calculate Q
        Q = self.m_sample * self.cp_sample * (self.sample_inlet_temp - self.desired_outlet_temp)
        # Calculate outlet water temp
        outlet_water_temp = self.inlet_water_temp + Q / (self.m_water * self.cp_water)
        # LMTD for counter-current
        lmtd = self.calculate_lmtd(
            self.sample_inlet_temp, self.desired_outlet_temp,
            self.inlet_water_temp, outlet_water_temp
        )
        # U is unknown, so return Q/(A*LMTD) as U
        U = Q / (self.area * lmtd) if lmtd > 0 else 0
        return {
            "outlet_sample_temp": self.desired_outlet_temp,
            "outlet_water_temp": outlet_water_temp,
            "U_required": U
        }

    def calculate_pressure_drop(self):
        # Placeholder for pressure drop calculation
        return 0.1

    def check_phase_change(self, outlet_water_temp):
        if outlet_water_temp >= 100:
            return "Phase change occurred: Water is vapor."
        else:
            return "No phase change: Water remains liquid."

    def run(self):
        results = self.calculate_outlet_sample_temperature()
        pressure_drop = self.calculate_pressure_drop()
        phase_change_status = self.check_phase_change(results["outlet_water_temp"])

        return {
            "outlet_sample_temp": results["outlet_sample_temp"],
            "pressure_drop": pressure_drop,
            "phase_change_status": phase_change_status,
            "outlet_water_temp": results["outlet_water_temp"],
            "U_required": results["U_required"],
            "Approach_temp": self.approach_temp,
        }

    def print_results(self):
        results = self.run()
        print(f"Outlet Sample Temperature: {results['outlet_sample_temp']:.2f} °C")
        print(f"Outlet Water Temperature: {results['outlet_water_temp']:.2f} °C")
        print(f"Required U (W/m²·K): {results['U_required']:.2f}")
        print(f"Pressure Drop (bar): {results['pressure_drop']}")
        print(f"Phase Change Status: {results['phase_change_status']}")
        print(f"Approach Temperature: {results['Approach_temp']:.2f} °C")

class SampleFlowCalculator:
    """
    Calculates the required sample flowrate to achieve a given approach temperature,
    given the inlet sample temperature, cooling water flowrate, and inlet water temperature.
    """
    def __init__(
        self,
        approach_temp,           # °C, desired approach (sample out - water in)
        sample_inlet_temp,       # °C
        water_inlet_temp,        # °C
        water_flowrate,          # kg/s
        area=3.5,                # m^2, heat exchange area
        cp_sample=4180,          # J/kg-K
        cp_water=4180,           # J/kg-K
        U=800                    # W/m^2-K, assumed or estimated
    ):
        self.approach_temp = approach_temp
        self.sample_inlet_temp = sample_inlet_temp
        self.water_inlet_temp = water_inlet_temp
        self.water_flowrate = water_flowrate
        self.area = area
        self.cp_sample = cp_sample
        self.cp_water = cp_water
        self.U = U

    def calculate_required_sample_flow(self):
        # Guess an outlet water temperature (no phase change, so <100°C)
        # We'll use an iterative approach to solve for sample flowrate

        # Target outlet sample temperature
        sample_out_temp = self.water_inlet_temp + self.approach_temp

        # Initial guess for sample flowrate
        m_sample_guess = 0.01  # kg/s
        tolerance = 1e-5
        max_iter = 100
        for _ in range(max_iter):
            Q = m_sample_guess * self.cp_sample * (self.sample_inlet_temp - sample_out_temp)
            water_out_temp = self.water_inlet_temp + Q / (self.water_flowrate * self.cp_water)
            lmtd = self._calculate_lmtd(
                self.sample_inlet_temp, sample_out_temp,
                self.water_inlet_temp, water_out_temp
            )
            Q_max = self.U * self.area * lmtd
            if Q_max == 0:
                break
            m_sample_new = Q_max / (self.cp_sample * (self.sample_inlet_temp - sample_out_temp))
            if abs(m_sample_new - m_sample_guess) < tolerance:
                return {
                    "required_sample_flowrate": m_sample_new,
                    "sample_out_temp": sample_out_temp,
                    "water_out_temp": water_out_temp,
                    "Q": Q_max,
                    "LMTD": lmtd
                }
            m_sample_guess = m_sample_new
        return {
            "required_sample_flowrate": None,
            "sample_out_temp": sample_out_temp,
            "water_out_temp": None,
            "Q": None,
            "LMTD": None
        }

    def _calculate_lmtd(self, t_hot_in, t_hot_out, t_cold_in, t_cold_out):
        deltaT1 = t_hot_in - t_cold_out
        deltaT2 = t_hot_out - t_cold_in
        if abs(deltaT1 - deltaT2) < 1e-6:
            return deltaT1
        return (deltaT1 - deltaT2) / math.log(deltaT1 / deltaT2)

# Example usage:
if __name__ == "__main__":
    cooler = HelicalSampleCooler()
    cooler.print_results()
    # print("\n--- Sample Flow Calculator Example ---")
    # calc = SampleFlowCalculator(
    #     approach_temp=10,            # °C
    #     sample_inlet_temp=204,       # °C
    #     water_inlet_temp=35,         # °C
    #     water_flowrate=0.75,         # kg/s
    #     area=0.16,                    # m^2
    #     U=1800                        # W/m^2-K (assumed)
    # )
    # result = calc.calculate_required_sample_flow()
    # if result["required_sample_flowrate"] is not None:
    #     print(f"Required Sample Flowrate: {result['required_sample_flowrate']:.5f} kg/s")
    #     print(f"Sample Outlet Temp: {result['sample_out_temp']:.2f} °C")
    #     print(f"Water Outlet Temp: {result['water_out_temp']:.2f} °C")
    #     print(f"Heat Duty (Q): {result['Q']:.2f} W")
    #     print(f"LMTD: {result['LMTD']:.2f} °C")
    # else:
    #     print("Could not converge to a solution for the required sample flowrate.")

