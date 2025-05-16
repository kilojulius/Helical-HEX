class Sample:
    def __init__(self, inlet_temperature, desired_outlet_temperature):
        self.inlet_temperature = inlet_temperature
        self.desired_outlet_temperature = desired_outlet_temperature

class CoolingWater:
    def __init__(self, inlet_temperature):
        self.inlet_temperature = inlet_temperature
        self.outlet_temperature = None

    def calculate_outlet_temperature(self, sample_outlet_temperature):
        # Simple model for outlet temperature calculation
        self.outlet_temperature = (self.inlet_temperature + sample_outlet_temperature) / 2
        return self.outlet_temperature