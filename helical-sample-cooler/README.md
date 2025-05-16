# Helical Sample Cooler

## Overview
The Helical Sample Cooler project is designed to calculate the outlet sample temperature and pressure drop from a helical sample cooler heat exchange system, similar to a Sentry cooler for SWAS applications. The program takes the sample inlet temperature, inlet water temperature, and desired outlet sample temperature as inputs. It also checks for phase changes in the shell side cooling water and calculates the outlet temperature of the water.

## Features
- Calculate outlet sample temperature based on input parameters.
- Determine pressure drop across the cooler.
- Check for phase change in the cooling water and calculate the outlet water temperature.

## Installation
To install the required dependencies, run the following command:

```
pip install -r requirements.txt
```

## Usage
To use the Helical Sample Cooler, you can create an instance of the `HelicalSampleCooler` class from the `cooler.py` module and call its methods with the appropriate parameters.

### Example
```python
from src.cooler import HelicalSampleCooler

cooler = HelicalSampleCooler()
outlet_sample_temp = cooler.calculate_outlet_sample_temp(inlet_temp, inlet_water_temp, desired_outlet_temp)
pressure_drop = cooler.calculate_pressure_drop(inlet_temp, outlet_sample_temp)
outlet_water_temp = cooler.check_phase_change(inlet_water_temp)
```

## Contributing
Contributions to the Helical Sample Cooler project are welcome. Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.