from .reader_util import plot_data, Event, Scenario, StateColors, read_data
from dataclasses import dataclass

@dataclass
class component:
    name: str
    min_temp: float
    max_temp: float

class KNOWN_COMPONENTS:
    list_of_components = [
        component("CAMERA", -40, 50)
    ]



def check_thermal_limits(data_path: str) -> tuple[str, float]:
    
    temp, time, _ = read_data(data_path)
    
    for t, timestamp in zip(temp, time):
        for component in KNOWN_COMPONENTS.list_of_components:
            if t < component.min_temp or t > component.max_temp:
                return (f"Temperature out of range for {component.name} at time {timestamp}: {t}°C", timestamp)
                