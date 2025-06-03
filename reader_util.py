""" EXAMPLE DATA FILE
% Model:              COPY__DISCO_thermal_analysis.mph
% Version:            COMSOL 6.2.0.415
% Date:               Mar 25 2025, 09:23
% Dimension:          0
% Nodes:              526
% Expressions:        526
% Description:        Temperature
% Length unit:        m
% X
0.0535
% Y
0.026
% Z
0.2615
% T (K) @ t=0
267.15000000000043
% T (K) @ t=0.40462
267.1499991540429
% T (K) @ t=0.79143
267.149998910309
% T (K) @ t=1.5651
267.149998767943
% T (K) @ t=3.1123
267.1500109978694
"""

import numpy as np
import os 
import matplotlib.pyplot as plt

from typing import Tuple, List
from dataclasses import dataclass

import io

# ========== Data Classes ==========
@dataclass
class Event:
    t_start: float
    t_stop: float
    color: str = "grey"
    label: str = "Event"

@dataclass
class Scenario:
    name: str
    events: list[Event]

@dataclass
class StateColors:
    no_states: str = "grey"
    standby: str = "blue"
    nominal: str = "green"
    observe: str = "orange"
    load: str = "red"
    transmit: str = "purple"

    def get(self, state:str) -> str:
        if state.lower() == "no_states":
            return self.no_states
        elif state.lower() == "standby":
            return self.standby
        elif state.lower() == "nominal":
            return self.nominal
        elif state.lower() == "observe":
            return self.observe
        elif state.lower() == "load":
            return self.load

# MIN_TEMP = -84.44 # deg C
# MAX_TEMP = 48.88  # deg C

# ========== Functions ==========
def read_data(path:str) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
    with open(path, "r") as file:
        data = file.readlines()

        if len(data) < 8:
            raise Exception(f'Invalid data file: {path}')

        # Ignore first 8 lines
        data = data[8:]

        # Extracting data
        x = float(data[1])
        y = float(data[3])
        z = float(data[5])

        # Extracting temperature data
        temp = []
        for i in range(7, len(data), 2):
            temp.append(float(data[i].strip()))
        temp = np.array(temp)

        # Extracting time data
        time = []
        for i in range(6, len(data), 2):
            t_val = data[i].split("=")[1].strip()
            time.append(float(t_val))
        time = np.array(time)
    
    return (temp, time, (x, y, z))

def convert_temp(temp:float, convert_unit:str) -> float:
    """
    Convert temperature from one unit to another.
    """
    if convert_unit == "K->C":
        return temp - 273.15
    elif convert_unit == "K->F":
        return (temp - 273.15) * 9/5 + 32
    elif convert_unit == "K->R":
        return (temp - 273.15) * 9/5 + 491.67
    elif convert_unit == "K->K":
        return temp
    elif convert_unit == "C->K":
        return temp + 273.15
    elif convert_unit == "F->K":
        return (temp - 32) * 5/9 + 273.15
    elif convert_unit == "R->K":
        return (temp - 491.67) * 5/9 + 273.15
    else:
        raise ValueError(f"Invalid conversion unit: {convert_unit}") 

def _within_bounds(value:float, min_value:float, max_value:float) -> bool:
    """
    Check if a value is within the specified bounds.
    """
    return min_value <= value <= max_value

def plot_data(
        data_path:str, 
        probes:list = None, 
        scenario:Scenario = None, 
        convert_unit:str = "K->C", 
        show_marks:bool = False, 
        title_addition:str = "", 
        title_overwrite:str="", 
        ignore_missing_data=False,
        save_path: str = None,
        fileobj = None, # For returning a file object instead of saving to disk
        xLookUp: Tuple[int, int] = None
        ) -> None:
    if len(title_addition) > 0 and len(title_overwrite) > 0:
        raise ValueError("title_addition and title_overwrite cannot be both set")
    
    if len(title_addition) > 0:
        title_addition = f" - {title_addition}"

    _print_temp_len_once = True #False
    # read all files in directory
    paths = []
    files = [file for file in os.listdir(data_path) if file.endswith(".txt")]
    if ignore_missing_data:
        files = [file for file in files if "T12" not in file]

    if probes:
        for file in files:
            if any(probe in file for probe in probes):
                paths.append(os.path.join(data_path, file))
    else:
        paths.extend(os.path.join(data_path, file) for file in files)

    # print(f"paths: {paths}")

    # big plot
    # print(f"Number of files: {len(paths)}")
    if len(paths) == 0:
        raise ValueError(f"No matching datasets found in {data_path}")
    row = len(paths) // 3
    if len(paths) % 3 != 0:
        row += 1
    col = 3
    if row == 1 and len(paths) < 3:
        col = len(paths)

    # print(f"row and col: {row}, {col}")
    
    fig, ax = plt.subplots(row, col, figsize=(5*col, 3.5 * row))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i, path in enumerate(paths):
        try: 
            if not os.path.exists(path):
                
                raise Exception(f"File not found: {path}")
            
            if os.path.exists(path) and os.path.getsize(path) == 0:
                raise Exception(f"Empty file: {path}")

            if col == 1 or row == 1:
                ax_index = ax[i]
            else:
                ax_index = ax[i//row, i%col]

            temp, time, (x, y, z) = read_data(path)
            if not _print_temp_len_once:
                # print(f"Length of temp: {len(temp)}, Length of time: {len(time)}")
                _print_temp_len_once = True
            
            # Check if there is any data
            if len(temp) == 0 or len(time) == 0:
                ax_index.set_title(f"{path.split('ata - ')[1].split('.')[0]}")
                raise Exception(f"Invalid data: {path}; Length of temp: {len(temp)}, Length of time: {len(time)}")
            
            # Check if the data is valid for plotting
            if len(temp) != len(time):
                ax_index.set_title(f"{path.split('ata - ')[1].split('.')[0]}")
                raise Exception(f"Invalid data: {path}; Length of temp: {len(temp)}, Length of time: {len(time)}")
            
            # Check if data contains any NaN values
            if np.isnan(temp).any() or np.isnan(time).any():
                ax_index.set_title(f"{path.split('ata - ')[1].split('.')[0]}")
                raise Exception(f"Invalid data: {path}; Contains NaN values")

            # Plotting
            temp = convert_temp(temp, convert_unit)
            
            if show_marks:
                ax_index.plot(time, temp, label="Temperature", marker='o', markersize=3)
            else:
                ax_index.plot(time, temp, label="Temperature")
            ax_index.set_title(f"{path.split('ata - ')[1].split('.')[0]}")
            ax_index.set_xlabel("Time (s)")
            ax_index.set_ylabel(f"Temperature ({convert_unit.split('->')[1]})")
            # ax_index.set_ylabel("Temperature (K)")
            ax_index.legend()

            # Include scenario events legend
            if scenario:
                for event in scenario.events:
                    ax_index.axvspan(event.t_start, event.t_stop, color=event.color, alpha=0.2, label=event.label)
            
            # Include max and min temperature lines
            # ax[i//col, i%col].axhline(y=MAX_TEMP, color='r', linestyle='--', label=f"Max Temp: {MAX_TEMP}°C")
            # ax[i//col, i%col].axhline(y=MIN_TEMP, color='b', linestyle='--', label=f"Min Temp: {MIN_TEMP}°C")
            
            # Add x-axis limits if xLookUp is provided
            if xLookUp:
                ax_index.set_xlim(xLookUp[0], xLookUp[1])

        except ValueError as e:
            print(f"ValueError: {e} \n at {path}")
            continue
        # except Exception as e:
        #     print(f"Error: {e}")
        #     continue

    labels = []
    handles = []
    if scenario:
        # Add a legend for the scenario events
        for event in scenario.events:
            if event.label in labels:
                continue
            labels.append(event.label)
            handles.append(plt.Line2D([0], [0], color=event.color, lw=4, alpha=0.4))
        if title_overwrite:
            fig.suptitle(f"Temperature vs Time - {title_overwrite}", fontsize=16)
        else:
            fig.suptitle(f"Temperature vs Time - {scenario.name}{title_addition}", fontsize=16)
 
    fig.legend(handles, labels, loc='upper left', fontsize=10)

    plt.tight_layout()
    # plt.show()    

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        # print(f"Plot saved to {save_path}")
    elif fileobj is not None:
        fig.savefig(fileobj, format="png", bbox_inches='tight')
    plt.close(fig)