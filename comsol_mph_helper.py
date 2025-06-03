import os
from dataclasses import dataclass
from mph import Model, Node
from datetime import datetime

"""
Efficiency	    0.3	        0.3 	    Solar cell efficiency
Temp	        -600[degC]	-326.85 K	Initial Temperature of the external structure
t_end	        500[s] + 2[h] + 20[min]	8900[s]	
R_pad	        0.5	        0.5 	    Effect of R_pad
SOMx_PSU_stb	206.5[mW]	0.2065 W	Effect of any SOM_PSU (internal PSU) in standby mode. Only one SOM can be active at a time! 80% effeciency.
SOMx_PSU_nom	392.5[mW]	0.3925 W	Effect of any SOM_PSU (internal PSU) in nominal mode. Only one SOM can be active at a time! 80% effeciency.
SOMx_PSU_load	764.5[mW]	0.7645 W	Effect of any SOM_PSU (internal PSU) in load mode. Only one SOM can be active at a time! 80% effeciency.
SOMx_PSU_obs	389.4[mW]	0.3894 W	Effect of any SOM_PSU (internal PSU) in observe mode. Only one SOM can be active at a time! 80% effeciency.
SOMx_PSU_tx	    391.8[mW]	0.3918 W	Effect of any SOM_PSU (internal PSU) in transmission mode. Only one SOM can be active at a time! 80% effeciency.
SOMx_CPU_stb	825.9[mW]	0.8259 W	Effect of any SOM_CPU in standby mode. Only one SOM can be active at a time! 0% effeciency.
SOMx_CPU_nom	1569.9[mW]	1.5699 W	Effect of any SOM_CPU in nominal mode. Only one SOM can be active at a time! 0% effeciency.
SOMx_CPU_load	3057.9[mW]	3.0579 W	Effect of any SOM_CPU in load mode. Only one SOM can be active at a time! 0% effeciency.
SOMx_CPU_obs	1557.5[mW]	1.5575 W	Effect of any SOM_CPU in observe mode. Only one SOM can be active at a time! 0% effeciency.
SOMx_CPU_tx	    1567.3[mW]	1.5673 W	Effect of any SOM_CPU in transmission mode. Only one SOM can be active at a time! 0% effeciency.
PSU_5V_stb	    91[mW]	    0.091 W	    Effect of 5V external PSU(x) in standby mode. 93% effeciency.
PSU_5V_nom	    161[mW]	    0.161 W	    Effect of 5V external PSU(x) in nominal mode. 93% effeciency.
PSU_5V_load	    301[mW]	    0.301 W	    Effect of 5V external PSU(x) in load mode. 93% effeciency.
PSU_5V_obs	    177.1[mW]	0.1771 W	Effect of 5V external PSU(x) in observe mode. 93% effeciency.
PSU_5V_tx	    161[mW]	    0.161 W	    Effect of 5V external PSU(x) in transmission mode. 93% effeciency.
PSU_3.3V_stb	11.6[mW]	0.0116 W	Effect of 3.3V external PSU(x) in standby mode. 93% effeciency.
PSU_3.3V_nom	11.6[mW]	0.0116 W	Effect of 3.3V external PSU(x) in nominal mode. 93% effeciency.
PSU_3.3V_load	11.6[mW]	0.0116 W	Effect of 3.3V external PSU(x) in load mode. 93% effeciency.
PSU_3.3V_obs	26.6[mW]	0.0266 W	Effect of 3.3V external PSU(x) in observe mode. 93% effeciency.
PSU_3.3V_tx	    11.6[mW]	0.0116 W	Effect of 3.3V external PSU(x) in transmission mode. 93% effeciency.
CAM1_obs	    3300[mW]	3.3 W	    (output)
CAM2_obs	    2000[mW]	2 W	        (output)
IR_CAM_obs	    1200[mW]	1.2 W	    (output)
"""


@dataclass
class power_consumption:
    name: str
    description: str
    unit: str
    SOMx_PSU: float
    SOMx_CPU: float
    PSU_5V: float
    PSU_3_3V: float
    CAM1_obs: float
    CAM2_obs: float
    IR_CAM_obs: float

STANDBY = power_consumption(
    name="standby",
    description="Effect of any SOM_PSU (internal PSU) in standby mode. Only one SOM can be active at a time! 80% effeciency.",
    unit="mW",
    SOMx_PSU=206.5,
    SOMx_CPU=825.9,
    PSU_5V=91,
    PSU_3_3V=11.6,
    CAM1_obs=0,
    CAM2_obs=0,
    IR_CAM_obs= 0
)

NOMINAL = power_consumption(
    name="nominal",
    description="Effect of any SOM_PSU (internal PSU) in nominal mode. Only one SOM can be active at a time! 80% effeciency.",
    unit="mW",
    SOMx_PSU=392.5,
    SOMx_CPU=1569.9,
    PSU_5V=161,
    PSU_3_3V=11.6,
    CAM1_obs=0,
    CAM2_obs=0,
    IR_CAM_obs= 0
)

LOAD = power_consumption(
    name="load",
    description="Effect of any SOM_PSU (internal PSU) in load mode. Only one SOM can be active at a time! 80% effeciency.",
    unit="mW",
    SOMx_PSU=764.5,
    SOMx_CPU=3057.9,
    PSU_5V=301,
    PSU_3_3V=11.6,
    CAM1_obs=0,
    CAM2_obs=0,
    IR_CAM_obs= 0
)

OBSERVE = power_consumption(
    name="observe",
    description="Effect of any SOM_PSU (internal PSU) in observe mode. Only one SOM can be active at a time! 80% effeciency.",
    unit="mW",
    SOMx_PSU=389.4,
    SOMx_CPU=1557.5,
    PSU_5V=177.1,
    PSU_3_3V=26.6,
    CAM1_obs=3300,
    CAM2_obs=2000,
    IR_CAM_obs=2500 # 1200
)

TRANSMISSION = power_consumption(
    name="transmission",
    description="Effect of any SOM_PSU (internal PSU) in transmission mode. Only one SOM can be active at a time! 80% effeciency.",
    unit="mW",
    SOMx_PSU=391.8,
    SOMx_CPU=1567.3,
    PSU_5V=161,
    PSU_3_3V=11.6,
    CAM1_obs=0,
    CAM2_obs=0,
    IR_CAM_obs= 0
)

def set_event_state_properties(node:Node, event_name:str, camera = 'CAM1', somx = 'SOM1', r_pad: float = 1.0):

    if event_name == "No_Event":
        return

    camera_options = ['CAM1', 'CAM2', 'IR_CAM']
    somx_options = ['SOM1', 'SOM2']
    if camera not in camera_options:
        print(f'Invalid camera "{camera}" - skipping...')
        return
    if somx not in somx_options:
        print(f'Invalid SOM "{somx}" - skipping...')
        return
    if not node.exists():
        print(f'Node "{node}" does not exist - skipping...')
        return
    
    event_value:power_consumption = None
    if "stand" in event_name.lower():
        event_value = STANDBY
    elif "nomi" in event_name.lower():
        event_value = NOMINAL
    elif "load" in event_name.lower():
        event_value = LOAD
    elif "obs" in event_name.lower():
        event_value = OBSERVE
    elif "trans" in event_name.lower():
        event_value = TRANSMISSION
    else:
        raise Exception(f'Invalid event name "{event_name}" - skipping...')
    
    # Calculate the power consumption
    _cam1_obs   =   r_pad * event_value.CAM1_obs
    _cam2_obs   =   r_pad * event_value.CAM2_obs
    _ir_cam_obs =   r_pad * event_value.IR_CAM_obs
    _som1_cpu   =   r_pad * event_value.SOMx_CPU
    _som1_psu   =   r_pad * event_value.SOMx_PSU   
    _som2_cpu   =   r_pad * event_value.SOMx_CPU   
    _som2_psu   =   r_pad * event_value.SOMx_PSU   
    _ps1_5v     =   r_pad * event_value.PSU_5V     
    _ps1_3_3v   =   r_pad * event_value.PSU_3_3V   
    _ps2_5v     =   r_pad * event_value.PSU_5V     
    _ps2_3_3v   =   r_pad * event_value.PSU_3_3V   
    
    # Create the event properties
    try:
        node.property('reInitName', [
            'CAM1',
            'CAM2',
            'SOM1_CPU',
            'SOM1_PSU',
            'PS1_5V',
            'PS1_3.3V',
            'SOM2_CPU',
            'SOM2_PSU',
            'PS2_5V',
            'PS2_3.3V',
            'IR_CAM'
            ])
        node.property('reInitValue', [
            f"{_cam1_obs}[{event_value.unit}]"   if camera == "CAM1"    else f"0[{event_value.unit}]",
            f"{_cam2_obs}[{event_value.unit}]"   if camera == "CAM2"    else f"0[{event_value.unit}]",
            f"{_som1_cpu}[{event_value.unit}]"   if somx == "SOM1"      else f"0[{event_value.unit}]",
            f"{_som1_psu}[{event_value.unit}]"   if somx == "SOM1"      else f"0[{event_value.unit}]",
            f"{_ps1_5v}[{event_value.unit}]"     if somx == "SOM1"      else f"0[{event_value.unit}]",
            f"{_ps1_3_3v}[{event_value.unit}]"   if somx == "SOM1"      else f"0[{event_value.unit}]",
            f"{_som2_cpu}[{event_value.unit}]"   if somx == "SOM2"      else f"0[{event_value.unit}]",
            f"{_som2_psu}[{event_value.unit}]"   if somx == "SOM2"      else f"0[{event_value.unit}]",
            f"{_ps2_5v}[{event_value.unit}]"     if somx == "SOM2"      else f"0[{event_value.unit}]",
            f"{_ps2_3_3v}[{event_value.unit}]"   if somx == "SOM2"      else f"0[{event_value.unit}]",
            f"{_ir_cam_obs}[{event_value.unit}]" if camera == "IR_CAM"  else f"0[{event_value.unit}]"
            ])
    except Exception as e:
        print(f"Error editing event {event_name}")
        raise e


def create_sequence(model:Model, sequence_name:str) -> Node | None:
    _node = model/'physics/Events'
    
    # Check if the sequence already exists
    if (_node/sequence_name).exists():
        print(f"Sequence {sequence_name} already exists - skipping...")
        return
    
    sequence = _node.create('EventSequence', name=sequence_name)

    sequence.rename(sequence_name)

    return sequence

def add_event_to_sequence(sequence:Node, event_name:str, duration:float, camera = 'CAM1', somx = 'SOM1', r_pad: float = 1.0):
    # Create SequenceMember
    # model.component("comp1").physics("ev").feature("es1").create("sm1", "SequenceMember", -1);
    sequence.create('SequenceMember', name=event_name)
    event = sequence/event_name

    # Set the properties of the SequenceMember
    event.property('endConditionOptions', 'duration')
    event.property('duration', f'{duration}[s]')
    event.property('stateName', event_name)

    set_event_state_properties(node = event, event_name=event_name, camera = camera, somx = somx, r_pad=r_pad)



def create_event(model:Model, event_name:str, event_time:float, event_type:str='ExplicitEvent', camera = 'CAM1', somx = 'SOM1', r_pad: float = 1.0):
    _node = model/'physics/Events'
    
    # Check if the event already exists
    if (_node/event_name).exists():
        print(f"Event {event_name} already exists - skipping...")
        return
    
    # Create the event
    try:
        event = _node.create(event_type, name=event_name)
        # event.rename(event_name)
        event.property('start', f"{event_time}[s]")
        set_event_state_properties(node = event, event_name=event_name, camera = camera, somx = somx, r_pad=r_pad)

    except Exception as e:
        print(f"Error creating event {event_name} - {e}")
        # Remove the event if it was created
        if event.exists():
            event.remove()
        # Remove any events that were created in the process but not renamed
        for child in _node.children():
            if "event" in child.name().lower():
                print(f"Warning: Removing event '{child.name()}' due to error in creating '{event_name}'")
                child.remove()
                break
        raise e
    


from mph import Model, Node
from datetime import datetime
import os
def extract_data(model:Model, data_path:str, scenario_name: str = "", ignore_existing:bool = True) -> None:
    """
    Extracts data from the model and saves it to the specified path.
    
    Parameters:
    - model: The COMSOL model object.
    - data_path: The path to save the data.
    - ignore_existing: If True, skips existing files. Defaults to True.
    """
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    save_data_dir_name = f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")} - {model.name()}{f" - {scenario_name}" if len(scenario_name) > 0 else ""}'
    new_dir_name = os.path.join(data_path, save_data_dir_name)
    if not os.path.exists(new_dir_name):
        os.makedirs(new_dir_name)
    else:
        print(f"Directory {new_dir_name} already exists - This should not happen as it is temporal!") 

    
    # Ensure that the model has exports for the data
    exports_node: Node = model/"exports"

    if not exports_node.exists():
        print("No exports found in the model.")
        return
    
    # Check if there are any data exports in the model
    # If not, create a new export node for each dataset
    list_of_exports = model.exports()
    list_of_datasets = model.datasets()
    for dataset in list_of_datasets:
        if f"Data - {dataset}" not in list_of_exports and not dataset.endswith(" 1"):
            export_node = exports_node.create('Data')
            export_node.rename(f"Data - {dataset}")
            (export_node).property('data', (model/'datasets'/dataset))
            (export_node).property('transpose', 'on')


    # Export the data
    _exports:list[str] = model.exports()
    for export in _exports:
        if export.endswith(" 1"):
            # Skip the default export that is created by COMSOL
            continue

        _export_file = f"{export}.txt"
        export_path = os.path.join(new_dir_name, _export_file)
        if ignore_existing and os.path.exists(export_path):
            print(f"Skipping {export}")
            continue

        if os.path.exists(export_path):
            # Assuming that the data directory is created correctly, this should never happen as the directory is created with a timestamp.
            print(f"Overwriting {export}")

        if export.startswith("Data"):
            try:
                print(f"Exporting {export}")
                model.export(export, export_path)
            except Exception as e:
                print(f"Error exporting {export}")
                raise e
    
    print("Finished exporting data.")
    print(f"Data saved to {new_dir_name}")

    return new_dir_name