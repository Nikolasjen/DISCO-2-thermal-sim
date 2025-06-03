from dataclasses import dataclass
from .flightPlan import FlightPlan

@dataclass
class Event:
    name: str
    t_start: int
    r_pad: float

@dataclass
class Scenario:
    name: str
    environ_temp: str
    events: list[Event]
    t_end: int = 8900
    T0: str  = "0.0[degC]"

# CONSTANTS
class KNOWN_COMMANDS:
    STANDBY = ("gpio-write", "debug-standby")
    NOMINAL = ("debug-nominal",)
    OBSERVATION = ("capture_image", "debug-observation")
    LOAD = ("debug-load")
    TRANSMIT = ("debug-transmit")


def _get_flight_plan_commands(flight_plan: FlightPlan):
    return flight_plan.flight_plan["body"]


def _update_state_and_duration(command, list_of_states_and_duration):
    if command["name"] in KNOWN_COMMANDS.STANDBY:
        if len(list_of_states_and_duration) == 0 or not list_of_states_and_duration[-1][0] == "STANDBY":
            list_of_states_and_duration.append(("STANDBY", 0))
    elif command["name"] in KNOWN_COMMANDS.NOMINAL:
        if len(list_of_states_and_duration) == 0 or not list_of_states_and_duration[-1][0] == "NOMINAL":
            list_of_states_and_duration.append(("NOMINAL", 0))
    elif command["name"] in KNOWN_COMMANDS.OBSERVATION:
        if len(list_of_states_and_duration) == 0 or not list_of_states_and_duration[-1][0] == "OBSERVATION":
            list_of_states_and_duration.append(("OBSERVATION", 0))
    elif command["name"] in KNOWN_COMMANDS.LOAD:
        if len(list_of_states_and_duration) == 0 or not list_of_states_and_duration[-1][0] == "LOAD":
            list_of_states_and_duration.append(("LOAD", 0))
    elif command["name"] in KNOWN_COMMANDS.TRANSMIT:
        if len(list_of_states_and_duration) == 0 or not list_of_states_and_duration[-1][0] == "TRANSMIT":
            list_of_states_and_duration.append(("TRANSMIT", 0))
    
    elif command["name"] == "wait-sec":
        duration = command["duration"]
        list_of_states_and_duration[-1] = (list_of_states_and_duration[-1][0], list_of_states_and_duration[-1][1] + duration)
    else:
        # print(f"Unknown command: {command['name']}")
        raise ValueError(f"Unknown command: {command['name']}")

def get_scenario_from_flight_plan(flight_plan: FlightPlan) -> Scenario:
    commands = _get_flight_plan_commands(flight_plan)
    list_of_states_and_duration = []

    # I should run through all commands from the top and then go into the children.
    for i in range(len(commands)):
        if commands[i]["name"] == "repeat-n":
            repeat_count = commands[i]["count"]
            command_list = commands[i]["body"]
            for j in range(repeat_count):
                for sub_command in command_list:
                    _update_state_and_duration(sub_command, list_of_states_and_duration)
        else:
            _update_state_and_duration(commands[i], list_of_states_and_duration)

    # create a scenario object based on the list of states and duration
    event_start_time = 0
    events = []
    for state, duration in list_of_states_and_duration:
        events.append(Event(name=state, t_start=event_start_time, r_pad=0.5))
        event_start_time += duration

    scenario = Scenario(
        name="DISCO-2",
        environ_temp="0.0[degC]",
        events=events,
        t_end=event_start_time
    )

    return scenario

def give_unique_name(scenario: Scenario) -> str:
    """
    Give a unique name to the scenario based on the events and their start times.
    """
    list_of_states = {
        "STANDBY": 0,
        "NOMINAL": 0,
        "OBSERVATION": 0,
        "LOAD": 0,
        "TRANSMIT": 0
    }

    for event in scenario.events:
        if event.name in list_of_states:
            list_of_states[event.name] = list_of_states[event.name] + 1
            event.name += f"{list_of_states[event.name]}"
        else:
            raise ValueError(f"Unknown event name: {event.name}")
