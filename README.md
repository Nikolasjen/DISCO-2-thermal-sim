# SatOP Thermal Simulation Tool
This tool is designed to simulate the thermal behavior of the DISCO-2 satellite using flight-plans created in the SatOP platform as input. It leverages the Mph library to access the Comsol Multiphysics API to interact with the thermal model of the satellite.

The COMSOL model that this tool relies on is too large for GitHub and is not included in this repository. A way to gain access to the model may be provided in the future.

## Future Work
- Save model and exports using flight-plan artifact ID... This way if people have the exact same flightplan and model, they don't have to re-run the simulation, they can just load the previous results.


### Limitations
- On windows python can be installed through their windows store... This isolates the python installation in a "package" which the Comsol software can not move out of to find the java installation. This is a problem as Mph uses Java to run the Comsol client/server and virtual environment - TODO: move this to README.md instead of report



<!-- ## Limitations and initial state
- its a known limitiation/problem that my tool does not update the initial state of the satellite after a flight-plan has been run, however this should be possible to do in the future by using the telemetry data from the flight-plan to set the initial state of the model.
- Furthermore, concatenating multiple flight-plans into a single scenario currently requires all the flight-plans to be combined into one flight-plan before being converted to a scenario. The conversion is already handled by the 'flight-plan to scenario' module but being able to start a simulation from the latest state before changes were made to the flight-plan (be it removing parts or simply changing the order of events) would be a great addition to the tool. This would allow the operator to run multiple simulations with different event configurations without having to start from scratch each time, thus saving time and effort.
- As it stands, the initial state of the model is always assumed to be the same between simulations -- for the time being this means that all scenarios are run under the assumption of the worst case of being in direct sunlight for the entire duration of the simulation as the model does not currently support orbits for the satellite, and thus does not take into account the fact that the satellite may be in Earth's shadow for part of the simulation. A future model may include this orbital feature without having to change my tool, however, this does mean that my tool currently requires a bit of manual work to change the initial state of the model to match the expected initial state of the satellite before running a simulation. -->

#### Thought experiment
If I know the duration of the events and I known the energy that each event uses, I can calculate the theoretical total energy for each event... Assuming each heat source is a block of aluminium with a specific heat capacity of 0.9 J/g°C, and a mass of 100 gram, I can calculate the theoretical temperature increase for each event. This can be used to compare the expected temperature increase with the e.g. +10°C increase for the camera block... Maybe this could be used in future work to make a much simpler model - and thus faster.


### Special Thanks
Thank you Jesper for the COMSOL model and helping me understand how to edit it - I will find a better way to give you credit in the future.