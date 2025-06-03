# import io
import os
from fastapi import APIRouter, Depends, Request, HTTPException, status, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
import io
import logging
import uuid
import time

import sqlite3
from contextlib import closing

import mph
from datetime import datetime, timedelta
from .comsol_mph_helper import create_event, extract_data
from dataclasses import dataclass

from satop_platform.plugin_engine.plugin import Plugin
from .flightPlan import FlightPlan
from .flightPlan_to_scenario import get_scenario_from_flight_plan, Scenario, give_unique_name
from multiprocessing import Process, Queue
from .check_thermal_limits import check_thermal_limits

from .reader_util import plot_data

import sys
import threading

logger = logging.getLogger('plugin.comsol_simulator')

# # TODO: Move the dataclasses into a file of its own
# @dataclass
# class Event:
#     name: str
#     t_start: int
#     r_pad: float

# @dataclass
# class Scenario:
#     name: str
#     environ_temp: str
#     events: list[Event]
#     t_end: int = 8900
#     T0: str  = "0.0[degC]"

def run_simulation(
        flightplan_scenario: Scenario, 
        mph_path: str, 
        save_model_path: str, 
        data_dir:str, 
        queue:Queue, 
        save_model: bool, 
    ) -> None:
        # Reconfigure logger for subprocess
        _logger = logging.getLogger()
        _logger.setLevel(logging.DEBUG)

        console_log_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)7s] -- %(filename)20s:%(lineno)-4s -- %(message)s",
            "%Y-%m-%d %H:%M:%S"
        )
        console_log_handler.setFormatter(formatter)

        # Clear existing handlers to avoid duplicates
        if _logger.hasHandlers():
            _logger.handlers.clear()

        _logger.addHandler(console_log_handler)

        """Runs the simulation in a separate process.
        """
        _logger.info(f"Starting Comsol client...")
        mph.option('classkit', True)
        client = mph.start()

        _start_time = datetime.now()
        _logger.info(f"Running scenario: {flightplan_scenario.name} ... start at {_start_time.strftime('%H:%M:%S')}")
        model = client.load(mph_path)
            
        model.parameter("T_0", flightplan_scenario.T0) # Initial temperature
        model.parameter("T_env", flightplan_scenario.environ_temp) # Environment temperature
        model.parameter("t_end", max(flightplan_scenario.t_end, 1000)) # End time of the simulation
        
        for event in flightplan_scenario.events:
            if event.t_start > flightplan_scenario.t_end:
                _logger.warning(f"Event {event.name} starts at {event.t_start} which is after the end time of the simulation ({flightplan_scenario.t_end}). Skipping this event.")
                continue
            
            # Create the event
            create_event(model, event.name, event.t_start, r_pad=event.r_pad)

        # Run the simulation for the current scenario
        model.solve()
        _stop_time = datetime.now()
        _logger.info(f"Finished simulating {flightplan_scenario.name} in: {_stop_time - _start_time}")

        # TODO: rethink this as it will otherwise become very expensive to save the model
        # if save_model:
        #     Save the model with the scenario name
        #     _logger.info(f"Saving model for scenario: {flightplan_scenario.name}")
        #     model.save(save_model_path)
        #     _logger.info(f"Model saved to: {save_model_path}")

        # Export the data results
        _logger.info(f"Exporting data for scenario: {flightplan_scenario.name}")
        
        try:
            extracted_data_dir = extract_data(model, data_dir)
            _logger.debug(f"Data exported to: {extracted_data_dir}")
            finished_exporting_data = True
        except Exception as e:
            _logger.error(f"Error exporting data: {e}")
            finished_exporting_data = False
            extracted_data_dir = None

        client.remove(model) # Remove the model from the client

        # Run the check_thermal_limits function
        if not finished_exporting_data:
            _logger.warning("Data export was not successful, skipping thermal limits check.")
            msg = "Data export was not successful, skipping thermal limits check."
            timestamp = None
        else:
            _logger.info(f"Checking thermal limits for scenario: {flightplan_scenario.name}")
            msg, timestamp = check_thermal_limits(extracted_data_dir)
        
        try:
            if msg:
                _logger.warning(msg)

                # find suspect event
                last_event = None
                for event in flightplan_scenario.events:
                    if event.t_start > timestamp:
                        _logger.warning(f"Suspect event: {last_event.name} at {last_event.t_start}")
                        break
                    last_event = event
            else:
                _logger.info(f"Thermal limits are within range for scenario: {flightplan_scenario.name}")
        except Exception as e:
            _logger.error(f"Error checking thermal limits: {e}")
            msg = f"Error checking thermal limits: {e}"
        finally:
            # Put the result in the queue
            queue.put({
                "status": "finished",
                "scenario": flightplan_scenario.name,
                "message": msg,
                # "timestamp": timestamp
                "end_time": time.time(),  # end of the simulation - for whatever reason it ended.
                "data_path": extracted_data_dir if finished_exporting_data else None
            })
            _logger.info(f"Simulation for scenario: {flightplan_scenario.name} finished and result put in queue.")


class DiscoSim(Plugin):
    def __init__(self, *args, **kwargs):
        plugin_dir = os.path.dirname(os.path.realpath(__file__))
        super().__init__(plugin_dir, *args, **kwargs)

        if not self.check_required_capabilities(['http.add_routes']):
            raise RuntimeError

        self.api_router = APIRouter()
        self.data_dir = os.path.join(plugin_dir, 'data')
        os.makedirs(self.data_dir, exist_ok=True) # TODO: create this on startup instead of here
        self.simulations = {}  # simulation_id: {'process': ..., 'queue': ..., 'status': ..., 'start_time': ..., 'scenario': ...}
        self.db_path = os.path.join(self.data_dir, "simulations.db")
        self._init_db()

        self.comsol_model = self.config["comsol_model"]
        
        # --- Background status checker ---
        self._stop_status_thread = threading.Event()
        self._status_thread = threading.Thread(target=self._background_status_checker, daemon=True)
        self._status_thread.start()

        # --- Register the API routes ---
        @self.api_router.post(
                '/run_simulation', 
                summary="Takes a flight-plan and simulates it.",
                description="Takes a flight-plan and starts a simulation for said flight-plan in a COMSOL model and returns an ID to track the simulation.",
                response_description="Returns an ID to track the simulation.",
                status_code=200,
                dependencies=[Depends(self.platform_auth.require_login)]
                )
        async def simulate_flihtplan_schedule(flight_plan:FlightPlan, req: Request, save_model:bool = False) -> dict[str, str]:
            """Simulates a flight plan and returns an ID to track the simulation.

            Args:
                flight_plan (FlightPlan): The flight plan to simulate.
                req (Request): The request object.
                save_model (bool, optional): Whether to save the model or not. Defaults to False.
            
            returns:
                str: A unique ID to track the simulation.
            """
            simulation_id = str(uuid.uuid4())
            flightplan_scenario: Scenario = self.convert_flightplan_to_states(flight_plan)
            mph_path = os.path.join(plugin_dir, self.comsol_model)
            save_model_path = os.path.join(self.data_dir, f"{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")} - DISCO_thermal_analysis_2 - {flightplan_scenario.name}.mph")
            queue = Queue()

            if save_model:
                logger.info(f"User: '{req.state.user.username}' requested model to be saved to: {save_model_path}")

            process = Process(
                target=run_simulation, 
                args=(flightplan_scenario, mph_path, save_model_path, self.data_dir, queue, save_model)
            )

            self._db_add_simulation(simulation_id, flightplan_scenario.name, time.time())
            self.simulations[simulation_id] = {
                'process': process,
                'queue': queue,
                'status': 'running',
                'start_time': time.time(),
                'end_time': None,  # This will be set when the process finishes
                'scenario': flightplan_scenario.name,
                'data_path': None  # This will be set when the data is exported
            } # keep this for process/queue tracking only
            
            process.start()
            logger.info(f"Started simulation {simulation_id} for scenario: {flightplan_scenario.name}")
            return {
                "status": "started",
                "simulation_id": simulation_id,
                "message": f"Simulation started for scenario: {flightplan_scenario.name}"
            }
        
        @self.api_router.get(
                '/check_simulation/{simulation_id}',
                summary="Check the status of a simulation.",
                description="Checks the status of a simulation by its ID. Returns the status, scenario name, and running time.",
                response_description="Returns the status of the simulation.",
                status_code=200,
                dependencies=[Depends(self.platform_auth.require_login)]
                )
        async def check_simulation(simulation_id: str) -> dict:
            sim_db = self._db_get_simulation(simulation_id)
            sim_mem = self.simulations.get(simulation_id)
            if not sim_db:
                raise HTTPException(status_code=404, detail="Simulation ID not found")
            if sim_mem and sim_mem['status'] == 'running':
                proc = sim_mem['process']
                if proc.is_alive():
                    status = 'running'
                    elapsed = time.time() - sim_mem['start_time']
                else:
                    status = sim_db['status']
                    elapsed = sim_db['end_time'] - sim_db['start_time'] if sim_db['end_time'] else time.time() - sim_db['start_time']
            else:
                status = sim_db['status']
                elapsed = sim_db['end_time'] - sim_db['start_time'] if sim_db['end_time'] else time.time() - sim_db['start_time']
            return {
                "simulation_id": simulation_id,
                "status": status,
                "scenario": sim_db['scenario'],
                "running_time_seconds": elapsed,
                "start_time": sim_db['start_time_readable'],
                "end_time": sim_db['end_time_readable'],
            }

        @self.api_router.get(
                '/get_all_simulation_status',
                summary="Get the status of all simulations.",
                description="Returns the status of all simulations, including their IDs, scenarios, and running times.",
                response_description="Returns a list of all simulations with their statuses.",
                status_code=200,
                dependencies=[Depends(self.platform_auth.require_login)]
                )
        async def get_all_simulation_status() -> list:
            return self._db_get_all_simulations()

        @self.api_router.post(
                '/stop_simulation/{simulation_id}',
                summary="Stop a running simulation.",
                description="Stops a running simulation by its ID. If the simulation is already finished, it returns the current status.",
                response_description="Returns the status of the stopped simulation.",
                status_code=200,
                dependencies=[Depends(self.platform_auth.require_login)]
                )
        async def stop_simulation(simulation_id: str) -> dict:
            sim = self.simulations.get(simulation_id)
            if not sim:
                raise HTTPException(status_code=404, detail="Simulation ID not found")
            proc = sim['process']
            if proc.is_alive():
                proc.terminate()
                proc.join()
                sim['status'] = 'terminated'
                sim['end_time'] = time.time()
                # Update the database as well
                self._db_update_simulation(
                    simulation_id,
                    status='terminated',
                    end_time=sim['end_time']
                )
                logger.info(f"Simulation {simulation_id} terminated by user.")
                return {"status": "terminated", "simulation_id": simulation_id}
            else:
                # Also update DB if not already marked as terminated/finished
                sim_db = self._db_get_simulation(simulation_id)
                if sim_db and sim_db['status'] == 'running':
                    self._db_update_simulation(
                        simulation_id,
                        status='finished',
                        end_time=sim_db['end_time'] or time.time()
                    )
                return {"status": sim['status'], "simulation_id": simulation_id, "message": "Process already finished."}
            
        # Optionally, add endpoint for last X simulations
        @self.api_router.get(
                '/get_last_simulations/{count}',
                summary="Get the last X simulations.",
                description="Returns the last X simulations, sorted by start time in descending order.",
                response_description="Returns a list of the last X simulations with their statuses.",
                status_code=200,
                dependencies=[Depends(self.platform_auth.require_login)]
                )
        async def get_last_simulations(count: int = 5) -> list:
            return self._db_get_simulations(last_n=count)
        
        # Plot data and return it as a file
        @self.api_router.get(
            '/plot_data/{simulation_id}',
            summary="Plot data from a simulation.",
            description="Plots the data from a simulation by its ID and returns it as a file.",
            response_description="Returns a plot of the simulation data.",
            status_code=200,
            dependencies=[Depends(self.platform_auth.require_login)],
            responses = {
                200: {
                    "content": {"image/png": {}}
                }
            },
            response_class=StreamingResponse
        )
        async def plot_data_graph(simulation_id: str, background_tasks: BackgroundTasks, lookUp_tStart: int = None, lookUp_tEnd: int = None) -> StreamingResponse:
            sim = self._db_get_simulation(simulation_id)
            if not sim:
                raise HTTPException(status_code=404, detail="Simulation ID not found")
            if sim['status'] != 'finished':
                raise HTTPException(status_code=400, detail="Simulation is not finished yet")
            data_path = sim.get('data_path')
            if not data_path:
                raise HTTPException(status_code=400, detail="Data path not found for the simulation")
            if not os.path.exists(data_path):
                raise HTTPException(status_code=404, detail=f"Data file not found: {data_path}")

            logger.info(f"Plotting data for simulation {simulation_id} with scenario {sim['scenario']}")

            try:
                buf = io.BytesIO()
                plot_data(
                    data_path=data_path,
                    probes=None,
                    scenario=None,
                    convert_unit="K->C",
                    show_marks=False,
                    title_addition="",
                    title_overwrite="",
                    ignore_missing_data=True,
                    fileobj=buf,  # <-- pass BytesIO buffer
                    xLookUp=None if (lookUp_tStart is None and lookUp_tEnd is None) else (lookUp_tStart, lookUp_tEnd)
                )
                buf.seek(0)
                return StreamingResponse(buf, media_type="image/png")
            except Exception as e:
                logger.error(f"Error plotting data for simulation {simulation_id}: {e}")
                raise HTTPException(status_code=500, detail=f"Error plotting data: {str(e)}")


            
        
        # --- MISC API ROUTES ---
        @self.api_router.post(
            '/convert-flightplan',
            summary="Converts a flight plan to a scenario.",
            description="Converts a flight plan to a scenario that can be used for simulation.",
            response_description="The converted scenario.",
            status_code=200,
            dependencies=[Depends(self.platform_auth.require_login)]
        )
        async def convert_flightplan(flight_plan: FlightPlan) -> Scenario:
            """Converts a flight plan to a scenario."""
            flightplan_scenario: Scenario = self.convert_flightplan_to_states(flight_plan)
            logger.debug(f"Converted flightplan scenario: {flightplan_scenario}")
            return flightplan_scenario
        

    def _background_status_checker(self):
        """Background thread to update simulation statuses and end times."""
        while not self._stop_status_thread.is_set():
            for sim_id, sim in list(self.simulations.items()):
                proc = sim['process']
                queue = sim['queue']
                # Check if process finished and update status/end_time from queue
                if not proc.is_alive() and sim['status'] == 'running':
                    # Try to get result from queue (non-blocking)
                    try:
                        if not queue.empty():
                            result = queue.get_nowait()
                            sim['status'] = result.get('status', 'finished')
                            sim['end_time'] = result.get('end_time', time.time())
                            sim['message'] = result.get('message', '')
                            sim['data_path'] = result.get('data_path', None)

                            self._db_update_simulation(
                                sim_id,
                                status=sim['status'],
                                end_time=sim['end_time'],
                                data_path=sim.get('data_path'),
                                message=sim.get('message')
                            )
                                                    
                            logger.info(f"Simulation {sim_id} finished with status: {sim['status']}")
                    except Exception as e:
                        logger.error(f"Error reading from simulation queue: {e}")
                    # Failsafe: if queue is empty, set end_time anyway
                    if 'end_time' not in sim or sim['end_time'] is None:
                        sim['end_time'] = time.time()
                        sim['status'] = 'finished'
            self._stop_status_thread.wait(10)  # Sleep for 10 seconds or until stop is set


    def startup(self):
        """Startup protocol for the plugin
        """
        super().startup()
        logger.info(f"Running '{self.name}' statup protocol")

    
    def shutdown(self):
        """Shutdown protocol for the plugin"""
        logger.info(f"'{self.name}' Shutting down gracefully")
        # Stop the background thread
        self._stop_status_thread.set()
        if self._status_thread.is_alive():
            self._status_thread.join(timeout=15)
        # Failsafe: terminate all running simulation processes
        for sim_id, sim in self.simulations.items():
            proc = sim['process']
            if proc.is_alive():
                logger.info(f"Terminating simulation process {sim_id} during shutdown.")
                proc.terminate()
                proc.join(timeout=10)
                sim['status'] = 'terminated'
                sim['end_time'] = time.time()
                # Update the database as well
                self._db_update_simulation(
                    sim_id,
                    status='terminated',
                    end_time=sim['end_time']
                )
            else:
                # If process is not alive but status is still running, update DB
                sim_db = self._db_get_simulation(sim_id)
                if sim_db and sim_db['status'] == 'running':
                    self._db_update_simulation(
                        sim_id,
                        status='finished',
                        end_time=sim_db['end_time'] or time.time()
                    )
        super().shutdown()


    def convert_flightplan_to_states(self, flight_plan: FlightPlan) -> Scenario:
        """Converts a flight plan to a list of scenarios."""
        flight_plan_scenario: Scenario = get_scenario_from_flight_plan(flight_plan)
        give_unique_name(flight_plan_scenario)
        return flight_plan_scenario
    

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS simulations (
                    simulation_id TEXT PRIMARY KEY,
                    scenario TEXT,
                    status TEXT,
                    start_time REAL,
                    end_time REAL,
                    data_path TEXT,
                    message TEXT
                )
            """)
    def _db_add_simulation(self, sim_id, scenario, start_time):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO simulations (simulation_id, scenario, status, start_time) VALUES (?, ?, ?, ?)",
                (sim_id, scenario, "running", start_time)
            )

    def _db_update_simulation(self, sim_id, **kwargs):
        keys = ", ".join(f"{k}=?" for k in kwargs)
        values = list(kwargs.values())
        values.append(sim_id)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"UPDATE simulations SET {keys} WHERE simulation_id=?", values)


    def _format_time(self, ts):
        if ts is None:
            return None
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

    def _format_duration(self, start, end):
        if start is None:
            return None
        if end is None:
            end = time.time()
            prefix = "running for"
        else:
            prefix = "ran for"
        duration = timedelta(seconds=int(end - start))
        return f"{prefix} {duration}"

    def _db_get_simulation(self, sim_id):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT * FROM simulations WHERE simulation_id=?", (sim_id,))
            row = cur.fetchone()
            if row:
                keys = [d[0] for d in cur.description]
                d = dict(zip(keys, row))
                d['start_time_readable'] = self._format_time(d['start_time'])
                d['end_time_readable'] = self._format_time(d['end_time'])
                d['running_time'] = self._format_duration(d['start_time'], d['end_time'])
                return d
            return None

    def _db_get_simulations(self, last_n: int = 5):
        """Get the last n simulations from the database."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT * FROM simulations ORDER BY start_time DESC LIMIT ?", (last_n,))
            keys = [d[0] for d in cur.description]
            result = []
            for row in cur.fetchall():
                d = dict(zip(keys, row))
                d['start_time_readable'] = self._format_time(d['start_time'])
                d['end_time_readable'] = self._format_time(d['end_time'])
                d['running_time'] = self._format_duration(d['start_time'], d['end_time'])
                result.append(d)
            return result

    def _db_get_all_simulations(self):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT * FROM simulations ORDER BY start_time DESC")
            keys = [d[0] for d in cur.description]
            result = []
            for row in cur.fetchall():
                d = dict(zip(keys, row))
                d['start_time_readable'] = self._format_time(d['start_time'])
                d['end_time_readable'] = self._format_time(d['end_time'])
                d['running_time'] = self._format_duration(d['start_time'], d['end_time'])
                result.append(d)
            return result