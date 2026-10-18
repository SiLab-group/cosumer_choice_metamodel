"""
Main model class for the Consumer Choice Metamodel
"""

import random
from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Any, Optional

from .agent import ConsumerAgent
from .environment import Environment
from .schedule import Scheduler, SequentialScheduler
from .utils import EventBus, ModelEvent


class ConsumerChoiceModel(ABC):
    """Abstract base class for the complete model"""

    def __init__(self, seed: Optional[int] = None,
                 scheduler: Optional[Scheduler] = None):
        self.environment: Optional[Environment] = None
        self.agents: List[ConsumerAgent] = []
        self.current_time: int = 0
        self.model_parameters: Dict[str, Any] = {}
        self.seed = seed
        self.rng = random.Random(seed)
        self.scheduler: Scheduler = scheduler or SequentialScheduler()
        self.event_bus = EventBus(clock=lambda: self.current_time)

    def reseed(self, seed: Optional[int]) -> None:
        """Reset the model's random number generator (e.g. from parameters['random_seed'])"""
        self.seed = seed
        self.rng.seed(seed)

    @abstractmethod
    def initialize_model(self, parameters: Dict[str, Any]) -> None:
        """Initialize model with parameters"""
        pass

    @abstractmethod
    def create_agents(self, agent_count: int, agent_config: Dict[str, Any]) -> List[ConsumerAgent]:
        """Create and configure agents"""
        pass

    @abstractmethod
    def create_environment(self, environment_config: Dict[str, Any]) -> Environment:
        """Create and configure environment"""
        pass

    def add_agents(self, agents: Iterable[ConsumerAgent]) -> None:
        """Register agents and give them the model's random number generator"""
        for agent in agents:
            agent.rng = self.rng
            self.agents.append(agent)

    def publish(self, event_type: str, source: str, data: Dict[str, Any]) -> ModelEvent:
        """Publish an event on the model's event bus, stamped with the current time"""
        event = ModelEvent(event_type, source, data)
        self.event_bus.publish(event)
        return event

    def step(self) -> None:
        """Execute one model step"""
        # Update environment
        if self.environment:
            self.environment.update_environment(self.current_time)

        # Execute agent steps
        self.scheduler.run(self, self.agents)

        self.current_time += 1

    def collect_data(self) -> Dict[str, Any]:
        """Collect data for analysis (override to record model-specific outputs)"""
        return {'time': self.current_time, 'agent_count': len(self.agents)}

    def run_simulation(self, steps: int) -> Dict[str, Any]:
        """Run complete simulation, collecting data before the first step and after each step"""
        simulation_data = [self.collect_data()]
        for _ in range(steps):
            self.step()
            simulation_data.append(self.collect_data())
        return {
            'simulation_data': simulation_data,
            'final_stats': simulation_data[-1],
            'parameters': self.model_parameters,
        }
