"""
Main model class for the Consumer Choice Metamodel
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

from .agent import ConsumerAgent
from .environment import Environment


class ConsumerChoiceModel(ABC):
    """Abstract base class for the complete model"""
    
    def __init__(self):
        self.environment: Optional[Environment] = None
        self.agents: List[ConsumerAgent] = []
        self.current_time: int = 0
        self.model_parameters: Dict[str, Any] = {}
    
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
    
    def step(self) -> None:
        """Execute one model step"""
        # Update environment
        if self.environment:
            self.environment.update_environment(self.current_time)
        
        # Execute agent steps
        for agent in self.agents:
            agent.step(self.environment)
        
        self.current_time += 1
    
    @abstractmethod
    def collect_data(self) -> Dict[str, Any]:
        """Collect data for analysis"""
        pass
    
    @abstractmethod
    def run_simulation(self, steps: int) -> Dict[str, Any]:
        """Run complete simulation"""
        pass
