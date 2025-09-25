"""
Agent-related classes for the Consumer Choice Metamodel
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union

from .types import TriggerType, EvaluationDimension


@dataclass
class AgentAttributes(ABC):
    """Abstract base class for agent attributes"""
    agent_id: str
    
    @abstractmethod
    def get_psychological_attributes(self) -> Dict[str, float]:
        """Return psychological factors (0-1 scale)"""
        pass
    
    @abstractmethod  
    def get_socioeconomic_attributes(self) -> Dict[str, Union[int, float]]:
        """Return socio-economic characteristics"""
        pass
    
    @abstractmethod
    def get_stock_variables(self) -> Dict[str, Any]:
        """Return owned assets/technologies"""
        pass
    
    @abstractmethod
    def update_attributes(self, changes: Dict[str, Any]) -> None:
        """Update attributes based on decisions or external changes"""
        pass


class ChoiceModule(ABC):
    """Abstract base class for agent choice module"""
    
    def __init__(self, agent_id: str, agent_attributes: AgentAttributes):
        self.agent_id = agent_id
        self.agent_attributes = agent_attributes
        self.pending_triggers: List[Dict[str, Any]] = []
    
    @abstractmethod
    def add_trigger(self, trigger_type: TriggerType, context: Dict[str, Any]) -> None:
        """Add trigger event"""
        pass
    
    @abstractmethod
    def evaluate_option(self, option: 'PhysicalAsset', 
                       evaluation_context: Dict[str, Any]) -> Dict[EvaluationDimension, float]:
        """Evaluate option across all relevant dimensions"""
        pass
    
    @abstractmethod
    def aggregate_evaluations(self, evaluations: Dict[EvaluationDimension, float]) -> float:
        """Aggregate dimension evaluations into final utility"""
        pass
    
    @abstractmethod
    def make_choice(self, options: List['PhysicalAsset'], 
                   choice_context: Dict[str, Any]) -> Optional['PhysicalAsset']:
        """Make final choice from available options"""
        pass
    
    def has_pending_triggers(self) -> bool:
        """Check if agent has pending decision triggers"""
        return len(self.pending_triggers) > 0
    
    def clear_triggers(self) -> None:
        """Clear all pending triggers"""
        self.pending_triggers.clear()


class ConsumerAgent(ABC):
    """Abstract base class for consumer agents"""
    
    def __init__(self, agent_id: str, attributes: AgentAttributes, 
                 choice_module: ChoiceModule, transformer: 'Transformer'):
        self.agent_id = agent_id
        self.attributes = attributes
        self.choice_module = choice_module
        self.transformer = transformer
        self.memory: Dict[str, Any] = {}
    
    @abstractmethod
    def perceive_environment(self, environment: 'Environment') -> Dict[str, Any]:
        """Perceive and process environmental information"""
        pass
    
    @abstractmethod
    def make_decisions(self, environment: 'Environment') -> List[Dict[str, Any]]:
        """Make decisions based on current state"""
        pass
    
    @abstractmethod
    def communicate(self, environment: 'Environment', decisions: List[Dict[str, Any]]) -> None:
        """Communicate decisions/information to environment"""
        pass
    
    @abstractmethod
    def update_state(self, decisions: List[Dict[str, Any]], environment_feedback: Dict[str, Any]) -> None:
        """Update agent state based on decisions and feedback"""
        pass
    
    def step(self, environment: 'Environment') -> None:
        """Main agent step function - template method pattern"""
        # 1. Perceive environment  
        perception = self.perceive_environment(environment)
        
        # 2. Make decisions
        decisions = self.make_decisions(environment)
        
        # 3. Communicate to environment
        self.communicate(environment, decisions)
        
        # 4. Update internal state
        environment_feedback = environment.get_network_context(self.agent_id)
        self.update_state(decisions, environment_feedback)
