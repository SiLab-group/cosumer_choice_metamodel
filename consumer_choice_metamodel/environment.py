"""
Environment and asset classes for the Consumer Choice Metamodel
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import uuid


@dataclass
class PhysicalAsset(ABC):
    """Abstract representation of physical assets in environment"""
    name: str
    asset_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    @abstractmethod
    def get_availability(self) -> float:
        """Return availability (0-1 scale)"""
        pass
    
    @abstractmethod
    def get_cost(self) -> float:
        """Return current cost"""
        pass
    
    @abstractmethod
    def get_properties(self) -> Dict[str, Any]:
        """Return all asset properties"""
        pass


@dataclass
class KnowledgeAsset(ABC):
    """Abstract representation of information in environment"""
    topic: str
    source: str
    asset_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    @abstractmethod
    def get_reliability(self) -> float:
        """Return information reliability (0-1 scale)"""
        pass
    
    @abstractmethod
    def get_content(self) -> Dict[str, Any]:
        """Return information content"""
        pass


class Network(ABC):
    """Abstract base class for agent networks"""
    
    @abstractmethod
    def get_neighbors(self, agent_id: str) -> List[str]:
        """Get neighboring agent IDs"""
        pass
    
    @abstractmethod
    def add_agent(self, agent_id: str, properties: Dict[str, Any]) -> None:
        """Add agent to network"""
        pass
    
    @abstractmethod
    def remove_agent(self, agent_id: str) -> None:
        """Remove agent from network"""
        pass
    
    @abstractmethod
    def get_network_properties(self, agent_id: str) -> Dict[str, Any]:
        """Get network-specific properties for agent"""
        pass


class RulesOfInteraction(ABC):
    """Abstract base class for interaction rules"""
    
    @abstractmethod
    def can_interact(self, agent_a: str, agent_b: str, context: Dict[str, Any]) -> bool:
        """Determine if two agents can interact"""
        pass
    
    @abstractmethod
    def get_interaction_effects(self, agent_a: str, agent_b: str, 
                              interaction_type: str) -> Dict[str, Any]:
        """Get effects of interaction between agents"""
        pass


class ExogenousProcess(ABC):
    """Abstract base class for external processes affecting environment"""
    
    @abstractmethod
    def update(self, current_time: int, environment_state: Dict[str, Any]) -> Dict[str, Any]:
        """Update process and return state changes"""
        pass
    
    @abstractmethod
    def affects_assets(self) -> List[str]:
        """Return list of asset types this process affects"""
        pass


class Environment(ABC):
    """Abstract base class for environment"""
    
    def __init__(self):
        self.physical_assets: List[PhysicalAsset] = []
        self.knowledge_assets: List[KnowledgeAsset] = []
        self.networks: List[Network] = []
        self.rules: List[RulesOfInteraction] = []
        self.exogenous_processes: List[ExogenousProcess] = []
        self.communications: List[Dict[str, Any]] = []
    
    @abstractmethod
    def add_physical_asset(self, asset: PhysicalAsset) -> None:
        """Add physical asset to environment"""
        pass
    
    @abstractmethod
    def add_knowledge_asset(self, asset: KnowledgeAsset) -> None:
        """Add knowledge asset to environment"""
        pass
    
    @abstractmethod
    def get_available_options(self, agent_id: str, context: Dict[str, Any]) -> List[PhysicalAsset]:
        """Get options available to specific agent"""
        pass
    
    @abstractmethod
    def get_relevant_knowledge(self, agent_id: str, context: Dict[str, Any]) -> List[KnowledgeAsset]:
        """Get knowledge assets relevant to agent"""
        pass
    
    @abstractmethod
    def update_environment(self, current_time: int) -> None:
        """Update environment state through exogenous processes"""
        pass
    
    @abstractmethod
    def add_communication(self, communication: Dict[str, Any]) -> None:
        """Add communication to environment"""
        pass
    
    def get_network_context(self, agent_id: str) -> Dict[str, Any]:
        """Get network context for agent across all networks"""
        context = {}
        for network in self.networks:
            network_props = network.get_network_properties(agent_id)
            context.update(network_props)
        return context
