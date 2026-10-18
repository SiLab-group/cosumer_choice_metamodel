"""
Factory pattern classes for creating model components
"""

import random
from abc import ABC, abstractmethod
from typing import Dict, Any

from .agent import AgentAttributes, ChoiceModule
from .environment import PhysicalAsset, KnowledgeAsset
from .information import Transformer


class ModelComponentFactory(ABC):
    """Abstract factory for creating model components"""

    @property
    def rng(self) -> random.Random:
        """Random number generator for stochastic component creation (set to the model's rng)"""
        if getattr(self, '_rng', None) is None:
            self._rng = random.Random()
        return self._rng

    @rng.setter
    def rng(self, value: random.Random) -> None:
        self._rng = value

    @abstractmethod
    def create_agent_attributes(self, config: Dict[str, Any]) -> AgentAttributes:
        """Create agent attributes"""
        pass
    
    @abstractmethod
    def create_choice_module(self, agent_id: str, attributes: AgentAttributes, 
                           config: Dict[str, Any]) -> ChoiceModule:
        """Create choice module"""
        pass
    
    @abstractmethod
    def create_transformer(self, config: Dict[str, Any]) -> Transformer:
        """Create transformer"""
        pass
    
    @abstractmethod
    def create_physical_asset(self, asset_config: Dict[str, Any]) -> PhysicalAsset:
        """Create physical asset"""
        pass
    
    @abstractmethod
    def create_knowledge_asset(self, knowledge_config: Dict[str, Any]) -> KnowledgeAsset:
        """Create knowledge asset"""
        pass
