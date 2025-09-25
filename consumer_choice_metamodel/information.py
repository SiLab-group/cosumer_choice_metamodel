"""
Information processing classes for the Consumer Choice Metamodel
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any

from .agent import AgentAttributes
from .environment import KnowledgeAsset


class InformationFilter(ABC):
    """Abstract base class for information filtering"""
    
    @abstractmethod
    def filter(self, agent_attributes: AgentAttributes, 
              knowledge_assets: List[KnowledgeAsset]) -> List[KnowledgeAsset]:
        """Filter knowledge assets based on agent characteristics"""
        pass


class InformationDistorter(ABC):
    """Abstract base class for information distortion"""
    
    @abstractmethod
    def distort(self, agent_attributes: AgentAttributes, 
               knowledge: KnowledgeAsset) -> KnowledgeAsset:
        """Apply cognitive biases and distortion to knowledge"""
        pass


class Transformer:
    """Manages information transformation between agent and environment"""
    
    def __init__(self, filters: List[InformationFilter], 
                 distorters: List[InformationDistorter]):
        self.filters = filters
        self.distorters = distorters
    
    def process_incoming(self, agent_attributes: AgentAttributes,
                        knowledge_assets: List[KnowledgeAsset]) -> List[KnowledgeAsset]:
        """Process incoming information through filters and distortion"""
        # Apply all filters
        filtered_assets = knowledge_assets
        for filter_impl in self.filters:
            filtered_assets = filter_impl.filter(agent_attributes, filtered_assets)
        
        # Apply distortion
        distorted_assets = []
        for asset in filtered_assets:
            distorted_asset = asset
            for distorter in self.distorters:
                distorted_asset = distorter.distort(agent_attributes, distorted_asset)
            distorted_assets.append(distorted_asset)
        
        return distorted_assets
    
    def process_outgoing(self, agent_attributes: AgentAttributes, 
                        communication: Dict[str, Any]) -> Dict[str, Any]:
        """Process outgoing communication (can be overridden for specific distortions)"""
        return communication
