"""
Consumer Choice Metamodel Framework
===================================

This package provides abstract base classes and interfaces for implementing
the Consumer Choice Metamodel as described in "Consumer Choice Metamodel: 
A Conceptual Validation Approach" by Amy Liffey et al.

The framework defines the core architectural components without specific
implementation details, allowing for flexible adaptation across different
domains and modeling platforms.

Usage:
    from consumer_choice_metamodel import (
        ConsumerChoiceModel, ConsumerAgent, Environment,
        PhysicalAsset, KnowledgeAsset, AgentAttributes
    )
"""

# Import all base types and enums
from .types import TriggerType, EvaluationDimension

# Import agent-related classes
from .agent import AgentAttributes, ChoiceModule, ConsumerAgent

# Import environment and asset classes
from .environment import (
    PhysicalAsset, KnowledgeAsset, Network, 
    RulesOfInteraction, ExogenousProcess, Environment
)

# Import information processing classes
from .information import InformationFilter, InformationDistorter, Transformer

# Import main model class
from .model import ConsumerChoiceModel

# Import factory classes
from .factory import ModelComponentFactory

# Import utilities
from .utils import ModelValidator, ModelEvent, EventBus

# Package metadata
__version__ = "1.0.0"
__author__ = "Consumer Choice Metamodel Framework"
__description__ = "Abstract framework for consumer choice modeling"

# Define what gets imported with "from consumer_choice_metamodel import *"
__all__ = [
    # Types
    'TriggerType',
    'EvaluationDimension',
    
    # Agent classes
    'AgentAttributes',
    'ChoiceModule',
    'ConsumerAgent',
    
    # Environment classes
    'PhysicalAsset',
    'KnowledgeAsset',
    'Network',
    'RulesOfInteraction',
    'ExogenousProcess',
    'Environment',
    
    # Information processing
    'InformationFilter',
    'InformationDistorter',
    'Transformer',
    
    # Main model
    'ConsumerChoiceModel',
    
    # Factory
    'ModelComponentFactory',
    
    # Utilities
    'ModelValidator',
    'ModelEvent',
    'EventBus',
]


def get_version():
    """Return the package version"""
    return __version__


def create_simple_model_template():
    """
    Return a simple template for implementing a basic consumer choice model.
    This can serve as a starting point for users.
    """
    template = """
# Example implementation template
from consumer_choice_metamodel import (
    ConsumerChoiceModel, ConsumerAgent, Environment,
    AgentAttributes, ChoiceModule, PhysicalAsset
)

class MyAgentAttributes(AgentAttributes):
    def get_psychological_attributes(self):
        return {'risk_aversion': 0.5, 'environmental_concern': 0.7}
    
    def get_socioeconomic_attributes(self):
        return {'income': 50000, 'age': 35}
    
    def get_stock_variables(self):
        return {'car': None, 'house': 'apartment'}
    
    def update_attributes(self, changes):
        pass

class MyChoiceModule(ChoiceModule):
    def add_trigger(self, trigger_type, context):
        self.pending_triggers.append({'type': trigger_type, 'context': context})
    
    def evaluate_option(self, option, evaluation_context):
        # Return evaluations for each dimension
        pass
    
    def aggregate_evaluations(self, evaluations):
        # Aggregate dimension scores into final utility
        pass
    
    def make_choice(self, options, choice_context):
        # Select best option
        pass

# Implement other required classes...
"""
    return template
