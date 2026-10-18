"""
Consumer Choice Metamodel Framework

This package provides abstract base classes and interfaces for implementing
the Consumer Choice Metamodel as described in "Consumer Choice Metamodel: 
A Conceptual Validation Approach" by Amy Liffey et al.

The framework defines the core architectural components as abstract classes,
to be implemented for a specific domain and modeling platform.

Usage:
    from consumer_choice_metamodel import (
        ConsumerChoiceModel, ConsumerAgent, Environment,
        PhysicalAsset, KnowledgeAsset, AgentAttributes
    )
"""

from .types import TriggerType, EvaluationDimension
from .agent import AgentAttributes, ChoiceModule, ConsumerAgent
from .environment import (
    PhysicalAsset, KnowledgeAsset, Network,
    RulesOfInteraction, ExogenousProcess, Environment
)
from .information import InformationFilter, InformationDistorter, Transformer
from .schedule import Scheduler, SequentialScheduler, RandomActivationScheduler, Phase, PhasedScheduler
from .model import ConsumerChoiceModel
from .factory import ModelComponentFactory
from .utils import ModelValidator, ModelEvent, EventBus

__version__ = "1.1.0"
__author__ = "Consumer Choice Metamodel Framework"
__description__ = "Abstract framework for consumer choice modeling"

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

    # Scheduling
    'Scheduler',
    'SequentialScheduler',
    'RandomActivationScheduler',
    'Phase',
    'PhasedScheduler',

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
    Return a starting template for implementing a basic consumer choice model.
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
    # add_trigger() has a default implementation; override it if needed

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
