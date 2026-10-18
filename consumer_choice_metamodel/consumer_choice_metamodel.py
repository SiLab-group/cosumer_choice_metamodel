"""
Consumer Choice Metamodel Framework - main entry point.

Re-exports all components for backward compatibility. New code can import from
the package directly:
    from consumer_choice_metamodel import ConsumerAgent, Environment
"""

from consumer_choice_metamodel import *
from consumer_choice_metamodel.types import TriggerType, EvaluationDimension
from consumer_choice_metamodel.agent import AgentAttributes, ChoiceModule, ConsumerAgent
from consumer_choice_metamodel.environment import (
    PhysicalAsset, KnowledgeAsset, Network, RulesOfInteraction,
    ExogenousProcess, Environment
)
from consumer_choice_metamodel.information import (
    InformationFilter, InformationDistorter, Transformer
)
from consumer_choice_metamodel.schedule import (
    Scheduler, SequentialScheduler, RandomActivationScheduler, Phase, PhasedScheduler
)
from consumer_choice_metamodel.model import ConsumerChoiceModel
from consumer_choice_metamodel.factory import ModelComponentFactory
from consumer_choice_metamodel.utils import ModelValidator, ModelEvent, EventBus

__doc__ = """
Abstract Consumer Choice Metamodel Framework

This module provides abstract base classes and interfaces for implementing
the Consumer Choice Metamodel as described in "Consumer Choice Metamodel:
A Conceptual Validation Approach" by Amy Liffey et al.

The framework defines the core architectural components as abstract classes,
to be implemented for a specific domain and modeling platform.
"""

__version__ = "1.1.0"
__author__ = "Consumer Choice Metamodel Framework"


def list_available_classes():
    """Return a list of all available classes in the framework"""
    return [
        'TriggerType', 'EvaluationDimension', 'AgentAttributes', 'ChoiceModule',
        'ConsumerAgent', 'PhysicalAsset', 'KnowledgeAsset', 'Network',
        'RulesOfInteraction', 'ExogenousProcess', 'Environment',
        'InformationFilter', 'InformationDistorter', 'Transformer',
        'Scheduler', 'SequentialScheduler', 'RandomActivationScheduler',
        'Phase', 'PhasedScheduler',
        'ConsumerChoiceModel', 'ModelComponentFactory', 'ModelValidator',
        'ModelEvent', 'EventBus'
    ]


def print_framework_overview():
    """Print an overview of the framework structure"""
    overview = """
Consumer Choice Metamodel Framework structure

consumer_choice_metamodel/
  __init__.py      package imports
  types.py         TriggerType, EvaluationDimension
  agent.py         AgentAttributes, ChoiceModule, ConsumerAgent
  environment.py   Environment, PhysicalAsset, KnowledgeAsset, Network, ...
  information.py   InformationFilter, InformationDistorter, Transformer
  schedule.py      SequentialScheduler, RandomActivationScheduler, PhasedScheduler
  model.py         ConsumerChoiceModel
  factory.py       ModelComponentFactory
  utils.py         ModelValidator, ModelEvent, EventBus

Key classes:
- ConsumerAgent: agent with choice-making capabilities
- Environment: container for assets, networks and interaction rules
- ChoiceModule: the agent's decision-making component
- Transformer: information processing between agent and environment
- ConsumerChoiceModel: the simulation model
    """
    print(overview)


if __name__ == "__main__":
    print_framework_overview()
