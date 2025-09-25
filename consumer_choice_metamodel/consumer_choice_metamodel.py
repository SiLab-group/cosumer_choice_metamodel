"""
Consumer Choice Metamodel Framework - Main Entry Point
======================================================

This module serves as the main entry point for the Consumer Choice Metamodel
framework. It imports and re-exports all components from the modular structure
for backward compatibility and convenience.

For new projects, it's recommended to import specific modules directly:
    from consumer_choice_metamodel.agent import ConsumerAgent
    from consumer_choice_metamodel.environment import Environment
    
Or use the package-level imports:
    from consumer_choice_metamodel import ConsumerAgent, Environment
"""

# Re-export everything from the package for convenience
from consumer_choice_metamodel import *

# Additional convenience imports that maintain the original structure
from consumer_choice_metamodel.types import TriggerType, EvaluationDimension
from consumer_choice_metamodel.agent import AgentAttributes, ChoiceModule, ConsumerAgent
from consumer_choice_metamodel.environment import (
    PhysicalAsset, KnowledgeAsset, Network, RulesOfInteraction, 
    ExogenousProcess, Environment
)
from consumer_choice_metamodel.information import (
    InformationFilter, InformationDistorter, Transformer
)
from consumer_choice_metamodel.model import ConsumerChoiceModel
from consumer_choice_metamodel.factory import ModelComponentFactory
from consumer_choice_metamodel.utils import ModelValidator, ModelEvent, EventBus

# Backward compatibility - recreate the original module structure
__doc__ = """
Abstract Consumer Choice Metamodel Framework
===========================================

This module provides abstract base classes and interfaces for implementing
the Consumer Choice Metamodel as described in "Consumer Choice Metamodel: 
A Conceptual Validation Approach" by Amy Liffey et al.

The framework defines the core architectural components without specific
implementation details, allowing for flexible adaptation across different
domains and modeling platforms.
"""

# Version information
__version__ = "1.0.0"
__author__ = "Consumer Choice Metamodel Framework"

# For users who want to check what's available
def list_available_classes():
    """Return a list of all available classes in the framework"""
    return [
        'TriggerType', 'EvaluationDimension', 'AgentAttributes', 'ChoiceModule',
        'ConsumerAgent', 'PhysicalAsset', 'KnowledgeAsset', 'Network',
        'RulesOfInteraction', 'ExogenousProcess', 'Environment',
        'InformationFilter', 'InformationDistorter', 'Transformer',
        'ConsumerChoiceModel', 'ModelComponentFactory', 'ModelValidator',
        'ModelEvent', 'EventBus'
    ]


def print_framework_overview():
    """Print an overview of the framework structure"""
    overview = """
Consumer Choice Metamodel Framework Structure:
============================================

📁 consumer_choice_metamodel/
├── 📄 __init__.py              # Package initialization and main imports
├── 📄 types.py                 # Base enums (TriggerType, EvaluationDimension)
├── 📄 agent.py                 # Agent classes (AgentAttributes, ChoiceModule, ConsumerAgent)
├── 📄 environment.py           # Environment and assets (Environment, PhysicalAsset, KnowledgeAsset, etc.)
├── 📄 information.py           # Information processing (InformationFilter, Transformer, etc.)
├── 📄 model.py                 # Main model class (ConsumerChoiceModel)
├── 📄 factory.py               # Factory pattern (ModelComponentFactory)
└── 📄 utils.py                 # Utilities (ModelValidator, EventBus, etc.)

Key Classes:
- ConsumerAgent: Main agent class with choice-making capabilities
- Environment: Container for assets, networks, and interaction rules
- ChoiceModule: Agent's decision-making component
- Transformer: Information processing between agent and environment
- ConsumerChoiceModel: Main simulation model
    """
    print(overview)


if __name__ == "__main__":
    print_framework_overview()
