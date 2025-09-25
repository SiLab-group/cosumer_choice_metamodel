# Consumer Choice Metamodel Framework
This package provides abstract base classes and interfaces for implementing
the Consumer Choice Metamodel as described in "Consumer Choice Metamodel: 
A Conceptual Validation Approach" by Amy Liffey et al. 

- Add figure here

[1] cite 

## Project Structure

```bash
consumer_choice_metamodel/
├── __init__.py                 # Package initialization and main imports
├── types.py                    # Base enums and type definitions
├── agent.py                    # Agent-related classes
├── environment.py              # Environment and asset classes  
├── information.py              # Information processing classes
├── model.py                    # Main model class
├── factory.py                  # Factory pattern classes
├── utils.py                    # Validation and event system utilities
└── consumer_choice_metamodel.py # Main entry point (backward compatibility)
```

## Module Breakdown

### 📄 `types.py`
- `TriggerType`: Enumeration of decision triggers
- `EvaluationDimension`: Enumeration of choice evaluation dimensions

### 📄 `agent.py`
- `AgentAttributes`: Abstract base for agent characteristics
- `ChoiceModule`: Abstract base for agent decision-making logic
- `ConsumerAgent`: Main agent class with complete behavior

### 📄 `environment.py`
- `PhysicalAsset`: Abstract base for physical objects/technologies
- `KnowledgeAsset`: Abstract base for information objects
- `Network`: Abstract base for agent networks
- `RulesOfInteraction`: Abstract base for interaction rules
- `ExogenousProcess`: Abstract base for external processes
- `Environment`: Main environment container class

### 📄 `information.py`
- `InformationFilter`: Abstract base for filtering information
- `InformationDistorter`: Abstract base for biasing information
- `Transformer`: Manages information flow between agents and environment

### 📄 `model.py`
- `ConsumerChoiceModel`: Main simulation model class

### 📄 `factory.py`
- `ModelComponentFactory`: Abstract factory for creating model components

### 📄 `utils.py`
- `ModelValidator`: Validation utilities for model components
- `ModelEvent`: Event system for model communication
- `EventBus`: Event distribution system

## Usage Examples

### Basic Import Patterns

```python
# Import specific classes
from consumer_choice_metamodel import ConsumerAgent, Environment, AgentAttributes

# Import entire modules
from consumer_choice_metamodel import agent, environment

# Import from specific modules
from consumer_choice_metamodel.agent import ChoiceModule
from consumer_choice_metamodel.environment import PhysicalAsset
```

### Implementation Template

```python
from consumer_choice_metamodel import (
    ConsumerChoiceModel, ConsumerAgent, Environment,
    AgentAttributes, ChoiceModule, PhysicalAsset
)

class MyAgentAttributes(AgentAttributes):
    def __init__(self, agent_id: str, income: float, age: int):
        super().__init__(agent_id)
        self.income = income
        self.age = age
    
    def get_psychological_attributes(self):
        return {
            'risk_aversion': 0.5,
            'environmental_concern': 0.7,
            'price_sensitivity': 0.8
        }
    
    def get_socioeconomic_attributes(self):
        return {
            'income': self.income,
            'age': self.age
        }
    
    def get_stock_variables(self):
        return {'car': None, 'house': 'apartment'}
    
    def update_attributes(self, changes):
        for key, value in changes.items():
            setattr(self, key, value)

# Implement other required classes...
```
