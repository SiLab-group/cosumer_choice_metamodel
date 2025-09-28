# Consumer Choice Metamodel

This package provides abstract base classes and interfaces for implementing the Consumer Choice Metamodel as described in "Consumer Choice Metamodel: A Conceptual Validation Approach" by Amy Liffey et al.

## Overview

The Consumer Choice Metamodel is a framework for building consumer behavior simulation models. This package provides the core infrastructure and abstract classes needed to implement agent-based models of consumer decision-making processes.


## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Install from GitHub

Since this package is hosted on GitHub, you can install it directly using pip:

```bash
pip install git+https://github.com/SiLab-group/cosumer_choice_metamodel.git
```

### Development Installation

For development or to contribute to the project:

1. Clone the repository:
```bash
git clone https://github.com/SiLab-group/cosumer_choice_metamodel.git
cd cosumer_choice_metamodel
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install in editable mode:
```bash
pip install -e .
```

### Dependencies

The package likely depends on standard Python libraries. If there are additional requirements, they will be automatically installed during the pip installation process.

## Package Structure

```
consumer_choice_metamodel/
├── __init__.py                    # Package initialization and main imports
├── types.py                       # Base enums and type definitions
├── agent.py                       # Agent-related classes
├── environment.py                 # Environment and asset classes
├── information.py                 # Information processing classes
├── model.py                       # Main model class
├── factory.py                     # Factory pattern classes
├── utils.py                       # Validation and event system utilities
└── consumer_choice_metamodel.py   # Main entry point (backward compatibility)
```

## Core Components

### Enumerations and Types
- `TriggerType`: Enumeration of decision triggers
- `EvaluationDimension`: Enumeration of choice evaluation dimensions

### Agent Classes
- `AgentAttributes`: Abstract base for agent characteristics
- `ChoiceModule`: Abstract base for agent decision-making logic
- `ConsumerAgent`: Main agent class with complete behavior

### Environment Classes
- `PhysicalAsset`: Abstract base for physical objects/technologies
- `KnowledgeAsset`: Abstract base for information objects
- `Network`: Abstract base for agent networks
- `RulesOfInteraction`: Abstract base for interaction rules
- `ExogenousProcess`: Abstract base for external processes
- `Environment`: Main environment container class

### Information Processing
- `InformationFilter`: Abstract base for filtering information
- `InformationDistorter`: Abstract base for biasing information
- `Transformer`: Manages information flow between agents and environment

### Model and Factory Classes
- `ConsumerChoiceModel`: Main simulation model class
- `ModelComponentFactory`: Abstract factory for creating model components

### Utilities
- `ModelValidator`: Validation utilities for model components
- `ModelEvent`: Event system for model communication
- `EventBus`: Event distribution system

## Quick Start

### Basic Usage

```python
from consumer_choice_metamodel import (
    ConsumerChoiceModel, ConsumerAgent, Environment,
    AgentAttributes, ChoiceModule, PhysicalAsset
)

# Import specific classes
from consumer_choice_metamodel import ConsumerAgent, Environment, AgentAttributes

# Import entire modules
from consumer_choice_metamodel import agent, environment

# Import from specific modules
from consumer_choice_metamodel.agent import ChoiceModule
from consumer_choice_metamodel.environment import PhysicalAsset
```

### Example Implementation

```python
from consumer_choice_metamodel import AgentAttributes

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

# Create an agent with custom attributes
agent_attrs = MyAgentAttributes("agent_001", income=50000, age=35)
agent = ConsumerAgent(attributes=agent_attrs)
```

## Documentation

For more detailed documentation and examples, please refer to the research paper:
"Consumer Choice Metamodel: A Conceptual Validation Approach" by Amy Liffey et al.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

Please check the repository for license information.

## Support

For questions, issues, or contributions, please:
- Open an issue on [GitHub](https://github.com/SiLab-group/cosumer_choice_metamodel/issues)
- Contact the SiLab-group team

## Citation

If you use this package in your research, please cite:

```
Liffey, A. et al. "Consumer Choice Metamodel: A Conceptual Validation Approach"
```

## Changelog

### Version History
- Check the repository's releases page for detailed version history and changes

---

*This package is maintained by the SiLab-group research team.*