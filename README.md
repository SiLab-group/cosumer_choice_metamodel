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
├── schedule.py                    # Agent activation schedules
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

### Scheduling
- `Scheduler`: Abstract base deciding how agents are activated in a step
- `SequentialScheduler`: Every agent once, in list order (default)
- `RandomActivationScheduler`: Every agent once, in a new random order each step
- `PhasedScheduler` / `Phase`: Named stages run one after another, each across all agents (or once per step for bookkeeping)

### Model and Factory Classes
- `ConsumerChoiceModel`: Main simulation model class; owns the seeded `rng`, the `scheduler` and the `event_bus`
- `ModelComponentFactory`: Abstract factory for creating model components

### Utilities
- `ModelValidator`: Validation utilities for model components (required socio-economic attributes are configurable)
- `ModelEvent`: Event system for model communication
- `EventBus`: Event distribution system with timestamps, wildcard subscriptions and `unsubscribe`

## Scheduling, randomness and events

```python
from consumer_choice_metamodel import ConsumerChoiceModel, PhasedScheduler, Phase

class MyModel(ConsumerChoiceModel):
    ...  # initialize_model, create_agents, create_environment

def consider_bundles(agent, model): ...
def consider_single_technologies(agent, model): ...
def recount_adopters(model): ...

model = MyModel(
    seed=42,                                   # same seed -> same run
    scheduler=PhasedScheduler([
        Phase("bundles", consider_bundles),
        Phase("recount", recount_adopters, per_agent=False),
        Phase("singles", consider_single_technologies),
    ]),                                        # default: SequentialScheduler()
)
```

- **Randomness**: draw every random number from `model.rng` (a `random.Random`). Register agents with `model.add_agents(agents)` so that `agent.rng` is the same generator, and set `factory.rng = model.rng` when a factory creates stochastic components.
- **Perception**: `ConsumerAgent.step()` stores the result of `perceive_environment()` in `agent.memory[ConsumerAgent.PERCEPTION_KEY]` before calling `make_decisions()`.
- **Events**: `model.publish("adopted", agent.agent_id, {...})` sends a `ModelEvent` stamped with `model.current_time`. Subscribe to one event type or to `EventBus.ALL_EVENTS` (`"*"`).
- **Validation**: `ModelValidator.validate_model(model, required_socioeconomic=["owner"])` checks the environment and every agent; the default still requires `income` and `age`.
- **Defaults**: `ChoiceModule.add_trigger()`, `ConsumerChoiceModel.collect_data()` and `run_simulation()` now have working default implementations that you can override.

## Running the tests

```bash
pip install -e ".[dev]"
pytest
```

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

#### 1.1.0
- Added `schedule.py` with `Scheduler`, `SequentialScheduler`, `RandomActivationScheduler`, `PhasedScheduler` and `Phase`; `ConsumerChoiceModel.step()` delegates agent activation to `model.scheduler` (default keeps the previous in-order behaviour).
- `ConsumerChoiceModel(seed=..., scheduler=...)` owns a seeded `rng`, with `reseed()` and `add_agents()`; `ConsumerAgent` and `ModelComponentFactory` gained an `rng` attribute.
- `ConsumerAgent.step()` keeps the perception in `memory["perception"]` (it was previously discarded).
- `ModelValidator.validate_agent_attributes()` takes `required_socioeconomic`; new `validate_model()`.
- `EventBus` gained `unsubscribe()`, wildcard subscriptions and automatic timestamps; the model owns an `event_bus` and a `publish()` helper. `ModelEvent` accepts a `timestamp`.
- Default implementations for `ChoiceModule.add_trigger()`, `ConsumerChoiceModel.collect_data()` and `run_simulation()`.
- Added tests; aligned the version number across `pyproject.toml`, `setup.py` and the package.

Existing subclasses keep working unchanged, provided that a subclass which overrides `ConsumerChoiceModel.__init__` calls `super().__init__()` (`step()` now needs `self.scheduler`).

---

*This package is maintained by the SiLab-group research team.*