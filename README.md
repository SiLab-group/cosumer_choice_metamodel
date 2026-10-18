# Consumer Choice Metamodel

This package provides abstract base classes and interfaces for implementing the Consumer Choice Framework described in [Liffey et al. (2026)](https://doi.org/10.1007/978-3-032-31712-4_27), "A Conceptual Validation of the Consumer Choice Framework", *Advances in Social Simulation* (SSC 2025), Springer Proceedings in Complexity.

## Overview

The Consumer Choice Metamodel is a framework for building consumer behavior simulation models. This package provides the core infrastructure and abstract classes needed to implement agent-based models of consumer decision-making processes.


## Installation

### Development Installation

For development or to contribute to the project:

1. Clone the repository:
```bash
git clone https://github.com/SiLab-group/cosumer_choice_metamodel.git
cd cosumer_choice_metamodel
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install in editable mode:
```bash
pip install -e .
```

### Dependencies

The core package uses only the Python standard library. Optional extras: `dev` (testing and linting), `docs` and `simulation` (numpy, pandas, networkx, matplotlib, scipy), e.g. `pip install -e ".[dev]"`.

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
- `PhasedScheduler` / `Phase`: Named stages run one after another, each across all agents; a `Phase` with `per_agent=False` runs once per step

### Model and Factory Classes
- `ConsumerChoiceModel`: Main simulation model class; owns the seeded `rng`, the `scheduler` and the `event_bus`
- `ModelComponentFactory`: Abstract factory for creating model components

### Utilities
- `ModelValidator`: Validation utilities for model components, with configurable required socio-economic attributes
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
- **Validation**: `ModelValidator.validate_model(model, required_socioeconomic=["owner"])` checks the environment and every agent. By default it requires `income` and `age`.
- **Defaults**: `ChoiceModule.add_trigger()`, `ConsumerChoiceModel.collect_data()` and `run_simulation()` have default implementations; override them as needed.

## Running the tests

```bash
pip install -e ".[dev]"
pytest
```

## Quick Start

A model implements the abstract classes and runs through `ConsumerChoiceModel`. This example
simulates solar panel adoption: each household weighs affordability and environmental benefit,
and the panel price falls by 5% per step.

```python
from dataclasses import dataclass

from consumer_choice_metamodel import (
    AgentAttributes, ChoiceModule, ConsumerAgent, ConsumerChoiceModel,
    Environment, EvaluationDimension, PhysicalAsset, Transformer,
)


@dataclass
class HouseholdAttributes(AgentAttributes):
    income: float = 50000.0
    age: int = 40
    environmental_concern: float = 0.5
    has_solar: bool = False

    def get_psychological_attributes(self):
        return {"environmental_concern": self.environmental_concern}

    def get_socioeconomic_attributes(self):
        return {"income": self.income, "age": self.age}

    def get_stock_variables(self):
        return {"has_solar": self.has_solar}

    def update_attributes(self, changes):
        for key, value in changes.items():
            setattr(self, key, value)


@dataclass(eq=False)
class SolarPanel(PhysicalAsset):
    price: float = 15000.0

    def get_availability(self):
        return 1.0

    def get_cost(self):
        return self.price

    def get_properties(self):
        return {"environmental_benefit": 0.9}


class SolarChoice(ChoiceModule):
    def evaluate_option(self, option, evaluation_context):
        a = self.agent_attributes
        return {
            EvaluationDimension.FINANCIAL: a.income / option.get_cost() / 10,
            EvaluationDimension.ENVIRONMENTAL:
                a.environmental_concern * option.get_properties()["environmental_benefit"],
        }

    def aggregate_evaluations(self, evaluations):
        return sum(evaluations.values())

    def make_choice(self, options, choice_context):
        scored = [(self.aggregate_evaluations(self.evaluate_option(o, choice_context)), o)
                  for o in options]
        utility, best = max(scored, key=lambda s: s[0], default=(0.0, None))
        return best if utility > choice_context["threshold"] else None


class Household(ConsumerAgent):
    def perceive_environment(self, environment):
        return {"options": environment.get_available_options(self.agent_id, {})}

    def make_decisions(self, environment):
        if self.attributes.has_solar:
            return []
        options = self.memory[self.PERCEPTION_KEY]["options"]
        choice = self.choice_module.make_choice(options, {"threshold": 1.0})
        return [{"action": "adopt", "technology": choice.name}] if choice else []

    def communicate(self, environment, decisions):
        for decision in decisions:
            environment.add_communication({"agent_id": self.agent_id, **decision})

    def update_state(self, decisions, environment_feedback):
        if decisions:
            self.attributes.update_attributes({"has_solar": True})


class Market(Environment):
    def add_physical_asset(self, asset):
        self.physical_assets.append(asset)

    def add_knowledge_asset(self, asset):
        self.knowledge_assets.append(asset)

    def get_available_options(self, agent_id, context):
        return list(self.physical_assets)

    def get_relevant_knowledge(self, agent_id, context):
        return list(self.knowledge_assets)

    def update_environment(self, current_time):
        for asset in self.physical_assets:
            asset.price *= 0.95

    def add_communication(self, communication):
        self.communications.append(communication)


class SolarModel(ConsumerChoiceModel):
    def initialize_model(self, parameters):
        self.model_parameters = parameters
        self.environment = self.create_environment(parameters)
        self.add_agents(self.create_agents(parameters["agent_count"], parameters))

    def create_environment(self, environment_config):
        market = Market()
        market.add_physical_asset(SolarPanel("solar_panel"))
        return market

    def create_agents(self, agent_count, agent_config):
        agents = []
        for i in range(agent_count):
            attrs = HouseholdAttributes(f"household_{i}",
                                        income=self.rng.uniform(30000, 120000),
                                        environmental_concern=self.rng.random())
            agents.append(Household(attrs.agent_id, attrs,
                                    SolarChoice(attrs.agent_id, attrs), Transformer([], [])))
        return agents

    def collect_data(self):
        return {"time": self.current_time,
                "adopters": sum(a.attributes.has_solar for a in self.agents)}


model = SolarModel(seed=42)
model.initialize_model({"agent_count": 100})
results = model.run_simulation(10)
print(results["final_stats"])     # {'time': 10, 'adopters': ...}
```

`examples/example_adoption.py` is a larger example with information filters, cognitive
biases, a spatial network, policy incentives and decision triggers.

## License

Apache License 2.0, see [LICENSE](LICENSE).

## Citation

If you use this package in your research, please cite:

> Liffey, A., Günther, A., Firoozyalizadeh, S., Elassal, O., Abrell, J., Weigt, H.,
> van der Kam, M., Panos, E., Schubert, I., Hahnel, U., Schumann, R. (2026).
> A Conceptual Validation of the Consumer Choice Framework.
> In: Verhagen, H., Chappin, E., Scholz, G. (eds), *Advances in Social Simulation* (SSC 2025),
> pp. 366–376. Springer Proceedings in Complexity. Springer, Cham.
> [https://doi.org/10.1007/978-3-032-31712-4_27](https://doi.org/10.1007/978-3-032-31712-4_27)

```bibtex
@inproceedings{liffey2026consumerchoice,
  author    = {Liffey, Amy and G{\"u}nther, Anne and Firoozyalizadeh, Shadi and
               Elassal, Ola and Abrell, Jan and Weigt, Hannes and
               van der Kam, Marten and Panos, Evangelos and Schubert, Iljana and
               Hahnel, Ulf and Schumann, Ren{\'e}},
  editor    = {Verhagen, Harko and Chappin, Emile and Scholz, Geeske},
  title     = {A Conceptual Validation of the Consumer Choice Framework},
  booktitle = {Advances in Social Simulation},
  series    = {Springer Proceedings in Complexity},
  publisher = {Springer Nature Switzerland},
  address   = {Cham},
  year      = {2026},
  pages     = {366--376},
  isbn      = {978-3-032-31712-4},
  doi       = {10.1007/978-3-032-31712-4_27}
}
```


