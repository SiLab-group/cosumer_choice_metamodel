"""
Minimal concrete implementations of the metamodel ABCs, shared by the tests.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest

from consumer_choice_metamodel import (
    AgentAttributes, ChoiceModule, ConsumerAgent, ConsumerChoiceModel,
    Environment, EvaluationDimension, KnowledgeAsset, Network, PhysicalAsset,
    Transformer,
)


@dataclass
class SimpleAttributes(AgentAttributes):
    income: float = 50000.0
    age: int = 40
    concern: float = 0.5
    owns: bool = False

    def get_psychological_attributes(self) -> Dict[str, float]:
        return {'concern': self.concern}

    def get_socioeconomic_attributes(self) -> Dict[str, Any]:
        return {'income': self.income, 'age': self.age}

    def get_stock_variables(self) -> Dict[str, Any]:
        return {'owns': self.owns}

    def update_attributes(self, changes: Dict[str, Any]) -> None:
        for key, value in changes.items():
            setattr(self, key, value)


class SimpleAsset(PhysicalAsset):
    def get_availability(self) -> float:
        return 1.0

    def get_cost(self) -> float:
        return 100.0

    def get_properties(self) -> Dict[str, Any]:
        return {}


class SimpleChoice(ChoiceModule):
    def evaluate_option(self, option, evaluation_context):
        return {EvaluationDimension.FINANCIAL: 1.0}

    def aggregate_evaluations(self, evaluations):
        return sum(evaluations.values())

    def make_choice(self, options, choice_context):
        return options[0] if options else None


class SimpleNetwork(Network):
    def __init__(self):
        self.ids: List[str] = []

    def get_neighbors(self, agent_id: str) -> List[str]:
        return [i for i in self.ids if i != agent_id]

    def add_agent(self, agent_id: str, properties: Dict[str, Any]) -> None:
        self.ids.append(agent_id)

    def remove_agent(self, agent_id: str) -> None:
        self.ids.remove(agent_id)

    def get_network_properties(self, agent_id: str) -> Dict[str, Any]:
        return {'neighbors': self.get_neighbors(agent_id)}


class SimpleEnvironment(Environment):
    def add_physical_asset(self, asset: PhysicalAsset) -> None:
        self.physical_assets.append(asset)

    def add_knowledge_asset(self, asset: KnowledgeAsset) -> None:
        self.knowledge_assets.append(asset)

    def get_available_options(self, agent_id, context):
        return list(self.physical_assets)

    def get_relevant_knowledge(self, agent_id, context):
        return list(self.knowledge_assets)

    def update_environment(self, current_time: int) -> None:
        pass

    def add_communication(self, communication: Dict[str, Any]) -> None:
        self.communications.append(communication)


class SimpleAgent(ConsumerAgent):
    """Records every call, and draws one random number per step."""

    def __init__(self, agent_id: str, log: Optional[list] = None):
        attrs = SimpleAttributes(agent_id)
        super().__init__(agent_id, attrs, SimpleChoice(agent_id, attrs), Transformer([], []))
        self.log = log if log is not None else []
        self.draws: List[float] = []

    def perceive_environment(self, environment):
        return {'time_seen': len(self.log)}

    def make_decisions(self, environment):
        self.log.append(('decide', self.agent_id))
        self.draws.append(self.rng.random())
        return [{'perception': self.memory.get(self.PERCEPTION_KEY)}]

    def communicate(self, environment, decisions):
        pass

    def update_state(self, decisions, environment_feedback):
        self.memory['feedback'] = environment_feedback


class SimpleModel(ConsumerChoiceModel):
    def initialize_model(self, parameters: Dict[str, Any]) -> None:
        self.model_parameters = parameters
        self.environment = self.create_environment({})
        self.add_agents(self.create_agents(parameters.get('agent_count', 5), {}))
        for agent in self.agents:
            self.environment.networks[0].add_agent(agent.agent_id, {})

    def create_agents(self, agent_count, agent_config):
        return [SimpleAgent(f"a{i}") for i in range(agent_count)]

    def create_environment(self, environment_config):
        env = SimpleEnvironment()
        env.add_physical_asset(SimpleAsset('thing'))
        env.networks.append(SimpleNetwork())
        return env


@pytest.fixture
def make_model():
    def _make(seed=None, scheduler=None, agent_count=5):
        model = SimpleModel(seed=seed, scheduler=scheduler)
        model.initialize_model({'agent_count': agent_count})
        return model
    return _make
