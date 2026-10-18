"""
Agent activation schedules for the Consumer Choice Metamodel
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional, Sequence

if TYPE_CHECKING:
    from .agent import ConsumerAgent
    from .model import ConsumerChoiceModel


class Scheduler(ABC):
    """Abstract base class deciding how agents are activated within one model step"""

    @abstractmethod
    def run(self, model: 'ConsumerChoiceModel', agents: Sequence['ConsumerAgent']) -> None:
        """Activate agents for one step"""
        pass


class SequentialScheduler(Scheduler):
    """Activate every agent once, in list order (default)"""

    def run(self, model: 'ConsumerChoiceModel', agents: Sequence['ConsumerAgent']) -> None:
        for agent in list(agents):
            agent.step(model.environment)


class RandomActivationScheduler(Scheduler):
    """Activate every agent once, in a new random order each step (NetLogo-style `ask`)"""

    def run(self, model: 'ConsumerChoiceModel', agents: Sequence['ConsumerAgent']) -> None:
        order = list(agents)
        model.rng.shuffle(order)
        for agent in order:
            agent.step(model.environment)


@dataclass
class Phase:
    """
    One stage of a phased step.

    With per_agent=True the action is called as action(agent, model) for every
    agent; with per_agent=False it is called once as action(model), which suits
    bookkeeping such as recounting adopters between stages.
    """
    name: str
    action: Callable[..., None]
    per_agent: bool = True


class PhasedScheduler(Scheduler):
    """
    Run named phases in sequence, each across all agents before the next starts.

    Use this when every agent must finish one kind of decision before any agent
    starts the next (e.g. bundle purchases, then single technologies). By default
    the agent order is shuffled once per step and shared by all phases; set
    reshuffle_each_phase=True to draw a new order for every phase.
    """

    def __init__(self, phases: Sequence[Phase], shuffle: bool = True,
                 reshuffle_each_phase: bool = False):
        self.phases: List[Phase] = list(phases)
        self.shuffle = shuffle
        self.reshuffle_each_phase = reshuffle_each_phase
        self.current_phase: Optional[str] = None

    def run(self, model: 'ConsumerChoiceModel', agents: Sequence['ConsumerAgent']) -> None:
        order = self._order(model, agents)
        for phase in self.phases:
            self.current_phase = phase.name
            if self.reshuffle_each_phase:
                order = self._order(model, agents)
            if phase.per_agent:
                for agent in order:
                    phase.action(agent, model)
            else:
                phase.action(model)
        self.current_phase = None

    def _order(self, model: 'ConsumerChoiceModel',
               agents: Sequence['ConsumerAgent']) -> List['ConsumerAgent']:
        order = list(agents)
        if self.shuffle:
            model.rng.shuffle(order)
        return order
