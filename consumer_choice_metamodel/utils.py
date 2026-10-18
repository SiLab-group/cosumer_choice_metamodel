"""
Validation and event system utilities for the Consumer Choice Metamodel
"""

from typing import TYPE_CHECKING, Dict, Iterable, List, Any, Optional, Callable

from .agent import AgentAttributes, ChoiceModule
from .environment import Environment

if TYPE_CHECKING:
    from .model import ConsumerChoiceModel


# Socio-economic attributes required by default; pass required_socioeconomic=()
# (or your own list) when a model's population data does not have them.
DEFAULT_REQUIRED_SOCIOECONOMIC = ('income', 'age')


class ModelValidator:
    """Validates model configuration and components"""

    @staticmethod
    def validate_agent_attributes(
            attributes: AgentAttributes,
            required_socioeconomic: Optional[Iterable[str]] = None) -> List[str]:
        """Validate agent attributes configuration"""
        errors = []

        # Check psychological attributes are in valid range
        psych_attrs = attributes.get_psychological_attributes()
        for attr, value in psych_attrs.items():
            if not 0 <= value <= 1:
                errors.append(f"Psychological attribute {attr} must be between 0 and 1")

        # Check required socioeconomic attributes exist
        socio_attrs = attributes.get_socioeconomic_attributes()
        if required_socioeconomic is None:
            required_socioeconomic = DEFAULT_REQUIRED_SOCIOECONOMIC
        for attr in required_socioeconomic:
            if attr not in socio_attrs:
                errors.append(f"Required socioeconomic attribute {attr} missing")

        return errors

    @staticmethod
    def validate_choice_module(choice_module: ChoiceModule) -> List[str]:
        """Validate choice module configuration"""
        errors = []

        # Validation logic for choice module
        # Can be extended based on specific requirements

        return errors

    @staticmethod
    def validate_environment(environment: Environment) -> List[str]:
        """Validate environment configuration"""
        errors = []

        if len(environment.physical_assets) == 0:
            errors.append("Environment must have at least one physical asset")

        if len(environment.networks) == 0:
            errors.append("Environment should have at least one network for agent interactions")

        return errors

    @staticmethod
    def validate_model(model: 'ConsumerChoiceModel',
                       required_socioeconomic: Optional[Iterable[str]] = None) -> List[str]:
        """Validate the environment and every agent of a model"""
        if required_socioeconomic is not None:
            required_socioeconomic = tuple(required_socioeconomic)
        errors = []
        if model.environment is None:
            errors.append("Model has no environment")
        else:
            errors.extend(ModelValidator.validate_environment(model.environment))
        for agent in model.agents:
            for error in ModelValidator.validate_agent_attributes(
                    agent.attributes, required_socioeconomic):
                errors.append(f"{agent.agent_id}: {error}")
            for error in ModelValidator.validate_choice_module(agent.choice_module):
                errors.append(f"{agent.agent_id}: {error}")
        return errors


class ModelEvent:
    """Base class for model events"""

    def __init__(self, event_type: str, source: str, data: Dict[str, Any],
                 timestamp: Optional[int] = None):
        self.event_type = event_type
        self.source = source
        self.data = data
        self.timestamp: Optional[int] = timestamp


class EventBus:
    """
    Simple event bus for model-wide communication.

    Subscribers to ALL_EVENTS ("*") receive every event. When a clock is given,
    events published without a timestamp are stamped with its current value.
    """

    ALL_EVENTS = "*"

    def __init__(self, clock: Optional[Callable[[], int]] = None):
        self.subscribers: Dict[str, List[Callable[[ModelEvent], None]]] = {}
        self.clock = clock

    def subscribe(self, event_type: str, callback: Callable[[ModelEvent], None]) -> None:
        """Subscribe to event type"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: Callable[[ModelEvent], None]) -> None:
        """Remove a subscription; does nothing if it was not subscribed"""
        callbacks = self.subscribers.get(event_type, [])
        if callback in callbacks:
            callbacks.remove(callback)

    def publish(self, event: ModelEvent) -> None:
        """Publish event to subscribers (in subscription order, specific before wildcard)"""
        if event.timestamp is None and self.clock is not None:
            event.timestamp = self.clock()
        for event_type in dict.fromkeys((event.event_type, self.ALL_EVENTS)):
            # copy so callbacks may (un)subscribe while handling the event
            for callback in list(self.subscribers.get(event_type, [])):
                callback(event)
