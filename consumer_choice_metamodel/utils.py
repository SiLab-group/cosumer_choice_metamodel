"""
Validation and event system utilities for the Consumer Choice Metamodel
"""

from typing import Dict, List, Any, Optional, Callable

from .agent import AgentAttributes, ChoiceModule
from .environment import Environment


class ModelValidator:
    """Validates model configuration and components"""
    
    @staticmethod
    def validate_agent_attributes(attributes: AgentAttributes) -> List[str]:
        """Validate agent attributes configuration"""
        errors = []
        
        # Check psychological attributes are in valid range
        psych_attrs = attributes.get_psychological_attributes()
        for attr, value in psych_attrs.items():
            if not 0 <= value <= 1:
                errors.append(f"Psychological attribute {attr} must be between 0 and 1")
        
        # Check required socioeconomic attributes exist
        socio_attrs = attributes.get_socioeconomic_attributes()
        required_attrs = ['income', 'age']  # Can be extended
        for attr in required_attrs:
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


class ModelEvent:
    """Base class for model events"""
    
    def __init__(self, event_type: str, source: str, data: Dict[str, Any]):
        self.event_type = event_type
        self.source = source
        self.data = data
        self.timestamp: Optional[int] = None


class EventBus:
    """Simple event bus for model-wide communication"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable[[ModelEvent], None]]] = {}
    
    def subscribe(self, event_type: str, callback: Callable[[ModelEvent], None]) -> None:
        """Subscribe to event type"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
    
    def publish(self, event: ModelEvent) -> None:
        """Publish event to subscribers"""
        if event.event_type in self.subscribers:
            for callback in self.subscribers[event.event_type]:
                callback(event)
