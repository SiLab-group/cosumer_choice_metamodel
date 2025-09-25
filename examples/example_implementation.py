"""
Example implementation of the Consumer Choice Metamodel
This file shows how to implement the abstract base classes
"""

from consumer_choice_metamodel import (
    ConsumerAgent, AgentAttributes, ChoiceModule, Environment,
    PhysicalAsset, KnowledgeAsset, ConsumerChoiceModel,
    TriggerType, EvaluationDimension
)
from typing import Dict, List, Any, Optional


class ExampleAgentAttributes(AgentAttributes):
    """Example implementation of agent attributes"""
    
    def __init__(self, agent_id: str, income: float = 50000, age: int = 35):
        super().__init__(agent_id)
        self.income = income
        self.age = age
        self.risk_aversion = 0.5
        self.environmental_concern = 0.7
    
    def get_psychological_attributes(self) -> Dict[str, float]:
        return {
            'risk_aversion': self.risk_aversion,
            'environmental_concern': self.environmental_concern,
            'price_sensitivity': 0.8
        }
    
    def get_socioeconomic_attributes(self) -> Dict[str, float]:
        return {
            'income': self.income,
            'age': self.age
        }
    
    def get_stock_variables(self) -> Dict[str, Any]:
        return {
            'car': None,
            'house': 'apartment',
            'savings': self.income * 0.1
        }
    
    def update_attributes(self, changes: Dict[str, Any]) -> None:
        for key, value in changes.items():
            if hasattr(self, key):
                setattr(self, key, value)


class ExampleChoiceModule(ChoiceModule):
    """Example implementation of choice module"""
    
    def add_trigger(self, trigger_type: TriggerType, context: Dict[str, Any]) -> None:
        self.pending_triggers.append({
            'type': trigger_type,
            'context': context,
            'timestamp': context.get('time', 0)
        })
    
    def evaluate_option(self, option: PhysicalAsset, 
                       evaluation_context: Dict[str, Any]) -> Dict[EvaluationDimension, float]:
        # Simple evaluation based on cost and agent attributes
        cost = option.get_cost()
        attributes = self.agent_attributes.get_psychological_attributes()
        
        return {
            EvaluationDimension.FINANCIAL: max(0, 1 - cost / 100000),  # Normalize cost
            EvaluationDimension.ENVIRONMENTAL: attributes['environmental_concern'],
            EvaluationDimension.CONVENIENCE: 0.8,  # Default convenience score
        }
    
    def aggregate_evaluations(self, evaluations: Dict[EvaluationDimension, float]) -> float:
        # Simple weighted sum
        weights = {
            EvaluationDimension.FINANCIAL: 0.4,
            EvaluationDimension.ENVIRONMENTAL: 0.3,
            EvaluationDimension.CONVENIENCE: 0.3
        }
        
        total_utility = 0.0
        for dimension, score in evaluations.items():
            if dimension in weights:
                total_utility += weights[dimension] * score
        
        return total_utility
    
    def make_choice(self, options: List[PhysicalAsset], 
                   choice_context: Dict[str, Any]) -> Optional[PhysicalAsset]:
        if not options:
            return None
        
        best_option = None
        best_utility = -1
        
        for option in options:
            evaluations = self.evaluate_option(option, choice_context)
            utility = self.aggregate_evaluations(evaluations)
            
            if utility > best_utility:
                best_utility = utility
                best_option = option
        
        return best_option


# Add more example implementations as needed...

if __name__ == "__main__":
    print("Example Consumer Choice Metamodel Implementation")
    print("This file shows how to implement the abstract base classes")
