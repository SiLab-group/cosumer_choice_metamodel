"""
Base types and enumerations for the Consumer Choice Metamodel
"""

from enum import Enum


class TriggerType(Enum):
    """Base trigger types that can initiate agent decision-making"""
    REPLACEMENT = "replacement"
    PRICE_CHANGE = "price_change"
    POLICY_CHANGE = "policy_change" 
    SOCIAL_INFLUENCE = "social_influence"
    INFRASTRUCTURE = "infrastructure"
    CUSTOM = "custom"


class EvaluationDimension(Enum):
    """Base evaluation dimensions for choice assessment"""
    FINANCIAL = "financial"
    ENVIRONMENTAL = "environmental"
    SOCIAL = "social"
    PERFORMANCE = "performance"
    CONVENIENCE = "convenience"
    CUSTOM = "custom"
