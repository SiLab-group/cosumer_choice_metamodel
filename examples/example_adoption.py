from dataclasses import dataclass
import random
import math

from consumer_choice_metamodel import (
    ConsumerAgent, AgentAttributes, ChoiceModule, Environment,
    PhysicalAsset, KnowledgeAsset, ConsumerChoiceModel,
    TriggerType, EvaluationDimension, InformationFilter, InformationDistorter,
    Network, RulesOfInteraction, ExogenousProcess, Transformer
)
from typing import Dict, List, Any, Optional, Union

@dataclass
class HouseholdAttributes(AgentAttributes):
    """Concrete implementation for household energy decisions"""

    # Socio-economic
    age: int = 45
    income: float = 60000.0
    education_level: int = 3  # 1-5 scale
    household_size: int = 3
    home_ownership: bool = True

    # Psychological
    environmental_concern: float = 0.5
    risk_tolerance: float = 0.5
    social_norm_sensitivity: float = 0.5
    technology_enthusiasm: float = 0.5
    financial_motivation: float = 0.7

    # Stock variables
    current_heating: str = "gas_boiler"
    current_car: str = "gasoline"
    has_solar: bool = False
    roof_suitable: bool = True

    def get_psychological_attributes(self) -> Dict[str, float]:
        return {
            'environmental_concern': self.environmental_concern,
            'risk_tolerance': self.risk_tolerance,
            'social_norm_sensitivity': self.social_norm_sensitivity,
            'technology_enthusiasm': self.technology_enthusiasm,
            'financial_motivation': self.financial_motivation
        }

    def get_socioeconomic_attributes(self) -> Dict[str, Union[int, float]]:
        return {
            'age': self.age,
            'income': self.income,
            'education_level': self.education_level,
            'household_size': self.household_size
        }

    def get_stock_variables(self) -> Dict[str, Any]:
        return {
            'current_heating': self.current_heating,
            'current_car': self.current_car,
            'has_solar': self.has_solar,
            'roof_suitable': self.roof_suitable,
            'home_ownership': self.home_ownership
        }

    def update_attributes(self, changes: Dict[str, Any]) -> None:
        for key, value in changes.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Update derived attributes
        if 'has_solar' in changes and changes['has_solar']:
            # Solar adoption might increase environmental concern
            self.environmental_concern = min(1.0, self.environmental_concern + 0.1)


class EnergyTechnology(PhysicalAsset):
    """Concrete implementation for energy technologies"""

    def __init__(self, name: str, base_cost: float, efficiency: float,
                 environmental_benefit: float, installation_complexity: int = 3):
        super().__init__(name)
        self.base_cost = base_cost
        self.efficiency = efficiency
        self.environmental_benefit = environmental_benefit
        self.installation_complexity = installation_complexity  # 1-5 scale
        self.market_maturity = 0.5  # Affects availability

    def get_availability(self) -> float:
        # Availability depends on market maturity and installation complexity
        base_availability = self.market_maturity
        complexity_penalty = (self.installation_complexity - 1) * 0.1
        return max(0.1, base_availability - complexity_penalty)

    def get_cost(self) -> float:
        # Cost varies with market conditions (could be modified by exogenous processes)
        return self.base_cost

    def get_properties(self) -> Dict[str, Any]:
        return {
            'efficiency': self.efficiency,
            'environmental_benefit': self.environmental_benefit,
            'installation_complexity': self.installation_complexity,
            'market_maturity': self.market_maturity,
            'payback_period': self._calculate_payback_period()
        }

    def _calculate_payback_period(self) -> float:
        # Simplified payback calculation
        annual_savings = self.base_cost * self.efficiency * 0.1  # 10% of cost as annual savings
        if annual_savings > 0:
            return self.base_cost / annual_savings
        return float('inf')


class EnergyInformation(KnowledgeAsset):
    """Concrete implementation for energy-related information"""

    def __init__(self, topic: str, source: str, reliability: float, content: Dict[str, Any]):
        super().__init__(topic, source)
        self.reliability_score = reliability
        self.information_content = content
        self.age = 0  # Information gets less reliable over time

    def get_reliability(self) -> float:
        # Reliability decreases with age
        age_penalty = self.age * 0.02
        return max(0.1, self.reliability_score - age_penalty)

    def get_content(self) -> Dict[str, Any]:
        return self.information_content

    def age_information(self) -> None:
        """Age the information (call periodically)"""
        self.age += 1


class DomainSpecificFilter(InformationFilter):
    def filter(self, agent_attributes: AgentAttributes,
               knowledge_assets: List[KnowledgeAsset]) -> List[KnowledgeAsset]:
        filtered = []
        attrs = agent_attributes

        for asset in knowledge_assets:
            # Education-based filtering
            if asset.source == 'academic':
                access_prob = 0.2 + attrs.education * 0.2
            elif asset.source == 'media':
                access_prob = 0.8
            else:
                access_prob = 0.5

            if random.random() < access_prob:
                filtered.append(asset)

        return filtered


class CognitiveBiasDistorter(InformationDistorter):
    def distort(self, agent_attributes: AgentAttributes,
                knowledge: KnowledgeAsset) -> KnowledgeAsset:
        # Apply confirmation bias
        distorted_content = knowledge.get_content().copy()

        # Bias information toward agent's predispositions
        if 'benefit_estimate' in distorted_content:
            original = distorted_content['benefit_estimate']
            agent_bias = agent_attributes.get_psychological_attributes().get('optimism', 0.5)
            bias_strength = 0.2

            distorted_content['benefit_estimate'] = (
                    original * (1 - bias_strength) + agent_bias * bias_strength
            )

        return type(knowledge)(
            topic=knowledge.topic,
            source=knowledge.source,
            reliability=knowledge.get_reliability() * 0.95,
            content=distorted_content
        )


class SpatialNetwork(Network):
    """Simple spatial network based on geographic proximity"""

    def __init__(self, grid_size: int = 50):
        self.grid_size = grid_size
        self.agent_positions: Dict[str, tuple] = {}
        self.neighborhood_radius = 5

    def get_neighbors(self, agent_id: str) -> List[str]:
        if agent_id not in self.agent_positions:
            return []

        agent_pos = self.agent_positions[agent_id]
        neighbors = []

        for other_id, other_pos in self.agent_positions.items():
            if other_id != agent_id:
                distance = math.sqrt((agent_pos[0] - other_pos[0]) ** 2 +
                                     (agent_pos[1] - other_pos[1]) ** 2)
                if distance <= self.neighborhood_radius:
                    neighbors.append(other_id)

        return neighbors

    def add_agent(self, agent_id: str, properties: Dict[str, Any]) -> None:
        # Random placement if position not specified
        x = properties.get('x', random.randint(0, self.grid_size - 1))
        y = properties.get('y', random.randint(0, self.grid_size - 1))
        self.agent_positions[agent_id] = (x, y)

    def remove_agent(self, agent_id: str) -> None:
        if agent_id in self.agent_positions:
            del self.agent_positions[agent_id]

    def get_network_properties(self, agent_id: str) -> Dict[str, Any]:
        neighbors = self.get_neighbors(agent_id)
        return {
            'neighbor_count': len(neighbors),
            'neighbors': neighbors,
            'network_density': len(neighbors) / max(1, len(self.agent_positions) - 1)
        }


class TechnologyDiffusionRules(RulesOfInteraction):
    """Rules governing how agents influence each other's technology adoption"""

    def can_interact(self, agent_a: str, agent_b: str, context: Dict[str, Any]) -> bool:
        # Agents can interact if they're in the same network
        return agent_b in context.get('neighbors', [])

    def get_interaction_effects(self, agent_a: str, agent_b: str,
                                interaction_type: str) -> Dict[str, Any]:
        # Simple word-of-mouth effects
        if interaction_type == "technology_discussion":
            return {
                'influence_strength': random.uniform(0.05, 0.15),
                'information_transfer': True,
                'social_pressure': random.uniform(0.0, 0.1)
            }
        return {}


class TechnologyLearningCurve(ExogenousProcess):
    """Technology costs decrease over time due to learning effects"""

    def __init__(self, learning_rate: float = 0.05):
        self.learning_rate = learning_rate
        self.affected_assets = ['solar_panel', 'heat_pump', 'electric_vehicle']

    def update(self, current_time: int, environment_state: Dict[str, Any]) -> Dict[str, Any]:
        changes = {}

        # Decrease costs based on learning curve
        for asset_name in self.affected_assets:
            cost_reduction = 1 - (self.learning_rate * (current_time / 12))  # Monthly reduction
            changes[f"{asset_name}_cost_multiplier"] = max(0.5, cost_reduction)

        return changes

    def affects_assets(self) -> List[str]:
        return self.affected_assets


class PolicyIncentives(ExogenousProcess):
    """Government policies affecting technology adoption"""

    def __init__(self):
        self.incentive_programs = {
            'solar_subsidy': {'amount': 0.3, 'duration': 24},  # 30% subsidy for 2 years
            'ev_rebate': {'amount': 5000, 'duration': 12}
        }
        self.program_start_times = {}

    def update(self, current_time: int, environment_state: Dict[str, Any]) -> Dict[str, Any]:
        changes = {}

        # Introduce incentive programs at specific times
        if current_time == 12 and 'solar_subsidy' not in self.program_start_times:
            self.program_start_times['solar_subsidy'] = current_time
            changes['solar_subsidy_active'] = True
            changes['solar_subsidy_rate'] = self.incentive_programs['solar_subsidy']['amount']

        if current_time == 18 and 'ev_rebate' not in self.program_start_times:
            self.program_start_times['ev_rebate'] = current_time
            changes['ev_rebate_active'] = True
            changes['ev_rebate_amount'] = self.incentive_programs['ev_rebate']['amount']

        # End programs after duration
        for program, start_time in self.program_start_times.items():
            duration = self.incentive_programs[program]['duration']
            if current_time >= start_time + duration:
                changes[f"{program}_active"] = False

        return changes

    def affects_assets(self) -> List[str]:
        return ['solar_panel', 'electric_vehicle']


class SocialMediaFilter(InformationFilter):
    """Filter based on social media exposure and education level"""

    def filter(self, agent_attributes: AgentAttributes,
               knowledge_assets: List[KnowledgeAsset]) -> List[KnowledgeAsset]:

        household_attrs = agent_attributes
        filtered = []

        for asset in knowledge_assets:
            # Higher education agents have access to more diverse sources
            if asset.source == 'scientific_study':
                access_prob = 0.3 + household_attrs.education_level * 0.15
            elif asset.source == 'social_media':
                access_prob = 0.8 - household_attrs.age * 0.01  # Younger agents more likely
            elif asset.source == 'government':
                access_prob = 0.6
            else:
                access_prob = 0.5

            if random.random() < access_prob:
                filtered.append(asset)

        return filtered


class ConfirmationBiasDistorter(InformationDistorter):
    """Apply confirmation bias based on environmental concern"""

    def distort(self, agent_attributes: AgentAttributes,
                knowledge: KnowledgeAsset) -> KnowledgeAsset:

        household_attrs = agent_attributes
        distorted_content = knowledge.get_content().copy()

        # Bias environmental benefits toward agent's concern level
        if 'environmental_benefit' in distorted_content:
            original = distorted_content['environmental_benefit']
            bias_strength = 0.3
            env_concern = household_attrs.environmental_concern

            distorted_content['environmental_benefit'] = (
                    original * (1 - bias_strength) + env_concern * bias_strength
            )

        # Financial information distortion based on financial motivation
        if 'cost_savings' in distorted_content:
            original = distorted_content['cost_savings']
            financial_motivation = household_attrs.financial_motivation
            multiplier = 0.8 + financial_motivation * 0.4
            distorted_content['cost_savings'] = original * multiplier

        return EnergyInformation(
            topic=knowledge.topic,
            source=knowledge.source,
            reliability=knowledge.get_reliability() * 0.9,  # Slight reliability reduction
            content=distorted_content
        )


class EnergyChoiceModule(ChoiceModule):
    """Concrete choice module for energy technology decisions"""

    def __init__(self, agent_id: str, agent_attributes: HouseholdAttributes):
        super().__init__(agent_id, agent_attributes)
        self.evaluation_weights = self._calculate_weights()

    def _calculate_weights(self) -> Dict[EvaluationDimension, float]:
        """Calculate evaluation weights based on agent attributes"""
        attrs = self.agent_attributes

        weights = {
            EvaluationDimension.FINANCIAL: attrs.financial_motivation * 0.4,
            EvaluationDimension.ENVIRONMENTAL: attrs.environmental_concern * 0.3,
            EvaluationDimension.SOCIAL: attrs.social_norm_sensitivity * 0.2,
            EvaluationDimension.CONVENIENCE: 0.3 - attrs.technology_enthusiasm * 0.1,
            EvaluationDimension.PERFORMANCE: attrs.technology_enthusiasm * 0.3
        }

        # Normalize weights
        total = sum(weights.values())
        return {dim: weight / total for dim, weight in weights.items()}

    def add_trigger(self, trigger_type: TriggerType, context: Dict[str, Any]) -> None:
        """Add trigger event for decision making"""
        trigger_event = {
            'type': trigger_type,
            'context': context,
            'timestamp': context.get('current_time', 0)
        }
        self.pending_triggers.append(trigger_event)

    def evaluate_option(self, option: PhysicalAsset,
                        evaluation_context: Dict[str, Any]) -> Dict[EvaluationDimension, float]:
        """Evaluate technology option across all dimensions"""
        attrs = self.agent_attributes
        props = option.get_properties()
        scores = {}

        # Financial evaluation
        cost = option.get_cost()
        subsidies = evaluation_context.get('subsidies', {})
        effective_cost = cost * (1 - subsidies.get(option.name, 0))
        affordability = min(1.0, attrs.income / effective_cost) if effective_cost > 0 else 0

        # Consider payback period
        payback = props.get('payback_period', 20)
        payback_score = max(0, 1 - payback / 20)  # 20 years = 0 score
        scores[EvaluationDimension.FINANCIAL] = (affordability + payback_score) / 2

        # Environmental evaluation
        env_benefit = props.get('environmental_benefit', 0.5)
        scores[EvaluationDimension.ENVIRONMENTAL] = env_benefit

        # Social evaluation
        adoption_rate = evaluation_context.get('neighbor_adoption_rate', 0.1)
        social_pressure = adoption_rate * 1  # Scale up to 0-2 range
        scores[EvaluationDimension.SOCIAL] = min(1.0, social_pressure)

        # Convenience evaluation (inverse of installation complexity)
        complexity = props.get('installation_complexity', 3)
        convenience = 1 - (complexity - 1) / 4  # Scale 1-5 to 1-0
        scores[EvaluationDimension.CONVENIENCE] = max(0, convenience)

        # Performance evaluation
        efficiency = props.get('efficiency', 0.7)
        reliability = option.get_availability()
        scores[EvaluationDimension.PERFORMANCE] = (efficiency + reliability) / 2

        return scores

    def aggregate_evaluations(self, evaluations: Dict[EvaluationDimension, float]) -> float:
        """Aggregate dimension evaluations using weighted sum"""
        total_utility = 0
        for dimension, score in evaluations.items():
            weight = self.evaluation_weights.get(dimension, 0)
            total_utility += score * weight

        return total_utility

    def make_choice(self, options: List[PhysicalAsset],
                    choice_context: Dict[str, Any]) -> Optional[PhysicalAsset]:
        """Make final choice using utility maximization with threshold"""
        if not options or not self.has_pending_triggers():
            return None

        # Evaluate all options
        option_utilities = []
        for option in options:
            evaluations = self.evaluate_option(option, choice_context)
            utility = self.aggregate_evaluations(evaluations)
            option_utilities.append((option, utility, evaluations))

        # Add bounded rationality noise
        attrs = self.agent_attributes
        noise_factor = 1 - (attrs.education_level / 5) * 0.3
        for i, (option, utility, evals) in enumerate(option_utilities):
            noise = random.gauss(0, noise_factor * 0.1)
            option_utilities[i] = (option, utility + noise, evals)

        # Find best option
        option_utilities.sort(key=lambda x: x[1], reverse=True)
        best_option, best_utility, best_evals = option_utilities[0]

        # Threshold decision
        base_threshold = 0.6
        threshold = base_threshold - attrs.technology_enthusiasm * 0.2 + attrs.risk_tolerance * 0.1

        if best_utility > threshold:
            self.clear_triggers()
            return best_option

        return None


class EnergyEnvironment(Environment):
    """Concrete environment for energy technology adoption"""

    def __init__(self):
        super().__init__()
        self.policy_state = {}
        self.market_state = {}
        self.adoption_stats = {}

    def add_physical_asset(self, asset: PhysicalAsset) -> None:
        """Add energy technology to environment"""
        self.physical_assets.append(asset)

    def add_knowledge_asset(self, asset: KnowledgeAsset) -> None:
        """Add information asset to environment"""
        self.knowledge_assets.append(asset)

    def get_available_options(self, agent_id: str, context: Dict[str, Any]) -> List[PhysicalAsset]:
        """Get technology options available to specific household"""
        available = []

        for asset in self.physical_assets:
            # Check availability and agent-specific constraints
            if asset.get_availability() > 0.1:
                # Add technology-specific constraints
                if asset.name == 'solar_panel':
                    agent_attrs = context.get('agent_attributes')
                    if agent_attrs and agent_attrs.roof_suitable and agent_attrs.home_ownership:
                        available.append(asset)
                elif asset.name == 'heat_pump':
                    agent_attrs = context.get('agent_attributes')
                    if agent_attrs and agent_attrs.home_ownership:
                        available.append(asset)
                else:
                    available.append(asset)

        return available

    def get_relevant_knowledge(self, agent_id: str, context: Dict[str, Any]) -> List[KnowledgeAsset]:
        """Get information relevant to agent"""
        # All knowledge assets are potentially relevant
        return self.knowledge_assets.copy()

    def update_environment(self, current_time: int) -> None:
        """Update environment through exogenous processes"""
        environment_state = {
            'policy_state': self.policy_state,
            'market_state': self.market_state,
            'current_time': current_time
        }

        # Apply all exogenous processes
        for process in self.exogenous_processes:
            changes = process.update(current_time, environment_state)

            # Apply changes to relevant assets
            for asset in self.physical_assets:
                if asset.name in process.affects_assets():
                    cost_multiplier = changes.get(f"{asset.name}_cost_multiplier", 1.0)
                    if hasattr(asset, 'base_cost'):
                        asset.base_cost *= cost_multiplier

            # Update policy and market state
            self.policy_state.update(
                {k: v for k, v in changes.items() if 'policy' in k or 'subsidy' in k or 'rebate' in k})
            self.market_state.update({k: v for k, v in changes.items() if 'market' in k or 'cost' in k})

        # Age information assets
        for asset in self.knowledge_assets:
            if hasattr(asset, 'age_information'):
                asset.age_information()

    def add_communication(self, communication: Dict[str, Any]) -> None:
        """Add agent communication to environment"""
        self.communications.append(communication)

        # Update adoption statistics
        if communication.get('action') == 'adopted':
            technology = communication.get('technology')
            if technology:
                if technology not in self.adoption_stats:
                    self.adoption_stats[technology] = 0
                self.adoption_stats[technology] += 1


class EnergyHousehold(ConsumerAgent):
    """Concrete household agent for energy decisions"""

    def __init__(self, agent_id: str):
        # Create household attributes with realistic variation
        attributes = self._generate_household_attributes(agent_id)

        # Create choice module
        choice_module = EnergyChoiceModule(agent_id, attributes)

        # Create transformer with filters and distorters
        filters = [SocialMediaFilter()]
        distorters = [ConfirmationBiasDistorter()]
        transformer = Transformer(filters, distorters)

        super().__init__(agent_id, attributes, choice_module, transformer)

    def _generate_household_attributes(self, agent_id: str) -> HouseholdAttributes:
        """Generate realistic household attributes with correlations"""
        # Basic demographics
        age = max(25, int(random.gauss(45, 12)))
        education = random.randint(1, 5)

        # Income correlated with education and age
        base_income = 30000 + education * 10000 + max(0, age - 25) * 800
        income = max(25000, random.gauss(base_income, base_income * 0.3))

        # Psychological attributes with some correlations
        env_concern = max(0, min(1, random.gauss(0.5, 0.2)))
        tech_enthusiasm = max(0, min(1, random.gauss(0.4 + education * 0.1, 0.2)))
        risk_tolerance = max(0, min(1, random.gauss(0.5, 0.15)))
        social_sensitivity = max(0, min(1, random.gauss(0.5, 0.2)))
        financial_motivation = max(0, min(1, random.gauss(0.7, 0.15)))

        # Stock variables
        home_ownership = random.random() < (0.4 + min(0.4, income / 100000))
        roof_suitable = random.random() < 0.7 if home_ownership else False

        return HouseholdAttributes(
            agent_id=agent_id,
            age=age,
            income=income,
            education_level=education,
            household_size=random.randint(1, 5),
            home_ownership=home_ownership,
            environmental_concern=env_concern,
            risk_tolerance=risk_tolerance,
            social_norm_sensitivity=social_sensitivity,
            technology_enthusiasm=tech_enthusiasm,
            financial_motivation=financial_motivation,
            roof_suitable=roof_suitable
        )

    def perceive_environment(self, environment: Environment) -> Dict[str, Any]:
        """Perceive and process environmental information"""
        # Get relevant knowledge
        raw_knowledge = environment.get_relevant_knowledge(self.agent_id, {})

        # Apply filters and distortion
        processed_knowledge = self.transformer.process_incoming(
            self.attributes, raw_knowledge
        )

        # Get network context
        network_context = environment.get_network_context(self.agent_id)

        # Calculate neighbor adoption rates
        neighbors = network_context.get('neighbors', [])
        neighbor_adoptions = {}

        if hasattr(environment, 'communications'):
            recent_adoptions = [c for c in environment.communications
                                if c.get('agent_id') in neighbors and
                                c.get('action') == 'adopted']

            for comm in recent_adoptions:
                tech = comm.get('technology')
                if tech not in neighbor_adoptions:
                    neighbor_adoptions[tech] = 0
                neighbor_adoptions[tech] += 1

        # Store in memory for decision making
        perception = {
            'processed_knowledge': processed_knowledge,
            'network_context': network_context,
            'neighbor_adoptions': neighbor_adoptions,
            'policy_incentives': getattr(environment, 'policy_state', {}),
            'current_time': getattr(environment, 'current_time', 0)
        }

        self.memory['last_perception'] = perception
        return perception

    def make_decisions(self, environment: Environment) -> List[Dict[str, Any]]:
        """Make technology adoption decisions"""
        decisions = []

        if not self.choice_module.has_pending_triggers():
            # Check for natural trigger events
            self._check_for_triggers(environment)

        if self.choice_module.has_pending_triggers():
            # Get available options
            context = {
                'agent_attributes': self.attributes,
                'current_time': getattr(environment, 'current_time', 0)
            }
            available_options = environment.get_available_options(self.agent_id, context)

            if available_options:
                # Prepare choice context
                perception = self.memory.get('last_perception', {})
                neighbor_adoptions = perception.get('neighbor_adoptions', {})

                choice_context = {
                    'subsidies': self._extract_subsidies(perception.get('policy_incentives', {})),
                    'neighbor_adoption_rate': self._calculate_neighbor_adoption_rate(neighbor_adoptions),
                    'current_time': perception.get('current_time', 0)
                }

                # Make choice
                chosen_option = self.choice_module.make_choice(available_options, choice_context)

                if chosen_option:
                    decision = {
                        'action': 'adopt',
                        'technology': chosen_option.name,
                        'agent_id': self.agent_id,
                        'cost': chosen_option.get_cost(),
                        'context': choice_context
                    }
                    decisions.append(decision)

        return decisions

    def communicate(self, environment: Environment, decisions: List[Dict[str, Any]]) -> None:
        """Communicate decisions to environment"""
        for decision in decisions:
            if decision['action'] == 'adopt':
                # Add some communication distortion
                satisfaction = random.uniform(0.6, 0.95)

                communication = {
                    'agent_id': self.agent_id,
                    'action': 'adopted',
                    'technology': decision['technology'],
                    'satisfaction': satisfaction,
                    'timestamp': decision['context'].get('current_time', 0)
                }

                # Apply outgoing transformation
                processed_comm = self.transformer.process_outgoing(self.attributes, communication)
                environment.add_communication(processed_comm)

    def update_state(self, decisions: List[Dict[str, Any]], environment_feedback: Dict[str, Any]) -> None:
        """Update agent state based on decisions"""
        for decision in decisions:
            if decision['action'] == 'adopt':
                technology = decision['technology']

                # Update attributes
                attribute_changes = {}
                if technology == 'solar_panel':
                    attribute_changes['has_solar'] = True
                elif technology == 'electric_vehicle':
                    attribute_changes['current_car'] = 'electric'
                elif technology == 'heat_pump':
                    attribute_changes['current_heating'] = 'heat_pump'

                self.attributes.update_attributes(attribute_changes)

                # Store in memory
                self.memory[f'adopted_{technology}'] = decision['context']['current_time']

    def _check_for_triggers(self, environment: Environment) -> None:
        """Check for natural trigger events"""
        current_time = getattr(environment, 'current_time', 0)

        # Technology replacement triggers (every 15-20 years)
        if current_time > 0 and current_time % 180 == 0:  # 15 years in months
            if random.random() < 0.3:  # 30% chance
                self.choice_module.add_trigger(
                    TriggerType.REPLACEMENT,
                    {'current_time': current_time, 'reason': 'equipment_replacement'}
                )

        # Policy change triggers
        policy_state = getattr(environment, 'policy_state', {})
        for policy, active in policy_state.items():
            if active and policy not in self.memory.get('seen_policies', set()):
                self.choice_module.add_trigger(
                    TriggerType.POLICY_CHANGE,
                    {'current_time': current_time, 'policy': policy}
                )
                if 'seen_policies' not in self.memory:
                    self.memory['seen_policies'] = set()
                self.memory['seen_policies'].add(policy)

        # Social influence triggers
        perception = self.memory.get('last_perception', {})
        neighbor_adoptions = perception.get('neighbor_adoptions', {})
        adoption_rate = self._calculate_neighbor_adoption_rate(neighbor_adoptions)

        if adoption_rate > 0.3 and random.random() < 0.2:  # High adoption in neighborhood
            self.choice_module.add_trigger(
                TriggerType.SOCIAL_INFLUENCE,
                {'current_time': current_time, 'adoption_rate': adoption_rate}
            )

    def _extract_subsidies(self, policy_state: Dict[str, Any]) -> Dict[str, float]:
        """Extract current subsidy rates from policy state"""
        subsidies = {}

        if policy_state.get('solar_subsidy_active', False):
            subsidies['solar_panel'] = policy_state.get('solar_subsidy_rate', 0)

        return subsidies

    def _calculate_neighbor_adoption_rate(self, neighbor_adoptions: Dict[str, int]) -> float:
        """Calculate overall neighbor adoption rate"""
        total_adoptions = sum(neighbor_adoptions.values())
        network_context = self.memory.get('last_perception', {}).get('network_context', {})
        neighbor_count = network_context.get('neighbor_count', 1)

        return min(1.0, total_adoptions / neighbor_count) if neighbor_count > 0 else 0


class EnergyAdoptionModel(ConsumerChoiceModel):
    """Complete energy technology adoption model"""

    def initialize_model(self, parameters: Dict[str, Any]) -> None:
        """Initialize the energy adoption model"""
        self.model_parameters = parameters

        # Create environment
        self.environment = self.create_environment(parameters.get('environment', {}))

        # Create agents
        agent_count = parameters.get('agent_count', 100)
        self.agents = self.create_agents(agent_count, parameters.get('agents', {}))

        # Add agents to networks
        for network in self.environment.networks:
            for agent in self.agents:
                network.add_agent(agent.agent_id, {})

    def create_agents(self, agent_count: int, agent_config: Dict[str, Any]) -> List[ConsumerAgent]:
        """Create household agents"""
        agents = []
        for i in range(agent_count):
            agent = EnergyHousehold(f"household_{i}")
            agents.append(agent)
        return agents

    def create_environment(self, environment_config: Dict[str, Any]) -> Environment:
        """Create energy environment"""
        env = EnergyEnvironment()

        # Add physical assets (technologies)
        technologies = [
            EnergyTechnology('solar_panel', 15000, 0.8, 0.9, 3),
            EnergyTechnology('heat_pump', 12000, 0.9, 0.7, 4),
            EnergyTechnology('electric_vehicle', 35000, 0.85, 0.8, 2)
        ]

        for tech in technologies:
            env.add_physical_asset(tech)

        # Add knowledge assets
        knowledge = [
            EnergyInformation(
                'technology_benefits', 'government', 0.8,
                {'cost_savings': 0.3, 'environmental_benefit': 0.8}
            ),
            EnergyInformation(
                'adoption_trends', 'social_media', 0.6,
                {'adoption_rate': 0.15, 'satisfaction_rate': 0.75}
            ),
            EnergyInformation(
                'technical_specs', 'scientific_study', 0.9,
                {'efficiency_data': 0.85, 'reliability_data': 0.8}
            )
        ]

        for info in knowledge:
            env.add_knowledge_asset(info)

        # Add networks
        spatial_network = SpatialNetwork(50)
        env.networks.append(spatial_network)

        # Add interaction rules
        diffusion_rules = TechnologyDiffusionRules()
        env.rules.append(diffusion_rules)

        # Add exogenous processes
        learning_curve = TechnologyLearningCurve(0.02)
        policy_incentives = PolicyIncentives()
        env.exogenous_processes.extend([learning_curve, policy_incentives])

        return env

    def collect_data(self) -> Dict[str, Any]:
        """Collect simulation data for analysis"""
        data = {
            'time': self.current_time,
            'adoption_counts': {},
            'agent_attributes': [],
            'environment_state': {
                'policy_state': self.environment.policy_state.copy(),
                'adoption_stats': self.environment.adoption_stats.copy()
            }
        }

        # Collect adoption data
        technologies = ['solar_panel', 'heat_pump', 'electric_vehicle']
        for tech in technologies:
            count = sum(1 for agent in self.agents
                        if hasattr(agent.attributes, 'has_solar') and
                        ((tech == 'solar_panel' and agent.attributes.has_solar) or
                         (tech == 'electric_vehicle' and agent.attributes.current_car == 'electric') or
                         (tech == 'heat_pump' and agent.attributes.current_heating == 'heat_pump')))
            data['adoption_counts'][tech] = count

        # Sample agent attributes
        for i in range(min(10, len(self.agents))):
            agent = self.agents[i]
            attrs = {
                'agent_id': agent.agent_id,
                'income': agent.attributes.income,
                'environmental_concern': agent.attributes.environmental_concern,
                'has_solar': getattr(agent.attributes, 'has_solar', False)
            }
            data['agent_attributes'].append(attrs)

        return data

    def run_simulation(self, steps: int) -> Dict[str, Any]:
        """Run complete simulation"""
        print(f"Running energy adoption simulation for {steps} steps...")

        simulation_data = []

        for step in range(steps):
            self.step()

            # Collect data every 6 months
            if step % 6 == 0:
                data = self.collect_data()
                simulation_data.append(data)

                # Progress report
                total_adoptions = sum(data['adoption_counts'].values())
                print(f"Step {step}: Total adoptions = {total_adoptions}")

        return {
            'simulation_data': simulation_data,
            'final_stats': simulation_data[-1] if simulation_data else {},
            'parameters': self.model_parameters
        }


# Example usage and testing
def run_example():
    """Run a complete example simulation"""
    print("Energy Technology Adoption Example")
    print("=" * 50)

    # Model parameters
    # parameters = {
    #     'agent_count': 200,
    #     'environment': {},
    #     'agents': {},
    #     'random_seed': 42
    # }

    parameters = {
        'agent_count': 200,  # Number of agents
        'simulation_steps': 100,  # Time steps to run
        'random_seed': 42,  # For reproducibility
        'network_size': 50,  # Spatial grid size
        'initial_adoption': 0.05,  # Initial adoption rate

        # Agent configuration
        'agent_heterogeneity': 0.3,  # Variation in attributes
        'social_influence_strength': 0.2,

        # Environment configuration
        'number_of_assets': 3,
        'information_sources': 5,
        'policy_interventions': True
    }

    # Set random seed for reproducibility
    random.seed(parameters['random_seed'])

    # Create and initialize model
    model = EnergyAdoptionModel()
    model.initialize_model(parameters)

    print(f"Created {len(model.agents)} households")
    print(f"Environment has {len(model.environment.physical_assets)} technologies")
    print(f"Environment has {len(model.environment.knowledge_assets)} information sources")

    # Run simulation
    results = model.run_simulation(48)  # 4 years (48 months)
    curves = analyze_adoption_patterns(results)
    print(curves)
    plot_results(curves)

    # Print results
    final_stats = results['final_stats']
    print("\nFinal Adoption Results:")
    for tech, count in final_stats['adoption_counts'].items():
        percentage = (count / len(model.agents)) * 100
        print(f"  {tech}: {count} households ({percentage:.1f}%)")
    return results


def analyze_adoption_patterns(results: Dict[str, Any]):
    """Analyze technology adoption patterns over time"""
    data = results['simulation_data']

    # Extract adoption curves
    technologies = list(data[0]['adoption_counts'].keys())
    adoption_curves = {tech: [] for tech in technologies}

    for timestep in data:
        for tech in technologies:
            adoption_curves[tech].append(timestep['adoption_counts'][tech])

    # Calculate adoption rates
    agent_count = len(results.get('agents', [100]))  # fallback
    for tech, curve in adoption_curves.items():
        final_rate = curve[-1] / agent_count * 100
        print(f"{tech}: {final_rate:.1f}% final adoption rate")

    return adoption_curves


def analyze_network_effects(results: Dict[str, Any]):
    """Analyze social network influence on adoption"""
    # Implementation depends on your specific network structure
    pass


def plot_results(adoption_curves: Dict[str, List[int]]):
    """Plot adoption curves (requires matplotlib)"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('TkAgg')

        plt.figure(figsize=(10, 6))
        for tech, curve in adoption_curves.items():
            plt.plot(curve, label=tech)

        plt.xlabel('Time Steps')
        plt.ylabel('Number of Adopters')
        plt.title('Technology Adoption Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()
    except ImportError:
        print("matplotlib not available for plotting")


if __name__ == "__main__":
    run_example()