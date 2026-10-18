"""
Tests for scheduling, seeding, validation, perception memory, events and defaults.
"""

import pytest

from consumer_choice_metamodel import (
    EventBus, ModelEvent, ModelValidator, Phase, PhasedScheduler,
    RandomActivationScheduler, SequentialScheduler, TriggerType,
)

from conftest import SimpleAgent, SimpleAsset, SimpleAttributes, SimpleChoice


def _ids(model, order_log):
    return [agent_id for _, agent_id in order_log]


class TestSchedulers:

    def test_default_scheduler_is_sequential(self, make_model):
        model = make_model()
        assert isinstance(model.scheduler, SequentialScheduler)

    def test_sequential_runs_agents_in_list_order(self, make_model):
        model = make_model(agent_count=4)
        log = []
        for agent in model.agents:
            agent.log = log
        model.step()
        assert _ids(model, log) == ["a0", "a1", "a2", "a3"]
        assert model.current_time == 1

    def test_random_activation_is_a_permutation_and_seeded(self, make_model):
        orders = []
        for _ in range(2):
            model = make_model(seed=7, scheduler=RandomActivationScheduler(), agent_count=10)
            log = []
            for agent in model.agents:
                agent.log = log
            model.step()
            orders.append(_ids(model, log))
        assert sorted(orders[0]) == sorted(a.agent_id for a in model.agents)
        assert orders[0] == orders[1]

    def test_phased_runs_each_phase_across_all_agents(self, make_model):
        calls = []
        scheduler = PhasedScheduler([
            Phase("first", lambda agent, model: calls.append(("first", agent.agent_id))),
            Phase("count", lambda model: calls.append(("count", model.current_time)),
                  per_agent=False),
            Phase("second", lambda agent, model: calls.append(("second", agent.agent_id))),
        ], shuffle=False)
        model = make_model(scheduler=scheduler, agent_count=3)
        model.step()
        assert calls == [
            ("first", "a0"), ("first", "a1"), ("first", "a2"),
            ("count", 0),
            ("second", "a0"), ("second", "a1"), ("second", "a2"),
        ]
        assert scheduler.current_phase is None

    def test_phased_shares_one_shuffled_order_per_step(self, make_model):
        seen = {"a": [], "b": []}
        scheduler = PhasedScheduler([
            Phase("a", lambda agent, model: seen["a"].append(agent.agent_id)),
            Phase("b", lambda agent, model: seen["b"].append(agent.agent_id)),
        ])
        model = make_model(seed=3, scheduler=scheduler, agent_count=8)
        model.step()
        assert seen["a"] == seen["b"]

    def test_phased_reshuffle_each_phase(self, make_model):
        seen = {"a": [], "b": []}
        scheduler = PhasedScheduler([
            Phase("a", lambda agent, model: seen["a"].append(agent.agent_id)),
            Phase("b", lambda agent, model: seen["b"].append(agent.agent_id)),
        ], reshuffle_each_phase=True)
        model = make_model(seed=3, scheduler=scheduler, agent_count=20)
        model.step()
        assert sorted(seen["a"]) == sorted(seen["b"])
        assert seen["a"] != seen["b"]


class TestRandomness:

    def test_add_agents_shares_model_rng(self, make_model):
        model = make_model(seed=1)
        assert all(agent.rng is model.rng for agent in model.agents)

    def test_same_seed_same_run(self, make_model):
        def draws(seed):
            model = make_model(seed=seed, scheduler=RandomActivationScheduler())
            model.run_simulation(3)
            return [agent.draws for agent in model.agents]
        assert draws(42) == draws(42)
        assert draws(42) != draws(43)

    def test_reseed_restarts_sequence(self, make_model):
        model = make_model(seed=5)
        first = model.rng.random()
        model.reseed(5)
        assert model.rng.random() == first
        assert model.seed == 5

    def test_agent_without_model_gets_own_rng(self):
        agent = SimpleAgent("solo")
        assert 0 <= agent.rng.random() < 1


class TestPerception:

    def test_step_stores_perception_before_decisions(self, make_model):
        model = make_model(agent_count=1)
        agent = model.agents[0]
        decisions = []
        original = agent.make_decisions
        agent.make_decisions = lambda env: decisions.extend(original(env)) or decisions
        model.step()
        assert agent.memory[agent.PERCEPTION_KEY] == {'time_seen': 0}
        assert decisions[0]['perception'] == {'time_seen': 0}

    def test_update_state_receives_network_context(self, make_model):
        model = make_model(agent_count=3)
        model.step()
        assert model.agents[0].memory['feedback'] == {'neighbors': ["a1", "a2"]}


class TestValidator:

    def test_default_requires_income_and_age(self):
        attrs = SimpleAttributes("x")
        assert ModelValidator.validate_agent_attributes(attrs) == []
        attrs.get_socioeconomic_attributes = lambda: {}
        errors = ModelValidator.validate_agent_attributes(attrs)
        assert len(errors) == 2

    def test_required_socioeconomic_is_configurable(self):
        attrs = SimpleAttributes("x")
        attrs.get_socioeconomic_attributes = lambda: {'owner': True}
        assert ModelValidator.validate_agent_attributes(attrs, required_socioeconomic=()) == []
        errors = ModelValidator.validate_agent_attributes(
            attrs, required_socioeconomic=['owner', 'region'])
        assert errors == ["Required socioeconomic attribute region missing"]

    def test_psychological_range_still_checked(self):
        attrs = SimpleAttributes("x", concern=1.5)
        errors = ModelValidator.validate_agent_attributes(attrs, required_socioeconomic=())
        assert errors == ["Psychological attribute concern must be between 0 and 1"]

    def test_validate_model_prefixes_agent_ids(self, make_model):
        model = make_model(agent_count=2)
        assert ModelValidator.validate_model(model) == []
        model.agents[1].attributes.concern = -1
        errors = ModelValidator.validate_model(model)
        assert errors == ["a1: Psychological attribute concern must be between 0 and 1"]

    def test_validate_model_without_environment(self, make_model):
        model = make_model(agent_count=0)
        model.environment = None
        assert ModelValidator.validate_model(model) == ["Model has no environment"]


class TestEventBus:

    def test_publish_and_unsubscribe(self):
        bus = EventBus()
        received = []
        callback = received.append
        bus.subscribe("adopted", callback)
        bus.publish(ModelEvent("adopted", "a0", {}))
        bus.unsubscribe("adopted", callback)
        bus.publish(ModelEvent("adopted", "a0", {}))
        assert len(received) == 1
        bus.unsubscribe("never-subscribed", callback)   # no error

    def test_wildcard_receives_all_after_specific(self):
        bus = EventBus()
        order = []
        bus.subscribe(EventBus.ALL_EVENTS, lambda e: order.append(("all", e.event_type)))
        bus.subscribe("x", lambda e: order.append(("x", e.event_type)))
        bus.publish(ModelEvent("x", "s", {}))
        bus.publish(ModelEvent("y", "s", {}))
        assert order == [("x", "x"), ("all", "x"), ("all", "y")]

    def test_wildcard_event_delivered_once(self):
        bus = EventBus()
        received = []
        bus.subscribe(EventBus.ALL_EVENTS, received.append)
        bus.publish(ModelEvent(EventBus.ALL_EVENTS, "s", {}))
        assert len(received) == 1

    def test_model_publish_stamps_current_time(self, make_model):
        model = make_model()
        received = []
        model.event_bus.subscribe("adopted", received.append)
        model.step()
        model.step()
        event = model.publish("adopted", "a0", {"technology": "PV"})
        assert received == [event]
        assert event.timestamp == 2

    def test_explicit_timestamp_is_kept(self):
        bus = EventBus(clock=lambda: 99)
        event = ModelEvent("x", "s", {}, timestamp=3)
        bus.publish(event)
        assert event.timestamp == 3

    def test_callback_may_unsubscribe_itself(self):
        bus = EventBus()
        received = []

        def once(event):
            received.append(event)
            bus.unsubscribe("x", once)

        bus.subscribe("x", once)
        bus.publish(ModelEvent("x", "s", {}))
        bus.publish(ModelEvent("x", "s", {}))
        assert len(received) == 1


class TestDefaults:

    def test_default_add_trigger(self):
        attrs = SimpleAttributes("x")
        choice = SimpleChoice("x", attrs)
        choice.add_trigger(TriggerType.REPLACEMENT, {'current_time': 4})
        assert choice.has_pending_triggers()
        assert choice.pending_triggers == [
            {'type': TriggerType.REPLACEMENT, 'context': {'current_time': 4}, 'timestamp': 4}
        ]

    def test_run_simulation_collects_initial_and_each_step(self, make_model):
        model = make_model(agent_count=3)
        results = model.run_simulation(4)
        assert [d['time'] for d in results['simulation_data']] == [0, 1, 2, 3, 4]
        assert results['final_stats'] == {'time': 4, 'agent_count': 3}
        assert results['parameters'] == {'agent_count': 3}


class TestDataclasses:

    def test_model_event_fields(self):
        event = ModelEvent("adopted", "a0", {"technology": "PV"})
        assert event.timestamp is None
        assert event == ModelEvent("adopted", "a0", {"technology": "PV"})
        assert "adopted" in repr(event)

    def test_asset_subclass_can_be_a_dataclass(self):
        from dataclasses import dataclass

        @dataclass(eq=False)
        class HeatPump(SimpleAsset):
            price: float

        pump = HeatPump("heat_pump", 12000.0)
        assert pump.price == 12000.0
        assert pump.asset_id

    def test_assets_compare_by_identity(self):
        a, b = SimpleAsset("thing"), SimpleAsset("thing")
        assert a != b
        assert a.asset_id != b.asset_id
        assert len({a, b}) == 2
        assert {a: 1}[a] == 1

    def test_knowledge_asset_hashable(self):
        from consumer_choice_metamodel import KnowledgeAsset

        class Note(KnowledgeAsset):
            def get_reliability(self):
                return 1.0

            def get_content(self):
                return {}

        n = Note("topic", "source")
        assert n in {n}
