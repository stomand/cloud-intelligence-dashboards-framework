"""Mock-based unit tests for the ``list_agents`` handler.

Feature: cid-cmd-agent-flow-platform (task 11.3)

The handler under test is ``Cid.list_agents`` in ``cid/common.py``. Tests call the
UNDECORATED handler via ``Cid.list_agents.__wrapped__`` (the ``@command`` decorator
uses ``functools.wraps``), so no AWS login or catalog loading is performed.

Harness: ``Cid.__new__(Cid)`` with ``resources`` injected and a MagicMock ``agent``
helper written into ``__dict__`` to preempt the ``cached_property`` descriptor.
No real AWS call is ever made.

_Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_
"""
import io
import contextlib
from unittest.mock import MagicMock

from cid.common import Cid
from cid.helpers.quicksight.agent_logic import DEPLOYED_MARK

# the undecorated handler (the @command decorator uses functools.wraps)
list_agents = Cid.list_agents.__wrapped__


def make_catalog():
    """A catalog spanning multiple categories, including a Deprecated entry."""
    return {
        'finops': {'name': 'FinOps Advisor', 'agentId': 'finops', 'category': 'FinOps'},
        'kpi': {'name': 'KPI Advisor', 'agentId': 'kpi', 'category': 'FinOps'},
        'operations': {'name': 'Ops Advisor', 'agentId': 'operations', 'category': 'Operations'},
        'legacy': {'name': 'Legacy Agent', 'agentId': 'legacy', 'category': 'Deprecated'},
    }


def make_cid(catalog, deployed=(), failing=()):
    """Build a Cid harness with the given catalog and a mocked agent helper.

    :param catalog: the ``resources['agents']`` mapping
    :param deployed: agent ids for which agent.get returns a non-None describe
    :param failing: agent ids for which agent.get raises (live-query failure)
    """
    cid_obj = Cid.__new__(Cid)
    cid_obj.__dict__.clear()
    cid_obj.resources = {'agents': catalog}

    agent = MagicMock(name='agent')

    def fake_get(agent_id):
        if agent_id in failing:
            raise RuntimeError(f'DescribeAgent failed for {agent_id}')
        if agent_id in deployed:
            return {'AgentId': agent_id, 'Arn': f'arn:aws:quicksight:us-east-1:123456789012:agent/{agent_id}'}
        return None

    agent.get.side_effect = fake_get
    cid_obj.__dict__['agent'] = agent
    return cid_obj


def run(cid_obj, **kwargs):
    """Run the undecorated handler capturing stdout; returns (listing, output)."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = list_agents(cid_obj, **kwargs)
    return result, buffer.getvalue()


class TestCategoryGroupingAndDeployedMarking:
    """: category-grouped listing with ✓ on deployed entries."""

    def test_entries_grouped_by_category(self):
        cid_obj = make_cid(make_catalog())
        listing, output = run(cid_obj)
        assert set(listing) == {'FinOps', 'Operations'}
        assert [entry['key'] for entry in listing['FinOps']] == ['finops', 'kpi']
        assert [entry['key'] for entry in listing['Operations']] == ['operations']
        # category headers appear in the printed output
        assert 'FinOps' in output and 'Operations' in output

    def test_exactly_the_deployed_entries_are_check_marked(self):
        cid_obj = make_cid(make_catalog(), deployed=('finops', 'operations'))
        listing, output = run(cid_obj)
        deployed = {entry['agentId'] for entries in listing.values()
                    for entry in entries if entry['deployed']}
        assert deployed == {'finops', 'operations'}
        assert f'{DEPLOYED_MARK}[finops]' in output
        assert f'{DEPLOYED_MARK}[operations]' in output
        # the non-deployed entry is listed without the check indicator
        assert '[kpi] KPI Advisor' in output
        assert f'{DEPLOYED_MARK}[kpi]' not in output

    def test_no_deployed_entries_no_check_marks(self):
        cid_obj = make_cid(make_catalog())
        listing, output = run(cid_obj)
        assert all(not entry['deployed'] for entries in listing.values() for entry in entries)
        assert DEPLOYED_MARK not in output

    def test_empty_catalog_prints_message_and_returns_empty(self):
        cid_obj = make_cid({})
        listing, output = run(cid_obj)
        assert listing == {}
        assert 'No agents found' in output
        assert not cid_obj.agent.get.called


class TestLiveStateSource:
    """: deployment status comes from live DescribeAgent per entry,
    never from ListAgents (which omits PREVIEW/FAILED agents)."""

    def test_agent_get_called_once_per_non_deprecated_entry(self):
        cid_obj = make_cid(make_catalog(), deployed=('finops',))
        run(cid_obj)
        queried = sorted(call.args[0] for call in cid_obj.agent.get.call_args_list)
        assert queried == ['finops', 'kpi', 'operations']

    def test_status_never_relies_on_list_agents(self):
        cid_obj = make_cid(make_catalog(), deployed=('finops',))
        run(cid_obj)
        assert not cid_obj.agent.client.list_agents.called
        # no ListAgents-shaped call anywhere on the helper either
        assert not any('list_agents' in name for name, _, _ in cid_obj.agent.mock_calls
                       if name != 'get')


class TestDeprecatedHidden:
    """: Deprecated entries are absent from output AND never described."""

    def test_deprecated_entry_absent_from_listing_and_output(self):
        cid_obj = make_cid(make_catalog(), deployed=('legacy',))
        listing, output = run(cid_obj)
        assert 'Deprecated' not in listing
        keys = {entry['key'] for entries in listing.values() for entry in entries}
        assert 'legacy' not in keys
        assert 'legacy' not in output
        assert 'Legacy Agent' not in output

    def test_deprecated_entry_never_queried(self):
        cid_obj = make_cid(make_catalog())
        run(cid_obj)
        queried = {call.args[0] for call in cid_obj.agent.get.call_args_list}
        assert 'legacy' not in queried


class TestUnknownStatusFallback:
    """: a live-query failure marks that entry '?' and listing continues."""

    def test_failing_entry_listed_with_unknown_indicator(self):
        cid_obj = make_cid(make_catalog(), deployed=('finops',), failing=('kpi',))
        listing, output = run(cid_obj)
        assert '?[kpi] KPI Advisor' in output
        # the failing entry is not marked deployed
        kpi = next(entry for entry in listing['FinOps'] if entry['key'] == 'kpi')
        assert kpi['deployed'] is False

    def test_remaining_entries_still_listed_after_failure(self):
        cid_obj = make_cid(make_catalog(), deployed=('finops',), failing=('kpi',))
        listing, output = run(cid_obj)
        keys = {entry['key'] for entries in listing.values() for entry in entries}
        assert keys == {'finops', 'kpi', 'operations'}
        # deployed marking on the other entries is unaffected
        assert f'{DEPLOYED_MARK}[finops]' in output
        assert '[operations] Ops Advisor' in output

    def test_all_entries_failing_all_marked_unknown_none_deployed(self):
        cid_obj = make_cid(make_catalog(), failing=('finops', 'kpi', 'operations'))
        listing, output = run(cid_obj)
        assert output.count('?[') == 3
        assert DEPLOYED_MARK not in output
        assert all(not entry['deployed'] for entries in listing.values() for entry in entries)
