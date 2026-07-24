"""Mock-based unit tests and property tests for the ``delete_agent`` handler.

Feature: cid-cmd-agent-flow-platform (tasks 12.3 - 12.5)

The handler under test is ``Cid.delete_agent`` in ``cid/common.py``. Tests call the
UNDECORATED handler via ``Cid.delete_agent.__wrapped__`` (the ``@command`` decorator
uses ``functools.wraps``), so no AWS login or catalog loading is performed.

Harness: mirrors ``test_create_agent_flow.py`` — ``Cid.__new__(Cid)`` with injected
``resources``, a fake ``base`` (SimpleNamespace), and MagicMock ``qs`` / ``space`` /
``agent`` helpers written into ``__dict__`` to preempt the ``cached_property``
descriptors. Confirmation is controlled through the real ``cid.utils`` parameter
state (``set_parameters({'confirm-delete': ...})``). No real AWS call is ever made.
"""
import io
import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

import cid.utils as cid_utils
from cid.common import Cid
from cid.exceptions import CidError
from cid.helpers.quicksight.agent_logic import (
    CID_MANAGED_MARKER,
    CID_PROVENANCE_TAG_KEY,
    CID_PROVENANCE_TAG_VALUE,
)

ACCOUNT_ID = '123456789012'
REGION = 'us-east-1'
PARTITION = 'aws'
DOMAIN = 'aws.amazon.com'
AGENT_ID = 'test-agent'
SPACE_KEY = 'test-space'
SPACE_ARN = f'arn:{PARTITION}:quicksight:{REGION}:{ACCOUNT_ID}:space/{SPACE_KEY}'
AGENT_ARN = f'arn:{PARTITION}:quicksight:{REGION}:{ACCOUNT_ID}:agent/{AGENT_ID}'

# the undecorated handler (the @command decorator uses functools.wraps)
delete_agent = Cid.delete_agent.__wrapped__

# the ONLY delete calls the delete-agent flow may ever issue (Property 23)
ALLOWED_DELETE_CALLS = {'agent.delete', 'space.client.delete_space'}
# any create/delete call whose name contains one of these is out of scope for the
# whole platform: dashboards, datasets, knowledge bases, Research (Req 12.3, 16.1, 16.4)
FORBIDDEN_NAME_FRAGMENTS = (
    'create_dashboard', 'delete_dashboard',
    'create_data_set', 'delete_data_set', 'create_dataset', 'delete_dataset',
    'create_knowledge_base', 'delete_knowledge_base',
    'create_research', 'delete_research',
)


def agent_arn(agent_id):
    return f'arn:{PARTITION}:quicksight:{REGION}:{ACCOUNT_ID}:agent/{agent_id}'


def reset_parameters(initial=None):
    """Reset the cid.utils global parameter state (params/defaults/all_yes)."""
    cid_utils.params.clear()
    cid_utils.defaults.clear()
    cid_utils._all_yes = False
    if initial:
        cid_utils.set_parameters(initial)


@pytest.fixture(autouse=True)
def _isolated_parameter_state():
    """Save/restore cid.utils global state around every test."""
    params_backup = dict(cid_utils.params)
    defaults_backup = dict(cid_utils.defaults)
    yes_backup = cid_utils._all_yes
    reset_parameters()
    yield
    cid_utils.params.clear()
    cid_utils.params.update(params_backup)
    cid_utils.defaults.clear()
    cid_utils.defaults.update(defaults_backup)
    cid_utils._all_yes = yes_backup


def make_exceptions():
    """Real exception subclasses for the mocked quicksight client."""
    ns = SimpleNamespace()
    for name in ('ResourceNotFoundException', 'ResourceExistsException',
                 'AccessDeniedException', 'ConflictException', 'ClientError'):
        setattr(ns, name, type(name, (Exception,), {}))
    return ns


def make_target_definition(space_keys=(SPACE_KEY,)):
    """Catalog entry for the delete target."""
    return {
        'name': 'Test Agent',
        'agentId': AGENT_ID,
        'category': 'FinOps',
        'dependsOn': {'spaces': list(space_keys)},
    }


def make_deployed_agent(agent_id, cid_managed=True, space_arns=(SPACE_ARN,)):
    """DescribeAgent 'Agent' dict as returned by Agent.get."""
    description = f'agent {CID_MANAGED_MARKER}' if cid_managed else 'hand-made agent'
    return {
        'AgentId': agent_id,
        'Arn': agent_arn(agent_id),
        'Description': description,
        'Spaces': list(space_arns),
    }


def make_cid(agents_catalog, deployed_agents, spaces=None):
    """Build a Cid harness with mocked qs/space/agent helpers.

    :param agents_catalog: the catalog 'agents' map (key -> entry)
    :param deployed_agents: agent_id -> DescribeAgent dict (agent.get lookup table)
    :param spaces: space_id -> space dict (space.get lookup table); default: the
                   target Space existing with CID-managed provenance
    """
    cid_obj = Cid.__new__(Cid)
    cid_obj.__dict__.clear()
    excs = make_exceptions()

    cid_obj.resources = {'agents': dict(agents_catalog), 'dashboards': {}, 'datasets': {}}
    cid_obj.base = SimpleNamespace(
        account_id=ACCOUNT_ID, region=REGION, partition=PARTITION, domain=DOMAIN,
        username='test-user', session=MagicMock(name='session'),
    )

    qs = MagicMock(name='qs')
    qs.list_dashboards.return_value = []
    cid_obj.__dict__['qs'] = qs

    if spaces is None:
        spaces = {SPACE_KEY: {
            'spaceArn': SPACE_ARN, 'name': 'Test Space',
            'description': f'space {CID_MANAGED_MARKER}',
        }}
    space = MagicMock(name='space')
    space.client.exceptions = excs
    space.client.list_tags_for_resource.return_value = {'Tags': []}
    space.get.side_effect = lambda space_id: spaces.get(space_id)
    cid_obj.__dict__['space'] = space

    agent = MagicMock(name='agent')
    agent.client.exceptions = excs
    agent.get.side_effect = lambda an_id: deployed_agents.get(an_id)
    cid_obj.__dict__['agent'] = agent
    return cid_obj


def run(cid_obj, **kwargs):
    """Run the undecorated handler capturing stdout; returns (result, output)."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = delete_agent(cid_obj, **kwargs)
    return result, buffer.getvalue()


def run_raising(cid_obj, exception_type, **kwargs):
    """Run the handler expecting an exception; returns (excinfo, output)."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        with pytest.raises(exception_type) as excinfo:
            delete_agent(cid_obj, **kwargs)
    return excinfo, buffer.getvalue()


def all_call_names(cid_obj):
    """Fully qualified call names recorded on the qs/space/agent helper mocks."""
    names = []
    for helper_name in ('qs', 'space', 'agent'):
        helper = cid_obj.__dict__[helper_name]
        for call in helper.mock_calls:
            if call[0]:  # skip bare __call__ entries
                names.append(f'{helper_name}.{call[0]}')
    return names


def assert_only_in_scope_calls(cid_obj):
    """Property 23 core assertion: zero out-of-scope create or delete calls.

    The delete flow may only ever delete the target Agent (``agent.delete``) and,
    when allowed, the agent's Space (``space.client.delete_space``). It must never
    issue ANY create call, and never any dashboard/dataset/knowledge-base/Research
    create or delete call (Req 12.3, 16.1, 16.4).
    """
    for name in all_call_names(cid_obj):
        method = name.split('.')[-1]
        assert not any(fragment in method for fragment in FORBIDDEN_NAME_FRAGMENTS), (
            f'out-of-scope call issued by delete-agent: {name}')
        assert 'create' not in method, f'delete-agent must never create anything: {name}'
        if 'delete' in method:
            assert name in ALLOWED_DELETE_CALLS, (
                f'delete-agent issued a delete call outside its scope: {name}')


# ===========================================================================
# Task 12.3: mock-based unit tests for delete scoping
# ===========================================================================
class TestConfirmationGate:
    """Req 12.1: deletion is gated by a yes/no parameter defaulting to 'no'."""

    def test_confirmation_prompt_defaults_to_no(self):
        """The yes/no confirmation is requested with default='no', and a
        non-confirmed answer makes no delete call."""
        cid_obj = make_cid({AGENT_ID: make_target_definition()},
                           {AGENT_ID: make_deployed_agent(AGENT_ID)})
        with patch('cid.common.get_yesno_parameter', return_value=False) as yesno:
            result, output = run(cid_obj, agent_id=AGENT_ID)
        assert result == AGENT_ID
        yesno.assert_called_once()
        assert yesno.call_args.kwargs.get('default') == 'no'
        assert yesno.call_args.kwargs.get('param_name') == 'confirm-delete'
        assert not cid_obj.agent.delete.called
        assert 'not confirmed' in output

    def test_explicit_no_parameter_makes_no_delete_call(self):
        """With the real parameter engine and confirm-delete='no', the handler
        returns without deleting anything."""
        reset_parameters({'confirm-delete': 'no'})
        cid_obj = make_cid({AGENT_ID: make_target_definition()},
                           {AGENT_ID: make_deployed_agent(AGENT_ID)})
        result, output = run(cid_obj, agent_id=AGENT_ID)
        assert result == AGENT_ID
        assert 'not confirmed' in output
        assert not cid_obj.agent.delete.called
        assert not cid_obj.space.client.delete_space.called


class TestMissingTargetSuccess:
    """Req 12.2: a missing delete target is treated as success."""

    def test_absent_agent_returns_cleanly_without_delete_call(self):
        cid_obj = make_cid({AGENT_ID: make_target_definition()}, deployed_agents={})
        result, output = run(cid_obj, agent_id=AGENT_ID)
        assert result == AGENT_ID
        assert 'does not exist' in output
        assert not cid_obj.agent.delete.called
        assert not cid_obj.space.client.delete_space.called


class TestNoDashboardDelete:
    """Req 12.3: deleting an Agent never deletes any dashboard."""

    def test_successful_delete_issues_no_dashboard_delete_anywhere(self):
        reset_parameters({'confirm-delete': 'yes'})
        cid_obj = make_cid({AGENT_ID: make_target_definition()},
                           {AGENT_ID: make_deployed_agent(AGENT_ID)})
        result, output = run(cid_obj, agent_id=AGENT_ID)
        assert result == AGENT_ID
        assert 'No dashboard was deleted' in output
        cid_obj.agent.delete.assert_called_once_with(AGENT_ID)
        # no dashboard-delete method invoked anywhere: the qs helper (which owns
        # delete_dashboard) is never touched, and the only delete call is agent.delete
        assert not cid_obj.qs.mock_calls
        names = all_call_names(cid_obj)
        assert not any('delete_dashboard' in name for name in names)
        assert [name for name in names if 'delete' in name.split('.')[-1]] == ['agent.delete']


class TestNonCidTargetRefused:
    """Req 12.7: a target lacking CID_Managed provenance is refused and reported."""

    def test_non_cid_agent_refused_with_report_and_no_delete_call(self):
        reset_parameters({'confirm-delete': 'yes'})
        cid_obj = make_cid({AGENT_ID: make_target_definition()},
                           {AGENT_ID: make_deployed_agent(AGENT_ID, cid_managed=False)})
        excinfo, _ = run_raising(cid_obj, CidError, agent_id=AGENT_ID)
        message = str(excinfo.value)
        assert 'not managed' in message
        assert AGENT_ID in message
        assert not cid_obj.agent.delete.called
        assert not cid_obj.space.client.delete_space.called

    def test_provenance_tag_alone_is_sufficient_to_proceed(self):
        """Dual-mechanism detection: the provenance tag counts even without the
        Description marker (Req 13.4)."""
        reset_parameters({'confirm-delete': 'yes'})
        target = make_deployed_agent(AGENT_ID, cid_managed=False)
        cid_obj = make_cid({AGENT_ID: make_target_definition()}, {AGENT_ID: target})
        cid_obj.space.client.list_tags_for_resource.return_value = {
            'Tags': [{'Key': CID_PROVENANCE_TAG_KEY, 'Value': CID_PROVENANCE_TAG_VALUE}]}
        result, _ = run(cid_obj, agent_id=AGENT_ID)
        assert result == AGENT_ID
        cid_obj.agent.delete.assert_called_once_with(AGENT_ID)


class TestConflictRetryDelegation:
    """Req 12.8: ConflictException wait-and-retry during delete.

    The retry loop itself lives in ``Agent.delete`` (``helpers/quicksight/agent.py``)
    and is covered by ``test_agent_helper.py`` (ConflictException while the agent is
    UPDATING/CREATING is retried until the status settles, up to the 300-second
    limit). The handler's responsibility is to delegate the deletion to
    ``self.agent.delete``, which owns that retry behavior — asserted here.
    """

    def test_handler_delegates_deletion_to_the_retrying_agent_helper(self):
        reset_parameters({'confirm-delete': 'yes'})
        cid_obj = make_cid({AGENT_ID: make_target_definition()},
                           {AGENT_ID: make_deployed_agent(AGENT_ID)})
        result, _ = run(cid_obj, agent_id=AGENT_ID)
        assert result == AGENT_ID
        # exactly one delegation to the helper that owns the ConflictException retry
        cid_obj.agent.delete.assert_called_once_with(AGENT_ID)


# ===========================================================================
# Property tests (tasks 12.4 - 12.5)
# ===========================================================================
PROPERTY_SETTINGS = settings(
    max_examples=100, deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

# per other-agent configuration: (deployed?, CID-managed?, reference mechanism)
other_agent_configs = st.lists(
    st.tuples(
        st.booleans(),                                              # deployed?
        st.booleans(),                                              # CID-managed?
        st.sampled_from(['none', 'catalog', 'deployed', 'both']),   # references the space?
    ),
    min_size=0, max_size=4,
)


def build_catalog_and_deployed(configs):
    """Materialize other-agent configs into catalog entries + deployed agents.

    Returns (agents_catalog, deployed_agents, blocking_ids) where blocking_ids are
    the other agents that must retain the Space: deployed AND CID-managed AND
    referencing the space (via the catalog dependsOn.spaces or the deployed Spaces
    list).
    """
    agents_catalog = {AGENT_ID: make_target_definition()}
    deployed_agents = {AGENT_ID: make_deployed_agent(AGENT_ID)}
    blocking_ids = set()
    for index, (deployed, managed, reference) in enumerate(configs):
        other_id = f'other-{index}'
        entry = {'name': f'Other {index}', 'agentId': other_id, 'category': 'FinOps',
                 'dependsOn': {'spaces': [SPACE_KEY] if reference in ('catalog', 'both') else []}}
        agents_catalog[other_id] = entry
        if deployed:
            space_arns = [SPACE_ARN] if reference in ('deployed', 'both') else []
            deployed_agents[other_id] = make_deployed_agent(
                other_id, cid_managed=managed, space_arns=space_arns)
            if managed and reference != 'none':
                blocking_ids.add(other_id)
    return agents_catalog, deployed_agents, blocking_ids


# Feature: cid-cmd-agent-flow-platform, Property 22: Space deletion respects
# cross-agent dependencies
# **Validates: Requirements 12.4, 12.5**
@PROPERTY_SETTINGS
@given(configs=other_agent_configs)
def test_property_22_space_deleted_iff_no_other_cid_managed_agent_references_it(configs):
    """For any set of other catalog agents — each randomly deployed or not,
    CID-managed or not, referencing the Space or not — ``--delete-space`` deletes
    the Space if and only if no OTHER deployed CID-managed agent references it;
    otherwise the Space is retained and the dependency is reported."""
    reset_parameters({'confirm-delete': 'yes', 'delete-space': True})
    agents_catalog, deployed_agents, blocking_ids = build_catalog_and_deployed(configs)
    cid_obj = make_cid(agents_catalog, deployed_agents)
    result, output = run(cid_obj, agent_id=AGENT_ID)
    assert result == AGENT_ID
    cid_obj.agent.delete.assert_called_once_with(AGENT_ID)
    if blocking_ids:
        # retained + every blocking dependency reported (Req 12.5)
        assert not cid_obj.space.client.delete_space.called
        assert 'retained' in output
        for other_id in blocking_ids:
            assert other_id in output
    else:
        # no other deployed CID-managed agent references the space: deleted (Req 12.4)
        cid_obj.space.client.delete_space.assert_called_once_with(
            AwsAccountId=ACCOUNT_ID, SpaceId=SPACE_KEY)


# Feature: cid-cmd-agent-flow-platform, Property 23: The platform never issues
# out-of-scope create or delete calls
# **Validates: Requirements 12.3, 12.7, 16.1, 16.4**
@PROPERTY_SETTINGS
@given(
    target_exists=st.booleans(),
    target_managed=st.booleans(),
    confirmed=st.booleans(),
    delete_space_flag=st.booleans(),
)
def test_property_23_delete_never_issues_out_of_scope_create_or_delete_calls(
        target_exists, target_managed, confirmed, delete_space_flag):
    """Across generated delete scenarios (target exists/absent, CID/non-CID,
    confirmed/unconfirmed, delete-space on/off), the only delete calls ever made
    are ``agent.delete`` and (when allowed) ``space.client.delete_space`` — never
    any dashboard, dataset, knowledge-base, or Research create or delete call.
    A non-CID target additionally receives zero delete calls and is reported."""
    initial = {'confirm-delete': 'yes' if confirmed else 'no'}
    if delete_space_flag:
        initial['delete-space'] = True
    reset_parameters(initial)
    deployed_agents = {}
    if target_exists:
        deployed_agents[AGENT_ID] = make_deployed_agent(AGENT_ID, cid_managed=target_managed)
    cid_obj = make_cid({AGENT_ID: make_target_definition()}, deployed_agents)

    if target_exists and not target_managed:
        excinfo, _ = run_raising(cid_obj, CidError, agent_id=AGENT_ID)
        assert 'not managed' in str(excinfo.value)          # reported (Req 12.7)
        assert not cid_obj.agent.delete.called              # zero delete calls
        assert not cid_obj.space.client.delete_space.called
    else:
        result, _ = run(cid_obj, agent_id=AGENT_ID)
        assert result == AGENT_ID
        deletion_allowed = target_exists and target_managed and confirmed
        assert cid_obj.agent.delete.called == deletion_allowed
        if not (deletion_allowed and delete_space_flag):
            assert not cid_obj.space.client.delete_space.called

    assert_only_in_scope_calls(cid_obj)
