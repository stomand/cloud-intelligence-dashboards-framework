"""Mock-based unit tests and property tests for the ``create_agent`` handler.

Feature: cid-cmd-agent-flow-platform (tasks 9.4 - 9.12)

The handler under test is ``Cid.create_agent`` in ``cid/common.py``. Tests call the
UNDECORATED handler via ``Cid.create_agent.__wrapped__`` (the ``@command`` decorator
uses ``functools.wraps``), so no AWS login or catalog loading is performed.

Harness: ``Cid.__new__(Cid)`` with injected ``resources``, a fake ``base``
(SimpleNamespace), and MagicMock ``qs`` / ``space`` / ``agent`` /
``parameters_controller`` helpers written into ``__dict__`` to preempt the
``cached_property`` descriptors. Real exception subclasses are attached to the
mocked client ``exceptions`` namespaces. No real AWS call is ever made.
"""
import io
import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

import cid.utils as cid_utils
from cid.common import Cid
from cid.exceptions import CidError, CidCritical
from cid.helpers.quicksight.agent_logic import (
    CID_MANAGED_MARKER,
    CID_PROVENANCE_TAG_KEY,
    CID_PROVENANCE_TAG_VALUE,
)

ACCOUNT_ID = '123456789012'
REGION = 'us-east-1'
PARTITION = 'aws'
DOMAIN = 'aws.amazon.com'
USERNAME = 'test-user'
USER_ARN = f'arn:aws:quicksight:{REGION}:{ACCOUNT_ID}:user/default/{USERNAME}'
AGENT_ID = 'test-agent'
SPACE_KEY = 'test-space'
SPACE_ARN = f'arn:{PARTITION}:quicksight:{REGION}:{ACCOUNT_ID}:space/{SPACE_KEY}'
AGENT_ARN = f'arn:{PARTITION}:quicksight:{REGION}:{ACCOUNT_ID}:agent/{AGENT_ID}'

# the undecorated handler (the @command decorator uses functools.wraps)
create_agent = Cid.create_agent.__wrapped__

# qs methods a prerequisite-only create-agent run may legitimately call (read-only)
QS_READ_ONLY_ALLOWED = {'ensure_subscription', 'describe_user', 'describe_group', 'list_dashboards'}
# agent-helper methods that mutate the target agent
AGENT_MUTATING_METHODS = ('create_or_update', 'delete', 'grant_owner', 'write_provenance', 'wait_active')


def dash_id(key):
    return f'{key}-id'


def dash_arn(key):
    return f'arn:aws:quicksight:{REGION}:{ACCOUNT_ID}:dashboard/{key}-id'


def dataset_arn(key):
    return f'arn:aws:quicksight:{REGION}:{ACCOUNT_ID}:dataset/{key}-id'


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


def make_definition(required=(), optional=(), datasets=(), knowledge_bases=(),
                    lifecycle='PUBLISHED', name='Test Agent', agent_id=AGENT_ID,
                    space_keys=(SPACE_KEY,)):
    """Build a cap-valid catalog agent entry."""
    depends = {}
    if space_keys:
        depends['spaces'] = list(space_keys)
    if required:
        depends['dashboards'] = list(required)
    if optional:
        depends['optionalDashboards'] = list(optional)
    if datasets:
        depends['datasets'] = list(datasets)
    if knowledge_bases:
        depends['knowledgeBases'] = list(knowledge_bases)
    return {
        'name': name,
        'agentId': agent_id,
        'category': 'FinOps',
        'persona': {
            'identity': 'Cost optimization advisor',
            'customInstructions': 'Answer using the attached dashboards.',
            'tone': 'Professional',
            'outputStyle': 'Concise text',
            'responseLength': 'Short answers',
        },
        'starterPrompts': ['What is my spend?'],
        'welcomeMessage': 'Welcome to the advisor.',
        'lifecycle': lifecycle,
        'dependsOn': depends,
    }


def make_cid(definition, present=(), present_datasets=(), user_role='AUTHOR_PRO',
             existing_agent=None, resource_tags=()):
    """Build a Cid harness with mocked qs/space/agent/parameters_controller.

    :param definition: the catalog agent entry (registered under its agentId)
    :param present: dependency dashboard catalog keys deployed in the account
    :param present_datasets: dependency dataset catalog keys deployed in the account
    :param user_role: role returned by qs.describe_user
    :param existing_agent: DescribeAgent 'Agent' dict returned by agent.get, or None
    :param resource_tags: Tags list returned by list_tags_for_resource (provenance)
    """
    cid_obj = Cid.__new__(Cid)
    cid_obj.__dict__.clear()
    excs = make_exceptions()

    depends = definition.get('dependsOn') or {}
    resources = {'agents': {definition['agentId']: definition},
                 'spaces': {SPACE_KEY: {'name': 'Test Space', 'description': 'CID test space'}},
                 'dashboards': {}, 'datasets': {}}
    for key in list(depends.get('dashboards') or []) + list(depends.get('optionalDashboards') or []):
        resources['dashboards'][key] = {'dashboardId': dash_id(key)}
    for key in depends.get('datasets') or []:
        resources['datasets'][key] = {'data': {'DataSetId': f'{key}-id'}}
    cid_obj.resources = resources

    cid_obj.base = SimpleNamespace(
        account_id=ACCOUNT_ID, region=REGION, partition=PARTITION, domain=DOMAIN,
        username=USERNAME, session=MagicMock(name='session'),
    )

    qs = MagicMock(name='qs')
    qs.ensure_subscription.return_value = None
    qs.describe_user.return_value = {'Arn': USER_ARN, 'Role': user_role, 'UserName': USERNAME}
    qs.list_dashboards.return_value = [
        {'DashboardId': dash_id(key), 'Arn': dash_arn(key)} for key in present
    ]
    qs.datasets = {f'{key}-id': SimpleNamespace(arn=dataset_arn(key)) for key in present_datasets}
    cid_obj.__dict__['qs'] = qs

    space = MagicMock(name='space')
    space.client.exceptions = excs
    space.client.meta.method_to_api_mapping = {'create_space': 'CreateSpace', 'create_agent': 'CreateAgent'}
    space.client.list_tags_for_resource.return_value = {'Tags': list(resource_tags)}
    space.get.return_value = None
    space.create_or_update.return_value = SPACE_ARN
    space.find_by_name.return_value = []
    space.update_resources.return_value = []
    cid_obj.__dict__['space'] = space

    agent = MagicMock(name='agent')
    agent.client.exceptions = excs
    agent.client.list_agents.return_value = {'AgentsSummaries': []}  # enumeration-only, no NextToken
    agent.get.return_value = existing_agent
    agent.create_or_update.return_value = {
        'agentId': definition['agentId'], 'arn': AGENT_ARN,
        'action': 'created' if existing_agent is None else 'updated',
    }
    cid_obj.__dict__['agent'] = agent

    parameters_controller = MagicMock(name='parameters_controller')
    parameters_controller.load_parameters.return_value = {}
    cid_obj.__dict__['parameters_controller'] = parameters_controller

    # deploy-engine / data-layer sentinels: any call on these is out of scope
    for sentinel in ('athena', 'glue', 'organizations', 's3', 'cur1', 'cur2', 'iam', 'cfn'):
        cid_obj.__dict__[sentinel] = MagicMock(name=sentinel)
    return cid_obj


def run(cid_obj, **kwargs):
    """Run the undecorated handler capturing stdout; returns (result, output)."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = create_agent(cid_obj, **kwargs)
    return result, buffer.getvalue()


def run_raising(cid_obj, exception_type, **kwargs):
    """Run the handler expecting an exception; returns (excinfo, output)."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        with pytest.raises(exception_type) as excinfo:
            create_agent(cid_obj, **kwargs)
    return excinfo, buffer.getvalue()


def dashboard_update_calls(cid_obj):
    """The space.update_resources calls with resource_type='DASHBOARD'."""
    return [call for call in cid_obj.space.update_resources.call_args_list
            if call.kwargs.get('resource_type') == 'DASHBOARD']


def assert_nothing_created(cid_obj):
    """No Space or Agent create/update/grant/provenance call was made."""
    assert not cid_obj.space.create_or_update.called
    assert not cid_obj.space.update_resources.called
    assert not cid_obj.space.grant_owner.called
    assert not cid_obj.space.write_provenance.called
    for method in AGENT_MUTATING_METHODS:
        assert not getattr(cid_obj.agent, method).called


def assert_no_deploy_or_data_layer_calls(cid_obj):
    """Property 13 core assertion: zero deploy-engine and zero data-layer calls."""
    qs_methods = {call[0].split('.')[0] for call in cid_obj.qs.method_calls}
    assert qs_methods <= QS_READ_ONLY_ALLOWED, (
        f'create-agent must only issue read-only qs calls, got: {qs_methods - QS_READ_ONLY_ALLOWED}')
    for sentinel in ('athena', 'glue', 'organizations', 's3', 'cur1', 'cur2', 'iam', 'cfn'):
        helper = cid_obj.__dict__[sentinel]
        assert not helper.mock_calls, f'data-layer helper {sentinel!r} must never be touched: {helper.mock_calls}'


# ===========================================================================
# Task 9.4: mock-based unit tests for the create flow
# ===========================================================================
class TestErrorPropagation:
    """: CidError/CidCritical propagate cleanly out of the handler."""

    def test_cid_error_from_helper_propagates_unchanged(self):
        cid_obj = make_cid(make_definition(required=('dash-a',)), present=('dash-a',))
        original = CidError('helper-level error')
        cid_obj.agent.create_or_update.side_effect = original
        excinfo, _ = run_raising(cid_obj, CidError, agent_id=AGENT_ID)
        assert excinfo.value is original
        # CidError is re-raised directly, not routed through cleanup
        assert not cid_obj.agent.delete.called

    def test_cid_critical_from_helper_propagates_unchanged(self):
        cid_obj = make_cid(make_definition(required=('dash-a',)), present=('dash-a',))
        original = CidCritical('critical helper error')
        cid_obj.agent.create_or_update.side_effect = original
        excinfo, _ = run_raising(cid_obj, CidCritical, agent_id=AGENT_ID)
        assert excinfo.value is original
        assert not cid_obj.agent.delete.called


class TestSubscriptionDetectAndRequire:
    """: subscription failure is CidCritical and never activates."""

    def test_subscription_failure_raises_cid_critical_and_creates_nothing(self):
        cid_obj = make_cid(make_definition(required=('dash-a',)), present=('dash-a',))
        cid_obj.qs.ensure_subscription.side_effect = CidCritical('subscription not active')
        excinfo, _ = run_raising(cid_obj, CidCritical, agent_id=AGENT_ID)
        message = str(excinfo.value)
        assert 'Enterprise' in message
        assert 'will not activate' in message
        assert_nothing_created(cid_obj)
        # read-only detection: ensure_subscription is the ONLY qs call, no activation
        assert [call[0] for call in cid_obj.qs.method_calls] == ['ensure_subscription']


class TestCleanupOnFailure:
    """: best-effort delete of the partially created agent, then re-raise."""

    def test_generic_create_failure_triggers_cleanup_delete_and_reraises(self):
        cid_obj = make_cid(make_definition(required=('dash-a',)), present=('dash-a',))
        cid_obj.agent.create_or_update.side_effect = RuntimeError('boom')
        excinfo, _ = run_raising(cid_obj, RuntimeError, agent_id=AGENT_ID)
        assert 'boom' in str(excinfo.value)
        # cleanup delete goes through Agent.delete, which retries ConflictException
        # internally (, covered by the Agent helper tests)
        cid_obj.agent.delete.assert_called_once_with(AGENT_ID)

    def test_cleanup_delete_failure_is_tolerated_and_original_error_reraised(self):
        """ plumbing: even when the ConflictException retry loop ultimately
        fails, the cleanup stays best-effort and the original error surfaces."""
        cid_obj = make_cid(make_definition(required=('dash-a',)), present=('dash-a',))
        cid_obj.agent.create_or_update.side_effect = RuntimeError('boom')
        cid_obj.agent.delete.side_effect = CidError('agent still busy (ConflictException) after 300 seconds')
        excinfo, _ = run_raising(cid_obj, RuntimeError, agent_id=AGENT_ID)
        assert 'boom' in str(excinfo.value)
        cid_obj.agent.delete.assert_called_once_with(AGENT_ID)

    def test_no_cleanup_delete_when_agent_pre_existed(self):
        """A pre-existing CID-managed agent is never deleted by cleanup."""
        existing = {'Arn': AGENT_ARN, 'Description': f'mine {CID_MANAGED_MARKER}'}
        cid_obj = make_cid(make_definition(required=('dash-a',)), present=('dash-a',),
                           existing_agent=existing)
        cid_obj.agent.create_or_update.side_effect = RuntimeError('boom')
        run_raising(cid_obj, RuntimeError, agent_id=AGENT_ID)
        assert not cid_obj.agent.delete.called


class TestResourceExistsSuccessPlumbing:
    """: ResourceExistsException during create is success.

    The tolerate-ResourceExists behavior lives in Agent.create_or_update (covered by
    test_agent_helper.py); here we assert the handler treats the helper's 'created'
    result after a tolerated ResourceExistsException as full success.
    """

    def test_created_result_after_tolerated_resource_exists_completes_the_flow(self):
        cid_obj = make_cid(make_definition(required=('dash-a',)), present=('dash-a',))
        # exactly what Agent.create_or_update returns after a tolerated ResourceExists
        cid_obj.agent.create_or_update.return_value = {
            'agentId': AGENT_ID, 'arn': AGENT_ARN, 'action': 'created'}
        result, output = run(cid_obj, agent_id=AGENT_ID)
        assert result == AGENT_ID
        assert 'Congratulations' in output
        cid_obj.agent.write_provenance.assert_called_once_with(AGENT_ARN, AGENT_ID)
        assert not cid_obj.agent.delete.called


class TestBringYourOwnSpace:
    """: --space select/create and SearchSpaces resolution."""

    def test_single_name_match_reuses_the_existing_space(self):
        """: a single SearchSpaces match is reused, not re-created."""
        cid_obj = make_cid(make_definition(required=('dash-a',)), present=('dash-a',))
        existing_arn = f'arn:aws:quicksight:{REGION}:{ACCOUNT_ID}:space/existing-space'
        cid_obj.space.find_by_name.return_value = ['existing-space']
        cid_obj.space.get.return_value = {
            'spaceArn': existing_arn, 'name': 'My Space',
            'description': f'managed {CID_MANAGED_MARKER}',
        }
        result, _ = run(cid_obj, agent_id=AGENT_ID, space_name='My Space')
        assert result == AGENT_ID
        cid_obj.space.find_by_name.assert_called_once_with('My Space')
        # CID-managed reuse path updates the existing space id, never a derived one
        assert cid_obj.space.create_or_update.call_args.args[0] == 'existing-space'
        assert dashboard_update_calls(cid_obj)[0].args[0] == 'existing-space'

    def test_no_name_match_creates_space_with_derived_id(self):
        """: non-existent --space name creates a Space with derive_space_id."""
        cid_obj = make_cid(make_definition(required=('dash-a',)), present=('dash-a',))
        cid_obj.space.find_by_name.return_value = []
        result, _ = run(cid_obj, agent_id=AGENT_ID, space_name='My Space')
        assert result == AGENT_ID
        args = cid_obj.space.create_or_update.call_args
        assert args.args[0] == 'My-Space'   # deterministic sanitized id
        assert args.args[1] == 'My Space'   # display name preserved
        cid_obj.space.write_provenance.assert_called_once()
        cid_obj.space.grant_owner.assert_called_once_with('My-Space', USER_ARN)

    def test_ambiguous_name_match_raises_cid_error_listing_ids(self):
        """: multiple exact matches raise a CidError listing the matching ids."""
        cid_obj = make_cid(make_definition(required=('dash-a',)), present=('dash-a',))
        cid_obj.space.find_by_name.return_value = ['space-one', 'space-two']
        excinfo, _ = run_raising(cid_obj, CidError, agent_id=AGENT_ID, space_name='My Space')
        message = str(excinfo.value)
        assert 'space-one' in message and 'space-two' in message
        assert not cid_obj.space.create_or_update.called
        assert not cid_obj.agent.create_or_update.called


class TestKnowledgeBaseAttachment:
    """: valid KB ARNs attach; unresolvable references warn and skip."""

    def test_valid_arns_attach_and_non_arn_reference_warns(self):
        kb_arn = f'arn:aws:quicksight:{REGION}:{ACCOUNT_ID}:knowledge-base/kb-one'
        definition = make_definition(required=('dash-a',),
                                     knowledge_bases=(kb_arn, 'my-kb-note'))
        cid_obj = make_cid(definition, present=('dash-a',))
        result, output = run(cid_obj, agent_id=AGENT_ID)
        assert result == AGENT_ID
        kb_calls = [call for call in cid_obj.space.update_resources.call_args_list
                    if call.kwargs.get('resource_type') == 'KNOWLEDGE_BASE']
        assert len(kb_calls) == 1
        assert kb_calls[0].args[1] == [kb_arn]
        assert 'my-kb-note' in output and 'Warning' in output

    def test_no_kb_update_call_when_nothing_resolvable(self):
        definition = make_definition(required=('dash-a',), knowledge_bases=('not-an-arn',))
        cid_obj = make_cid(definition, present=('dash-a',))
        run(cid_obj, agent_id=AGENT_ID)
        kb_calls = [call for call in cid_obj.space.update_resources.call_args_list
                    if call.kwargs.get('resource_type') == 'KNOWLEDGE_BASE']
        assert kb_calls == []


class TestPrincipalResolution:
    """: Author Pro entitlement and registered-principal checks."""

    def test_missing_author_pro_role_raises_cid_critical(self):
        cid_obj = make_cid(make_definition(required=('dash-a',)), present=('dash-a',),
                           user_role='AUTHOR')
        excinfo, _ = run_raising(cid_obj, CidCritical, agent_id=AGENT_ID)
        assert 'Author Pro' in str(excinfo.value)
        assert_nothing_created(cid_obj)

    def test_unregistered_principal_raises_before_any_grant(self):
        cid_obj = make_cid(make_definition(required=('dash-a',)), present=('dash-a',))
        cid_obj.qs.describe_user.return_value = None
        excinfo, _ = run_raising(cid_obj, CidCritical, agent_id=AGENT_ID)
        message = str(excinfo.value)
        assert USERNAME in message
        assert 'register' in message.lower()
        assert not cid_obj.space.grant_owner.called
        assert not cid_obj.agent.grant_owner.called
        assert_nothing_created(cid_obj)


class TestGenAiAvailabilityAndAccessDenied:
    """: gen-AI availability pre-check and AccessDenied handling."""

    def test_missing_genai_operations_raise_cid_critical_naming_region_and_partition(self):
        cid_obj = make_cid(make_definition(required=('dash-a',)), present=('dash-a',))
        cid_obj.space.client.meta.method_to_api_mapping = {'describe_dashboard': 'DescribeDashboard'}
        excinfo, _ = run_raising(cid_obj, CidCritical, agent_id=AGENT_ID)
        message = str(excinfo.value)
        assert REGION in message and PARTITION in message
        assert_nothing_created(cid_obj)

    def test_access_denied_raises_cid_critical_naming_action_and_permission(self):
        cid_obj = make_cid(make_definition(required=('dash-a',)), present=('dash-a',))
        denied = cid_obj.space.client.exceptions.AccessDeniedException('denied')
        denied.operation_name = 'UpdateSpaceResources'
        cid_obj.space.update_resources.side_effect = denied
        excinfo, _ = run_raising(cid_obj, CidCritical, agent_id=AGENT_ID)
        message = str(excinfo.value)
        assert 'UpdateSpaceResources' in message
        assert 'quicksight:UpdateSpaceResources' in message
        # no further create or modify calls after the denial
        assert not cid_obj.agent.create_or_update.called
        assert not cid_obj.agent.grant_owner.called


class TestParameterPersistence:
    """: per-agent parameter persistence."""

    def test_parameters_persisted_keyed_by_agent_id(self):
        cid_obj = make_cid(make_definition(required=('dash-a',)), present=('dash-a',))
        run(cid_obj, agent_id=AGENT_ID)
        dump = cid_obj.parameters_controller.dump_parameters
        dump.assert_called_once()
        assert dump.call_args.kwargs.get('context') == AGENT_ID


class TestConflictGuard:
    """: non-CID collisions are refused; CID-managed agents proceed."""

    def test_non_cid_existing_agent_is_refused_without_mutation(self):
        existing = {'Arn': AGENT_ARN, 'Description': 'hand-made agent'}
        cid_obj = make_cid(make_definition(required=('dash-a',)), present=('dash-a',),
                           existing_agent=existing)
        excinfo, _ = run_raising(cid_obj, CidError, agent_id=AGENT_ID)
        assert 'Refusing' in str(excinfo.value)
        for method in AGENT_MUTATING_METHODS:
            assert not getattr(cid_obj.agent, method).called

    def test_cid_managed_existing_agent_proceeds_with_update_path(self):
        existing = {'Arn': AGENT_ARN, 'Description': f'agent {CID_MANAGED_MARKER}'}
        cid_obj = make_cid(make_definition(required=('dash-a',)), present=('dash-a',),
                           existing_agent=existing)
        result, output = run(cid_obj, agent_id=AGENT_ID)
        assert result == AGENT_ID
        cid_obj.agent.create_or_update.assert_called_once()
        assert 'updated' in output


class TestNonCidSpaceReuse:
    """: a non-CID --space is reused additively, never mutated itself."""

    def test_non_cid_space_is_not_renamed_tagged_or_repermissioned(self):
        cid_obj = make_cid(make_definition(required=('dash-a',)), present=('dash-a',))
        shared_arn = f'arn:aws:quicksight:{REGION}:{ACCOUNT_ID}:space/shared-space'
        cid_obj.space.find_by_name.return_value = ['shared-space']
        cid_obj.space.get.return_value = {
            'spaceArn': shared_arn, 'name': 'Shared Space', 'description': 'user space'}
        result, _ = run(cid_obj, agent_id=AGENT_ID, space_name='Shared Space')
        assert result == AGENT_ID
        assert not cid_obj.space.create_or_update.called
        assert not cid_obj.space.write_provenance.called
        assert not cid_obj.space.grant_owner.called
        # resources are still added additively to the reused space
        calls = dashboard_update_calls(cid_obj)
        assert len(calls) == 1 and calls[0].args[0] == 'shared-space'
        assert calls[0].args[1] == [dash_arn('dash-a')]


class TestDatasetKnowledge:
    """: present datasets attach as DATA_SET; missing ones warn; zero creates."""

    def test_present_dataset_attaches_missing_dataset_warns_no_creates(self):
        definition = make_definition(required=('dash-a',),
                                     datasets=('ds-present', 'ds-missing'))
        cid_obj = make_cid(definition, present=('dash-a',), present_datasets=('ds-present',))
        result, output = run(cid_obj, agent_id=AGENT_ID)
        assert result == AGENT_ID
        dataset_calls = [call for call in cid_obj.space.update_resources.call_args_list
                         if call.kwargs.get('resource_type') == 'DATA_SET']
        assert len(dataset_calls) == 1
        assert dataset_calls[0].args[1] == [dataset_arn('ds-present')]
        assert 'ds-missing' in output and 'Warning' in output
        # never create a dataset: only read-only qs calls happened
        qs_methods = {call[0].split('.')[0] for call in cid_obj.qs.method_calls}
        assert qs_methods <= QS_READ_ONLY_ALLOWED

    def test_no_dataset_update_call_when_none_present(self):
        definition = make_definition(required=('dash-a',), datasets=('ds-missing',))
        cid_obj = make_cid(definition, present=('dash-a',))
        run(cid_obj, agent_id=AGENT_ID)
        dataset_calls = [call for call in cid_obj.space.update_resources.call_args_list
                         if call.kwargs.get('resource_type') == 'DATA_SET']
        assert dataset_calls == []


class TestDualProvenanceAndLifecycle:
    """: dual provenance writes and PUBLISHED wait_active."""

    def test_dual_provenance_written_on_created_space_and_agent(self):
        cid_obj = make_cid(make_definition(required=('dash-a',)), present=('dash-a',))
        run(cid_obj, agent_id=AGENT_ID)
        cid_obj.space.write_provenance.assert_called_once_with(SPACE_ARN, SPACE_KEY)
        cid_obj.agent.write_provenance.assert_called_once_with(AGENT_ARN, AGENT_ID)

    def test_wait_active_called_for_published_lifecycle(self):
        cid_obj = make_cid(make_definition(required=('dash-a',), lifecycle='PUBLISHED'),
                           present=('dash-a',))
        run(cid_obj, agent_id=AGENT_ID)
        cid_obj.agent.wait_active.assert_called_once_with(AGENT_ID)

    def test_wait_active_not_called_for_preview_lifecycle(self):
        cid_obj = make_cid(make_definition(required=('dash-a',), lifecycle='PREVIEW'),
                           present=('dash-a',))
        run(cid_obj, agent_id=AGENT_ID)
        assert not cid_obj.agent.wait_active.called

    def test_wait_active_failed_fail_fast_surfaces(self):
        """: FAILED status raised by wait_active surfaces from the handler."""
        cid_obj = make_cid(make_definition(required=('dash-a',), lifecycle='PUBLISHED'),
                           present=('dash-a',))
        cid_obj.agent.wait_active.side_effect = CidError(
            f"Agent {AGENT_ID!r} entered FAILED status while waiting for ACTIVE. "
            "ErrorMessage: model error")
        excinfo, _ = run_raising(cid_obj, CidError, agent_id=AGENT_ID)
        message = str(excinfo.value)
        assert 'FAILED' in message and 'model error' in message


# ===========================================================================
# Property tests (tasks 9.5 - 9.12)
# ===========================================================================
PROPERTY_SETTINGS = settings(
    max_examples=100, deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

# generated dependency states: presence flags for required and optional keys
required_flags = st.lists(st.booleans(), min_size=1, max_size=4)
optional_flags = st.lists(st.booleans(), min_size=0, max_size=3)


def keys_from_flags(req_flags, opt_flags):
    """Derive (required, optional, present) catalog key lists from presence flags."""
    required = [f'req-{i}' for i in range(len(req_flags))]
    optional = [f'opt-{i}' for i in range(len(opt_flags))]
    present = ([key for key, flag in zip(required, req_flags) if flag]
               + [key for key, flag in zip(optional, opt_flags) if flag])
    return required, optional, present


# Feature: cid-cmd-agent-flow-platform, Property 13: create-agent issues zero deploy
# and zero data-layer calls
# **Validates: Requirements 8.2, 8.7**
@PROPERTY_SETTINGS
@given(req_flags=required_flags, opt_flags=optional_flags)
def test_property_13_zero_deploy_and_zero_data_layer_calls(req_flags, opt_flags):
    """Across zero/partial/full dependency states, create-agent only reads via
    ListDashboards and creates or updates the Space and Agent — it never invokes any
    dashboard-deploy engine or data-layer (CUR/Data Exports/Data Collection) call."""
    reset_parameters()
    required, optional, present = keys_from_flags(req_flags, opt_flags)
    definition = make_definition(required=required, optional=optional)
    cid_obj = make_cid(definition, present=present)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        if present:
            create_agent(cid_obj, agent_id=AGENT_ID)
        else:
            with pytest.raises(CidError):  # zero-present guidance error path
                create_agent(cid_obj, agent_id=AGENT_ID)
    assert_no_deploy_or_data_layer_calls(cid_obj)


# Feature: cid-cmd-agent-flow-platform, Property 10: Zero dashboards present raises
# a guidance error and creates nothing
# **Validates: Requirements 8.3, 19.1**
@PROPERTY_SETTINGS
@given(
    n_required=st.integers(min_value=1, max_value=4),
    n_optional=st.integers(min_value=0, max_value=3),
    n_unrelated=st.integers(min_value=0, max_value=3),
)
def test_property_10_zero_present_raises_guidance_error_and_creates_nothing(
        n_required, n_optional, n_unrelated):
    """With every dependency dashboard absent (even when unrelated dashboards are
    deployed), the handler stops with a CidError carrying the CID/CUDOS and Data
    Collection guidance, names no cid-cmd command, and creates nothing."""
    reset_parameters()
    required = [f'req-{i}' for i in range(n_required)]
    optional = [f'opt-{i}' for i in range(n_optional)]
    definition = make_definition(required=required, optional=optional)
    cid_obj = make_cid(definition, present=())
    # unrelated deployed dashboards never satisfy catalog dependencies
    cid_obj.qs.list_dashboards.return_value = [
        {'DashboardId': f'unrelated-{i}',
         'Arn': f'arn:aws:quicksight:{REGION}:{ACCOUNT_ID}:dashboard/unrelated-{i}'}
        for i in range(n_unrelated)
    ]
    excinfo, _ = run_raising(cid_obj, CidError, agent_id=AGENT_ID)
    message = str(excinfo.value)
    assert 'https://catalog.workshops.aws/awscid/en-US' in message   # CID/CUDOS guidance
    assert 'data-collection' in message                              # Data Collection guidance
    assert 'cid-cmd' not in message
    assert_nothing_created(cid_obj)


# Feature: cid-cmd-agent-flow-platform, Property 11: At least one present dashboard
# proceeds using exactly the present set
# **Validates: Requirements 8.4, 19.2, 19.4**
@PROPERTY_SETTINGS
@given(req_flags=required_flags, opt_flags=optional_flags)
def test_property_11_proceeds_attaching_exactly_the_present_set(req_flags, opt_flags):
    """With >=1 dependency dashboard present, the Space update receives exactly
    the present dashboard ARNs (never an absent one) and the Agent is built."""
    if not any(req_flags) and not any(opt_flags):
        req_flags = [True] + list(req_flags[1:])  # force at least one present
    reset_parameters()
    required, optional, present = keys_from_flags(req_flags, opt_flags)
    definition = make_definition(required=required, optional=optional)
    cid_obj = make_cid(definition, present=present)
    result, _ = run(cid_obj, agent_id=AGENT_ID)
    assert result == AGENT_ID
    calls = dashboard_update_calls(cid_obj)
    assert len(calls) == 1
    attached = calls[0].args[1]
    assert set(attached) == {dash_arn(key) for key in present}
    assert len(attached) == len(set(attached))  # de-duplicated
    cid_obj.agent.create_or_update.assert_called_once()


# Feature: cid-cmd-agent-flow-platform, Property 12: Missing required vs missing
# optional emit the correct guidance and never deploy
# **Validates: Requirements 8.5, 8.6, 19.3, 19.5**
@PROPERTY_SETTINGS
@given(req_flags=required_flags, opt_flags=optional_flags)
def test_property_12_missing_dependency_warnings_carry_correct_guidance(req_flags, opt_flags):
    """Each missing Required dashboard warns with the CID/CUDOS guidance, each
    missing Optional dashboard warns with the Data Collection guidance, no warning
    names a cid-cmd command, and nothing is deployed."""
    if not any(req_flags) and not any(opt_flags):
        req_flags = [True] + list(req_flags[1:])  # a proceeding run needs >=1 present
    reset_parameters()
    required, optional, present = keys_from_flags(req_flags, opt_flags)
    missing_required = [key for key in required if key not in present]
    missing_optional = [key for key in optional if key not in present]
    definition = make_definition(required=required, optional=optional)
    cid_obj = make_cid(definition, present=present)
    result, output = run(cid_obj, agent_id=AGENT_ID)
    assert result == AGENT_ID
    lines = output.splitlines()
    for key in missing_required:
        matching = [line for line in lines if key in line and 'required dashboard' in line]
        assert len(matching) == 1, f'expected one required-dashboard warning for {key!r}'
        assert 'CID/CUDOS' in matching[0]
        assert 'cid-cmd' not in matching[0]
    for key in missing_optional:
        matching = [line for line in lines if key in line and 'optional dashboard' in line]
        assert len(matching) == 1, f'expected one optional-dashboard warning for {key!r}'
        assert 'Data Collection' in matching[0]
        assert 'cid-cmd' not in matching[0]
    assert_no_deploy_or_data_layer_calls(cid_obj)


# Feature: cid-cmd-agent-flow-platform, Property 20: Name/id collisions with non-CID
# resources are refused, not mutated
# **Validates: Requirements 13.1, 13.2**
@PROPERTY_SETTINGS
@given(
    has_tag=st.booleans(),
    has_marker=st.booleans(),
    description=st.text(alphabet='abcdefghij 0123456789', max_size=40),
)
def test_property_20_non_cid_agent_collisions_refused_not_mutated(has_tag, has_marker, description):
    """An existing agent with the target id proceeds to the idempotent update path
    if and only if it carries CID-managed provenance (tag OR description marker);
    otherwise it is refused with a CidError and left entirely unmodified."""
    reset_parameters()
    full_description = f'{description} {CID_MANAGED_MARKER}' if has_marker else description
    tags = [{'Key': CID_PROVENANCE_TAG_KEY, 'Value': CID_PROVENANCE_TAG_VALUE}] if has_tag else []
    existing = {'Arn': AGENT_ARN, 'Description': full_description}
    cid_obj = make_cid(make_definition(required=('dash-a',)), present=('dash-a',),
                       existing_agent=existing, resource_tags=tags)
    if has_tag or has_marker:  # CID-managed: idempotent update/no-op path proceeds
        result, _ = run(cid_obj, agent_id=AGENT_ID)
        assert result == AGENT_ID
        cid_obj.agent.create_or_update.assert_called_once()
    else:                      # non-CID: refused, zero mutating calls on the agent
        excinfo, _ = run_raising(cid_obj, CidError, agent_id=AGENT_ID)
        assert 'Refusing' in str(excinfo.value)
        for method in AGENT_MUTATING_METHODS:
            assert not getattr(cid_obj.agent, method).called


# Feature: cid-cmd-agent-flow-platform, Property 21: Reusing a non-CID space tags
# only the newly added resources
# **Validates: Requirements 13.3**
@PROPERTY_SETTINGS
@given(
    space_id=st.text(alphabet='abcdefghij-0123456789', min_size=1, max_size=20),
    description=st.text(alphabet='abcdefghij 0123456789', max_size=40),
    req_flags=required_flags,
)
def test_property_21_non_cid_space_reuse_never_mutates_the_space(space_id, description, req_flags):
    """Reusing a pre-existing non-CID space adds this agent's resources additively
    while the space itself is never renamed, re-described, tagged (write_provenance),
    or re-permissioned (grant_owner)."""
    if not any(req_flags):
        req_flags = [True] + list(req_flags[1:])
    reset_parameters()
    required, _, present = keys_from_flags(req_flags, [])
    definition = make_definition(required=required)
    cid_obj = make_cid(definition, present=present)
    cid_obj.space.find_by_name.return_value = [space_id]
    cid_obj.space.get.return_value = {
        'spaceArn': f'arn:aws:quicksight:{REGION}:{ACCOUNT_ID}:space/{space_id}',
        'name': 'Shared Space',
        'description': description,  # never carries the CID-managed marker
    }
    result, _ = run(cid_obj, agent_id=AGENT_ID, space_name='Shared Space')
    assert result == AGENT_ID
    # the space itself is never mutated
    assert not cid_obj.space.create_or_update.called
    assert not cid_obj.space.write_provenance.called
    assert not cid_obj.space.grant_owner.called
    # while this agent's resources are still added additively
    calls = dashboard_update_calls(cid_obj)
    assert len(calls) == 1
    assert calls[0].args[0] == space_id
    assert set(calls[0].args[1]) == {dash_arn(key) for key in present}


# Feature: cid-cmd-agent-flow-platform, Property 24: Successful creation completes
# cleanly and reports success
@PROPERTY_SETTINGS
@given(
    name=st.text(alphabet='abcdefghijklmnopqrstuvwxyz ABCDEFG', min_size=1, max_size=40)
        .filter(lambda text: text.strip()),
    lifecycle=st.sampled_from(['PUBLISHED', 'PREVIEW']),
)
def test_property_24_creation_completes_and_reports_success(name, lifecycle):
    """For any successful creation, the flow completes, returns the agent id, and
    reports success."""
    reset_parameters()
    definition = make_definition(required=('dash-a',), name=name, lifecycle=lifecycle)
    cid_obj = make_cid(definition, present=('dash-a',))
    result, output = run(cid_obj, agent_id=AGENT_ID)
    assert result == AGENT_ID
    cid_obj.agent.create_or_update.assert_called_once()
    assert 'Congratulations' in output


# Feature: cid-cmd-agent-flow-platform, Property 26: Missing required input in
# non-interactive mode is reported by name
# **Validates: Requirements 17.2**
@PROPERTY_SETTINGS
@given(
    agent_keys=st.lists(
        st.text(alphabet='abcdefghij', min_size=1, max_size=10),
        min_size=1, max_size=3, unique=True,
    ),
)
def test_property_26_missing_agent_id_reported_by_name_non_interactively(agent_keys):
    """With no --agent-id, no stored default, and no fallback in a non-interactive
    environment, resolution raises an error naming the missing 'agent-id' input."""
    reset_parameters()
    definition = make_definition(required=('dash-a',))
    cid_obj = make_cid(definition, present=('dash-a',))
    cid_obj.resources['agents'] = {
        key: {'name': key.title(), 'agentId': key, 'category': 'FinOps'}
        for key in agent_keys
    }
    buffer = io.StringIO()
    with patch('cid.utils.isatty', return_value=False):
        with contextlib.redirect_stdout(buffer):
            with pytest.raises((CidError, CidCritical)) as excinfo:
                create_agent(cid_obj)  # no agent_id supplied
    assert 'agent-id' in str(excinfo.value)
    assert_nothing_created(cid_obj)
