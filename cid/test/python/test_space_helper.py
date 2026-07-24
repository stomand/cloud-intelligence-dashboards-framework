"""Mock-based unit tests and property tests for the Space helper
(cid/helpers/quicksight/space.py).

Feature: cid-cmd-agent-flow-platform
- Task 5.3: mock-based unit tests (Requirements 5.1, 5.2, 5.4, 6.5, 6.7, 6.8, 6.9, 9.6, 13.4)
- Task 5.4: Property 17 (Space portion) owner-grant action set (Requirements 6.9, 6.11)

All tests mock the boto3 quicksight client; no I/O is performed.
"""
import logging
from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings, strategies as st

from cid.base import CidBase
from cid.helpers.quicksight.agent_logic import (
    CID_MANAGED_MARKER,
    CID_PROVENANCE_TAG_KEY,
    CID_PROVENANCE_TAG_VALUE,
)
from cid.helpers.quicksight.space import SPACE_OWNER_ACTIONS, Space

ACCOUNT_ID = '123456789012'


def make_space(resources=None):
    """Build a Space over a fully mocked boto3 session/client.

    ``client.exceptions.*`` must be real exception classes so ``except`` clauses work.
    """
    client = MagicMock()
    client.exceptions.ResourceNotFoundException = type('ResourceNotFoundException', (Exception,), {})
    client.exceptions.ResourceExistsException = type('ResourceExistsException', (Exception,), {})
    client.exceptions.AccessDeniedException = type('AccessDeniedException', (Exception,), {})
    client.exceptions.ClientError = type('ClientError', (Exception,), {})
    session = MagicMock()
    session.client.return_value = client
    session.region_name = 'us-east-1'
    session.get_partition_for_region.return_value = 'aws'
    space = Space(session, resources=resources)
    space.awsIdentity = {'Account': ACCOUNT_ID}  # avoid the STS call in CidBase
    return space, client, session


# ---------------------------------------------------------------------------
# CidBase subclass + quicksight client wiring (Requirements 5.1, 5.2)
# ---------------------------------------------------------------------------
def test_space_subclasses_cidbase():
    assert issubclass(Space, CidBase)


def test_constructor_obtains_quicksight_client_from_session():
    space, client, session = make_space()
    session.client.assert_called_once_with('quicksight')
    assert space.client is client


def test_constructor_stores_resources():
    resources = {'agents': {'finops': {}}}
    space, _, _ = make_space(resources=resources)
    assert space.resources == resources
    default_space, _, _ = make_space()
    assert default_space.resources == {}


# ---------------------------------------------------------------------------
# create vs reuse (Requirement 6.5)
# ---------------------------------------------------------------------------
def test_create_or_update_creates_when_space_does_not_exist():
    space, client, _ = make_space()
    client.describe_space.side_effect = client.exceptions.ResourceNotFoundException()
    arn = f'arn:aws:quicksight:us-east-1:{ACCOUNT_ID}:space/my-space'
    client.create_space.return_value = {'spaceArn': arn}

    result = space.create_or_update('my-space', 'My Space', description='desc')

    assert result == arn
    client.create_space.assert_called_once_with(
        AwsAccountId=ACCOUNT_ID, SpaceId='my-space', Name='My Space', Description='desc',
    )
    client.update_space.assert_not_called()


def test_create_or_update_updates_when_space_exists():
    space, client, _ = make_space()
    client.describe_space.return_value = {'Space': {'name': 'Old Name'}}
    arn = f'arn:aws:quicksight:us-east-1:{ACCOUNT_ID}:space/my-space'
    client.update_space.return_value = {'spaceArn': arn}

    result = space.create_or_update('my-space', 'New Name')

    assert result == arn
    client.update_space.assert_called_once_with(
        AwsAccountId=ACCOUNT_ID, SpaceId='my-space', Name='New Name',
    )
    client.create_space.assert_not_called()


def test_create_or_update_falls_back_to_update_on_create_race():
    space, client, _ = make_space()
    client.describe_space.side_effect = client.exceptions.ResourceNotFoundException()
    client.create_space.side_effect = client.exceptions.ResourceExistsException()
    arn = f'arn:aws:quicksight:us-east-1:{ACCOUNT_ID}:space/my-space'
    client.update_space.return_value = {'spaceArn': arn}

    result = space.create_or_update('my-space', 'My Space')

    assert result == arn
    client.create_space.assert_called_once()
    client.update_space.assert_called_once()


# ---------------------------------------------------------------------------
# additive add-only space updates (Requirement 6.7)
# ---------------------------------------------------------------------------
def _arn(suffix):
    return f'arn:aws:quicksight:us-east-1:{ACCOUNT_ID}:dashboard/{suffix}'


def test_update_resources_adds_only_missing_arns():
    space, client, _ = make_space()
    existing = _arn('already-there')
    new = _arn('new-dashboard')
    client.list_space_resources.return_value = {
        'Resources': [{'ResourceType': 'DASHBOARD', 'ResourceDetails': {'resourceArn': existing}}],
    }
    client.update_space_resources.return_value = {'FailedResourceOperations': []}

    failed = space.update_resources('my-space', [existing, new])

    assert failed == []
    client.update_space_resources.assert_called_once()
    kwargs = client.update_space_resources.call_args.kwargs
    added_arns = [item['ResourceDetails']['resourceArn'] for item in kwargs['AddResources']]
    assert added_arns == [new]  # only the ARN not already present
    assert 'RemoveResources' not in kwargs  # strictly additive by default


def test_update_resources_noop_when_everything_present():
    space, client, _ = make_space()
    existing = _arn('already-there')
    client.list_space_resources.return_value = {
        'Resources': [{'ResourceType': 'DASHBOARD', 'ResourceDetails': {'resourceArn': existing}}],
    }

    failed = space.update_resources('my-space', [existing])

    assert failed == []
    client.update_space_resources.assert_not_called()


# ---------------------------------------------------------------------------
# partial FailedResourceOperations: warn + continue (Requirement 6.8)
# ---------------------------------------------------------------------------
def test_update_resources_reports_partial_failures_and_continues(caplog):
    space, client, _ = make_space()
    bad = _arn('denied-dashboard')
    good = _arn('fine-dashboard')
    client.list_space_resources.return_value = {'Resources': []}
    failure = {
        'ResourceType': 'DASHBOARD',
        'ResourceDetails': {'resourceArn': bad},
        'ErrorMessage': 'Access denied',
    }
    client.update_space_resources.return_value = {'FailedResourceOperations': [failure]}

    with caplog.at_level(logging.WARNING, logger='cid.helpers.quicksight.space'):
        failed = space.update_resources('my-space', [good, bad])

    assert failed == [failure]  # the failed list is returned to the caller
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any(bad in r.getMessage() and 'Access denied' in r.getMessage() for r in warnings)


# ---------------------------------------------------------------------------
# SPACE_OWNER_ACTIONS grant (Requirement 6.9)
# ---------------------------------------------------------------------------
def test_grant_owner_sends_exactly_the_16_action_owner_set():
    space, client, _ = make_space()
    principal = f'arn:aws:quicksight:us-east-1:{ACCOUNT_ID}:user/default/admin'

    space.grant_owner('my-space', principal)

    client.update_space_permissions.assert_called_once()
    kwargs = client.update_space_permissions.call_args.kwargs
    assert kwargs['AwsAccountId'] == ACCOUNT_ID
    assert kwargs['SpaceId'] == 'my-space'
    grants = kwargs['GrantPermissions']
    assert len(grants) == 1
    assert grants[0]['Principal'] == principal
    assert set(grants[0]['Actions']) == set(SPACE_OWNER_ACTIONS)
    assert len(grants[0]['Actions']) == 16


# ---------------------------------------------------------------------------
# find_by_name via SearchSpaces (Requirement 9.6)
# ---------------------------------------------------------------------------
def test_find_by_name_single_match():
    space, client, _ = make_space()
    client.search_spaces.return_value = {'SpaceSummaries': [{'spaceId': 'space-1'}]}

    assert space.find_by_name('My Space') == ['space-1']
    kwargs = client.search_spaces.call_args.kwargs
    assert kwargs['Filters'] == [{'name': 'SPACE_NAME', 'operator': 'STRING_EQUALS', 'value': 'My Space'}]


def test_find_by_name_multiple_matches():
    space, client, _ = make_space()
    client.search_spaces.return_value = {
        'SpaceSummaries': [{'spaceId': 'space-1'}, {'spaceId': 'space-2'}],
    }

    assert space.find_by_name('My Space') == ['space-1', 'space-2']


def test_find_by_name_zero_matches():
    space, client, _ = make_space()
    client.search_spaces.return_value = {'SpaceSummaries': []}

    assert space.find_by_name('No Such Space') == []


# ---------------------------------------------------------------------------
# Dual-provenance write (Requirement 13.4)
# ---------------------------------------------------------------------------
def test_write_provenance_tags_and_writes_description_marker():
    space, client, _ = make_space()
    space_arn = f'arn:aws:quicksight:us-east-1:{ACCOUNT_ID}:space/my-space'
    client.describe_space.return_value = {
        'Space': {'name': 'My Space', 'description': 'FinOps knowledge'},
    }

    space.write_provenance(space_arn, 'my-space')

    client.tag_resource.assert_called_once_with(
        ResourceArn=space_arn,
        Tags=[{'Key': CID_PROVENANCE_TAG_KEY, 'Value': CID_PROVENANCE_TAG_VALUE}],
    )
    client.update_space.assert_called_once()
    kwargs = client.update_space.call_args.kwargs
    assert kwargs['SpaceId'] == 'my-space'
    assert kwargs['Name'] == 'My Space'
    assert CID_MANAGED_MARKER in kwargs['Description']
    assert kwargs['Description'].startswith('FinOps knowledge')  # original description preserved


def test_write_provenance_tolerates_tag_failure_and_still_writes_marker():
    space, client, _ = make_space()
    space_arn = f'arn:aws:quicksight:us-east-1:{ACCOUNT_ID}:space/my-space'
    client.tag_resource.side_effect = client.exceptions.AccessDeniedException()
    client.describe_space.return_value = {'Space': {'name': 'My Space', 'description': ''}}

    space.write_provenance(space_arn, 'my-space')  # must not raise

    client.update_space.assert_called_once()
    kwargs = client.update_space.call_args.kwargs
    assert CID_MANAGED_MARKER in kwargs['Description']


def test_write_provenance_skips_update_when_marker_already_present():
    space, client, _ = make_space()
    space_arn = f'arn:aws:quicksight:us-east-1:{ACCOUNT_ID}:space/my-space'
    client.describe_space.return_value = {
        'Space': {'name': 'My Space', 'description': f'Something. {CID_MANAGED_MARKER}'},
    }

    space.write_provenance(space_arn, 'my-space')

    client.update_space.assert_not_called()


# ---------------------------------------------------------------------------
# Feature: cid-cmd-agent-flow-platform, Property 17: Owner grants use exactly the
# canonical action sets (Space portion)
# **Validates: Requirements 6.9, 6.11**
# ---------------------------------------------------------------------------
_space_ids = st.text(
    alphabet='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_=.+',
    min_size=1, max_size=40,
)
_principal_arns = st.text(min_size=1, max_size=60).map(
    lambda s: f'arn:aws:quicksight:us-east-1:{ACCOUNT_ID}:user/default/{s}'
)


@settings(max_examples=100, deadline=None)
@given(space_id=_space_ids, principal_arn=_principal_arns)
def test_property_17_space_owner_grant_uses_exactly_the_canonical_action_set(space_id, principal_arn):
    space, client, _ = make_space()

    space.grant_owner(space_id, principal_arn)

    client.update_space_permissions.assert_called_once()
    kwargs = client.update_space_permissions.call_args.kwargs
    grants = kwargs['GrantPermissions']
    assert len(grants) == 1
    assert grants[0]['Principal'] == principal_arn
    actions = grants[0]['Actions']
    # exactly the 16 canonical owner actions: no more, no fewer, no duplicates
    assert len(actions) == 16
    assert set(actions) == set(SPACE_OWNER_ACTIONS)
    assert kwargs['SpaceId'] == space_id


# ---------------------------------------------------------------------------
# Regression: write_provenance retries through ConflictException (defensive
# mirror of the Agent.write_provenance live-bug fix — pattern)
# ---------------------------------------------------------------------------
def test_write_provenance_conflict_settles_then_marker_written():
    """UpdateSpace ConflictException while the Space settles is retried until
    the Description marker write succeeds (must not raise)."""
    from unittest.mock import patch

    space, client, _ = make_space()
    client.exceptions.ConflictException = type('ConflictException', (Exception,), {})
    space_arn = f'arn:aws:quicksight:us-east-1:{ACCOUNT_ID}:space/my-space'
    client.describe_space.return_value = {
        'Space': {'name': 'My Space', 'description': 'FinOps knowledge'},
    }
    client.update_space.side_effect = [
        client.exceptions.ConflictException('Cannot update Space my-space in UPDATING status'),
        client.exceptions.ConflictException('Cannot update Space my-space in UPDATING status'),
        None,
    ]

    with patch('cid.helpers.quicksight.space.time.sleep') as mock_sleep:
        space.write_provenance(space_arn, 'my-space')  # must not raise

    assert client.update_space.call_count == 3
    assert mock_sleep.call_count == 2
    kwargs = client.update_space.call_args.kwargs
    assert kwargs['Name'] == 'My Space'
    assert CID_MANAGED_MARKER in kwargs['Description']
    assert kwargs['Description'].startswith('FinOps knowledge')
