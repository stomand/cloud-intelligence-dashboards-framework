"""Tests for Cid.get_definition type allowlist and agent/space template substitution.

Feature: cid-cmd-agent-flow-platform

Covers:
- Property 5: Invalid definition types are rejected (Requirements 4.4)
- Unit tests for agent/space ${var} substitution and unresolved-token errors
  (Requirements 4.2, 4.3)

Cid is constructed without __init__ (no AWS session needed); only the attributes
that get_definition touches are provided, and get_template_parameters is stubbed
with a controlled parameter map so the substitution and unresolved-token logic in
get_definition itself is what is under test.
"""
import pytest
from hypothesis import given, settings, strategies as st

from cid.common import Cid

ACCEPTED_TYPES = ('dashboard', 'dataset', 'view', 'schedule', 'crawler', 'agent', 'space')

# Controlled parameter map used in place of Cid.get_template_parameters.
KNOWN_PARAMS = {'kvar': 'resolved-value', 'kother': 'other-value'}


def _make_cid(resources=None, params=None):
    """Build a Cid without __init__ and stub get_template_parameters."""
    cid_obj = Cid.__new__(Cid)
    base = {f'{t}s': {} for t in ACCEPTED_TYPES}
    base.update(resources or {})
    cid_obj.resources = base
    fixed = dict(KNOWN_PARAMS if params is None else params)
    cid_obj.get_template_parameters = lambda parameters, param_prefix='', others=None: dict(fixed)
    return cid_obj


# ---------------------------------------------------------------------------
# Feature: cid-cmd-agent-flow-platform, Property 5: Invalid definition types
# are rejected
# **Validates: Requirements 4.4**
# ---------------------------------------------------------------------------
_invalid_types = st.text(max_size=20).filter(lambda s: s not in ACCEPTED_TYPES)


@settings(max_examples=100, deadline=None)
@given(bad_type=_invalid_types)
def test_property_5_invalid_definition_types_are_rejected(bad_type):
    """Any type string outside the extended allowlist raises ValueError naming the type."""
    cid_obj = _make_cid()
    with pytest.raises(ValueError) as excinfo:
        cid_obj.get_definition(bad_type, name='anything')
    assert 'is not a valid definition type' in str(excinfo.value)


@pytest.mark.parametrize('accepted_type', ACCEPTED_TYPES)
def test_property_5_accepted_types_are_not_rejected(accepted_type):
    """All 7 allowlisted types pass the type check (unknown name may return None)."""
    cid_obj = _make_cid()
    result = cid_obj.get_definition(accepted_type, name='no-such-entry')
    assert result is None  # unknown name, but no unsupported-type ValueError


# ---------------------------------------------------------------------------
# Unit tests: agent/space substitution and unresolved-token errors
# **Validates: Requirements 4.2, 4.3**
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('definition_type,resource_key', [
    ('agent', 'agents'),
    ('space', 'spaces'),
])
def test_known_tokens_are_substituted(definition_type, resource_key):
    """: ${var} tokens with matching parameters are replaced in agent/space definitions."""
    cid_obj = _make_cid(resources={resource_key: {
        'my-entry': {
            'name': 'my-entry',
            'description': 'value is ${kvar} and ${kother}',
            'parameters': {},
        },
    }})
    res = cid_obj.get_definition(definition_type, name='my-entry')
    assert res is not None
    assert res['description'] == 'value is resolved-value and other-value'


@pytest.mark.parametrize('definition_type,resource_key', [
    ('agent', 'agents'),
    ('space', 'spaces'),
])
def test_unresolved_token_raises_valueerror_naming_token(definition_type, resource_key):
    """: an unresolved ${var} token in an agent/space definition raises ValueError naming it."""
    cid_obj = _make_cid(resources={resource_key: {
        'my-entry': {
            'name': 'my-entry',
            'description': 'known ${kvar} unknown ${missing_token}',
            'parameters': {},
        },
    }})
    with pytest.raises(ValueError) as excinfo:
        cid_obj.get_definition(definition_type, name='my-entry')
    assert 'missing_token' in str(excinfo.value)
