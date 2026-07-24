"""Launch-library content tests.

Launch agents and spaces are catalog content at the repo root (agents/,
spaces/), listed in dashboards/catalog.yaml and loaded through the same
pipeline as dashboards. Content merges via git PRs (no PyPI release), so these
tests are the referential-completeness gate for contributions: every
dependency key an agent declares must resolve in the merged catalog, and every
persona must be complete.

The builtin plugin still provides the core dashboard / dataset / view catalog
the agents reference, so both sources are loaded ONCE at module scope and
merged — mirroring what Cid.load_resources does at runtime.
"""
import os
from pathlib import Path

import pytest
import yaml

from cid.common import Cid
from cid.plugin import Plugin

REPO_ROOT = Path(__file__).resolve().parents[3]
CATALOG_PATH = REPO_ROOT / 'dashboards' / 'catalog.yaml'

LAUNCH_AGENT_IDS = ('operations', 'finops', 'security')
PERSONA_FIELDS = ('identity', 'customInstructions', 'tone', 'outputStyle', 'responseLength')


def _load_runtime_resources():
    """Builtin plugin resources merged with the local repo catalog (once)."""
    resources = Plugin('cid.builtin.core').provides()
    cid_obj = Cid.__new__(Cid)
    cid_obj.resources = resources
    cid_obj.load_catalog(str(CATALOG_PATH))
    return cid_obj.resources


# Module-scope load: content is fixed shipped data.
RESOURCES = _load_runtime_resources()


@pytest.mark.parametrize('agent_id', LAUNCH_AGENT_IDS)
def test_launch_agent_is_referentially_complete(agent_id):
    """Every dependsOn.dashboards / optionalDashboards key resolves to a
    dashboard catalog entry, every dependsOn.datasets key resolves to a
    datasets catalog entry, and every dependsOn.spaces id resolves to the
    spaces catalog."""
    agent = RESOURCES['agents'][agent_id]
    depends_on = agent.get('dependsOn', {})

    dashboards_catalog = RESOURCES.get('dashboards', {})
    datasets_catalog = RESOURCES.get('datasets', {})
    spaces_catalog = RESOURCES.get('spaces', {})

    for key in list(depends_on.get('dashboards', [])) + list(depends_on.get('optionalDashboards', [])):
        assert key in dashboards_catalog, (
            f'Agent {agent_id!r} references dashboard key {key!r} '
            f'that is not in the dashboards catalog')

    for key in depends_on.get('datasets', []):
        assert key in datasets_catalog, (
            f'Agent {agent_id!r} references dataset key {key!r} '
            f'that is not in the datasets catalog')

    for space_id in depends_on.get('spaces', []):
        assert space_id in spaces_catalog, (
            f'Agent {agent_id!r} references space id {space_id!r} '
            f'that is not in the spaces catalog')


@pytest.mark.parametrize('agent_id', LAUNCH_AGENT_IDS)
def test_launch_agent_persona_is_complete(agent_id):
    """The personaFile resolves against the manifest's source folder and
    carries all five persona fields non-empty."""
    agent = RESOURCES['agents'][agent_id]
    persona_file = agent.get('personaFile')
    source = agent.get('source')
    assert persona_file and source, f'Agent {agent_id!r} has no personaFile reference'
    persona_path = os.path.join(os.path.dirname(source), persona_file)
    assert os.path.isfile(persona_path), (
        f'Agent {agent_id!r} personaFile does not resolve to a file: {persona_path}')
    with open(persona_path, encoding='utf-8') as handle:
        persona = yaml.safe_load(handle)
    assert isinstance(persona, dict), f'Agent {agent_id!r} persona is not a YAML mapping'
    for field in PERSONA_FIELDS:
        value = persona.get(field)
        assert isinstance(value, str) and value.strip(), (
            f'Agent {agent_id!r} persona field {field!r} is missing or empty')


def test_launch_agents_and_shared_space_register_via_catalog():
    """The three launch agents and the shared space register through the
    catalog pipeline."""
    for agent_id in LAUNCH_AGENT_IDS:
        assert agent_id in RESOURCES.get('agents', {}), f'Launch agent {agent_id!r} did not register'
    assert 'cid-dashboards-space' in RESOURCES.get('spaces', {})


def test_builtin_plugin_ships_no_agents_or_spaces():
    """The builtin plugin data carries no agents or spaces kinds — they are
    catalog content, not package content."""
    builtin = Plugin('cid.builtin.core').provides()
    assert 'agents' not in builtin
    assert 'spaces' not in builtin
