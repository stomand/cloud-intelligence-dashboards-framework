"""Tests for loading agents and spaces through the dashboards catalog pipeline.

Agents and spaces are catalog content, defined exactly like dashboards: a
kind-wrapped resource file (top-level ``agents:`` / ``spaces:`` map) listed in
``dashboards/catalog.yaml`` and merged by ``Cid.load_resource_file`` — no
agent-specific loader code. Sibling references (``personaFile``) stay relative
in the manifest and are resolved lazily against the entry's ``source``.

These tests cover the one mechanism the agents feature relies on beyond what
dashboards already exercise (relative persona resolution), the loader's
skip-and-continue error handling, and the shipped catalog wiring.
"""
import tempfile
from pathlib import Path

import pytest
import yaml

from cid.common import Cid
from cid.exceptions import CidCritical, CidError

REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Helpers: build a temp repo-like layout and load it through the catalog
# ---------------------------------------------------------------------------

def _make_cid():
    """Build a Cid with only the attributes the resource loaders touch."""
    cid_obj = Cid.__new__(Cid)
    cid_obj.resources = {}
    return cid_obj


def _write_repo_layout(root, agents=None, spaces=None):
    """Create <root>/dashboards/catalog.yaml + agents/<id>/<id>.yaml + spaces/<id>.yaml.

    Mirrors the real repo convention: catalog urls are relative to the catalog
    file, agent manifests are kind-wrapped and live in per-agent folders.
    """
    root = Path(root)
    (root / 'dashboards').mkdir(parents=True, exist_ok=True)
    urls = []
    for agent_id, entry in (agents or {}).items():
        agent_dir = root / 'agents' / agent_id
        agent_dir.mkdir(parents=True)
        (agent_dir / f'{agent_id}.yaml').write_text(yaml.safe_dump({'agents': {agent_id: entry}}))
        urls.append(f'../agents/{agent_id}/{agent_id}.yaml')
    for space_id, entry in (spaces or {}).items():
        spaces_dir = root / 'spaces'
        spaces_dir.mkdir(parents=True, exist_ok=True)
        (spaces_dir / f'{space_id}.yaml').write_text(yaml.safe_dump({'spaces': {space_id: entry}}))
        urls.append(f'../spaces/{space_id}.yaml')
    catalog = root / 'dashboards' / 'catalog.yaml'
    catalog.write_text(yaml.safe_dump({'Resources': [{'Url': url} for url in urls]}))
    return catalog


# ---------------------------------------------------------------------------
# Catalog loading: registration, source provenance, persona resolution
# ---------------------------------------------------------------------------

def test_catalog_registers_agents_and_spaces_with_source():
    """Kind-wrapped agent and space files listed in the catalog merge into
    resources['agents']/['spaces'] with content preserved and source metadata
    pointing at the originating file — through the same pipeline dashboards use."""
    agents = {'my-agent': {'name': 'my agent', 'category': 'FinOps'}}
    spaces = {'my-space': {'name': 'my space'}}
    with tempfile.TemporaryDirectory(prefix='cid_catalog_test_') as tmpdir:
        catalog = _write_repo_layout(tmpdir, agents=agents, spaces=spaces)
        cid_obj = _make_cid()
        cid_obj.load_catalog(str(catalog))

        loaded_agent = cid_obj.resources['agents']['my-agent']
        loaded_space = cid_obj.resources['spaces']['my-space']
        assert loaded_agent['name'] == 'my agent'
        assert loaded_agent['category'] == 'FinOps'
        assert loaded_space['name'] == 'my space'
        root = Path(tmpdir)
        assert Path(loaded_agent['source']).resolve() == (root / 'agents' / 'my-agent' / 'my-agent.yaml').resolve()
        assert Path(loaded_space['source']).resolve() == (root / 'spaces' / 'my-space.yaml').resolve()


def test_persona_file_resolves_against_manifest_folder():
    """A relative personaFile resolves against the manifest's source folder and
    returns the parsed persona mapping."""
    persona = {'identity': 'intended sibling content'}
    with tempfile.TemporaryDirectory(prefix='cid_catalog_test_') as tmpdir:
        catalog = _write_repo_layout(
            tmpdir, agents={'my-agent': {'name': 'agent', 'personaFile': 'persona.yaml'}})
        (Path(tmpdir) / 'agents' / 'my-agent' / 'persona.yaml').write_text(yaml.safe_dump(persona))

        cid_obj = _make_cid()
        cid_obj.load_catalog(str(catalog))
        definition = cid_obj.resources['agents']['my-agent']

        assert cid_obj._load_agent_persona(definition) == persona


def test_inline_persona_wins_over_persona_file():
    """An inline persona mapping is used as-is without touching personaFile."""
    cid_obj = _make_cid()
    persona = {'identity': 'inline'}
    assert cid_obj._load_agent_persona({'persona': persona, 'personaFile': 'x.yaml'}) == persona


# ---------------------------------------------------------------------------
# Loader error handling
# ---------------------------------------------------------------------------

def test_malformed_resource_file_is_skipped_and_loading_continues():
    """A malformed resource file is skipped with a warning; the remaining
    catalog entries still load (same behavior dashboards rely on)."""
    with tempfile.TemporaryDirectory(prefix='cid_catalog_test_') as tmpdir:
        catalog = _write_repo_layout(tmpdir, agents={'good': {'name': 'good agent'}})
        bad = Path(tmpdir) / 'agents' / 'bad'
        bad.mkdir(parents=True)
        (bad / 'bad.yaml').write_text('agents: [unclosed\n  sequence: {')
        catalog_content = yaml.safe_load(catalog.read_text())
        catalog_content['Resources'].insert(0, {'Url': '../agents/bad/bad.yaml'})
        catalog.write_text(yaml.safe_dump(catalog_content))

        cid_obj = _make_cid()
        cid_obj.load_catalog(str(catalog))

        assert 'bad' not in cid_obj.resources.get('agents', {})
        assert 'good' in cid_obj.resources['agents']


def test_missing_persona_file_raises_naming_the_file():
    """A personaFile that does not resolve raises a CID error naming the file.

    A missing local file surfaces as CidCritical from resolve_relative_path
    (the tool's fatal-error pattern); a present-but-unreadable file surfaces
    as CidError from the persona wrapper. Both name the reference.
    """
    with tempfile.TemporaryDirectory(prefix='cid_catalog_test_') as tmpdir:
        catalog = _write_repo_layout(
            tmpdir, agents={'my-agent': {'name': 'agent', 'personaFile': 'missing.yaml'}})
        cid_obj = _make_cid()
        cid_obj.load_catalog(str(catalog))
        definition = cid_obj.resources['agents']['my-agent']

        with pytest.raises((CidError, CidCritical)) as excinfo:
            cid_obj._load_agent_persona(definition)
        assert 'missing.yaml' in str(excinfo.value)


# ---------------------------------------------------------------------------
# Repo catalog wiring: the shipped catalog lists the launch agents and space
# ---------------------------------------------------------------------------

def test_repo_catalog_lists_launch_agents_and_shared_space():
    """dashboards/catalog.yaml references the repo-root agent and space files."""
    catalog = yaml.safe_load((REPO_ROOT / 'dashboards' / 'catalog.yaml').read_text())
    urls = [resource.get('Url') for resource in catalog.get('Resources', [])]
    for expected in ('../agents/finops/finops.yaml',
                     '../agents/operations/operations.yaml',
                     '../agents/security/security.yaml',
                     '../spaces/cid-dashboards-space.yaml'):
        assert expected in urls, f'{expected} is not listed in dashboards/catalog.yaml'
    for url in urls:
        referenced = (REPO_ROOT / 'dashboards' / url).resolve()
        assert referenced.is_file(), f'catalog url {url} does not resolve to a file'
