""" Agent helper: lifecycle operations for Quick Suite Chat Agents.

All Quick Suite generative-AI resources (Spaces, Agents) are served by the existing
``quicksight`` boto3 client (API ``quicksight-2018-04-01``); there is no separate
``quick``/``quicksuite`` client. This helper owns the Agent lifecycle
(describe / permissions / wait-active / delete / provenance) and mirrors the other
CidBase helpers (cur.py, glue.py, iam.py): one cohesive class over one shared client.
"""
import time
import logging

from cid.base import CidBase
from cid.exceptions import CidError
from cid.helpers.quicksight.agent_logic import (
    CID_PROVENANCE_TAG_KEY,
    CID_PROVENANCE_TAG_VALUE,
    CID_MANAGED_MARKER,
    PERSONA_API_FIELDS,
    persona_differs,
    compute_space_delta,
)

logger = logging.getLogger(__name__)

# The exact owner bundle for Agents (api-ref §12). Unlike Spaces, agent permissions
# must be granted as one recognized complete action set — piecemeal single-action
# grants fail with InvalidParameterValueException. Grant atomically.
AGENT_OWNER_ACTIONS = [
    'quicksight:DescribeAgent',
    'quicksight:UpdateAgent',
    'quicksight:DeleteAgent',
    'quicksight:DescribeAgentPermissions',
    'quicksight:UpdateAgentPermissions',
]  # 5 actions


class Agent(CidBase):
    """ Lifecycle operations for Quick Suite Chat Agents """

    def __init__(self, session, resources=None) -> None:
        super().__init__(session)
        logger.info('Creating QuickSight client for Agent helper')
        self.client = self.session.client('quicksight')
        self.resources = resources or {}

    def get(self, agent_id: str) -> dict:
        """ Read an Agent by id via DescribeAgent; return None when it does not exist.

        Existence/provenance checks MUST use describe_agent by id: ListAgents does NOT
        surface agents in PREVIEW or FAILED states (observed live) — ListAgents is
        enumeration-only.

        :param agent_id: the Agent id
        :returns: the ``Agent`` dict from the DescribeAgent response, or None if not found
        """
        try:
            response = self.client.describe_agent(
                AwsAccountId=self.account_id,
                AgentId=agent_id,
            )
            return response.get('Agent')
        except self.client.exceptions.ResourceNotFoundException:
            logger.debug(f'Agent {agent_id!r} not found.')
            return None

    def grant_owner(self, agent_id: str, principal_arn: str) -> None:
        """ Grant the principal the 5-action agent owner bundle as a SINGLE set.

        API-created Agents start with ``Permissions: []`` and the creating principal is
        not auto-granted owner rights. The grant must be the full recognized set in one
        UpdateAgentPermissions call — piecemeal grants fail (api-ref §12).

        :param agent_id: the Agent id
        :param principal_arn: QuickSight user/group ARN to grant owner permissions to
        """
        logger.info(f'Granting agent owner permissions on {agent_id!r} to {principal_arn}')
        self.client.update_agent_permissions(
            AwsAccountId=self.account_id,
            AgentId=agent_id,
            GrantPermissions=[{
                'Principal': principal_arn,
                'Actions': AGENT_OWNER_ACTIONS,
            }],
        )

    def wait_active(self, agent_id: str, timeout: int = 300, interval: int = 5) -> dict:
        """ Poll DescribeAgent until AgentStatus == ACTIVE.

        Polls every ``interval`` seconds up to ``timeout`` seconds. FAIL FAST: if
        ``AgentStatus`` becomes FAILED, stop polling immediately and raise including the
        agent id and the Agent's ``ErrorMessage`` field, without waiting for the timeout
        ( — observed live: a minimal agent went UPDATING -> FAILED).

        :param agent_id: the Agent id
        :param timeout: maximum seconds to wait (default 300)
        :param interval: seconds between polls (default 5)
        :returns: the Agent dict once ACTIVE
        :raises CidError: on FAILED status (immediately) or on timeout
        """
        elapsed = 0
        last_status = None
        while True:
            agent = self.get(agent_id)
            last_status = (agent or {}).get('AgentStatus')
            if last_status == 'ACTIVE':
                logger.info(f'Agent {agent_id!r} is ACTIVE.')
                return agent
            if last_status == 'FAILED':
                error_message = agent.get('ErrorMessage')
                raise CidError(
                    f'Agent {agent_id!r} entered FAILED status while waiting for ACTIVE. '
                    f'ErrorMessage: {error_message}'
                )
            if elapsed >= timeout:
                raise CidError(
                    f'Agent {agent_id!r} did not reach ACTIVE status within {timeout} seconds. '
                    f'Last observed status: {last_status}'
                )
            logger.debug(f'Agent {agent_id!r} status is {last_status}; waiting {interval}s ({elapsed}/{timeout}s elapsed).')
            time.sleep(interval)
            elapsed += interval

    def delete(self, agent_id: str, timeout: int = 300, interval: int = 5) -> None:
        """ Delete an Agent, tolerating a missing target.

        A ResourceNotFoundException is treated as success (the target is already gone).
        On ConflictException while the Agent is UPDATING/CREATING (observed live), wait
        for the status to settle and retry, up to ``timeout`` seconds total.

        :param agent_id: the Agent id
        :param timeout: maximum total seconds to retry through ConflictException (default 300)
        :param interval: seconds between retries (default 5)
        :raises CidError: when the conflict does not settle within the timeout
        """
        elapsed = 0
        while True:
            try:
                self.client.delete_agent(
                    AwsAccountId=self.account_id,
                    AgentId=agent_id,
                )
                logger.info(f'Deleted agent {agent_id!r}.')
                return
            except self.client.exceptions.ResourceNotFoundException:
                logger.info(f'Agent {agent_id!r} does not exist. Nothing to delete.')
                return
            except self.client.exceptions.ConflictException as exc:
                if elapsed >= timeout:
                    raise CidError(
                        f'Could not delete agent {agent_id!r}: the agent was still busy '
                        f'(ConflictException) after {timeout} seconds. Last error: {exc}'
                    ) from exc
                agent = self.get(agent_id)
                status = (agent or {}).get('AgentStatus')
                logger.debug(
                    f'Agent {agent_id!r} deletion hit ConflictException (status: {status}); '
                    f'retrying in {interval}s ({elapsed}/{timeout}s elapsed).'
                )
                time.sleep(interval)
                elapsed += interval

    def wait_settled(self, agent_id: str, timeout: int = 300, interval: int = 5) -> dict:
        """ Poll DescribeAgent until the agent leaves the transitional CREATING/UPDATING states.

        ROOT-CAUSE GUARD (observed live): CreateAgent with a PUBLISHED lifecycle starts an
        asynchronous publish workflow. Issuing a mutating call (UpdateAgent, TagResource)
        against the agent BEFORE that workflow settles corrupts the service-internal
        resource registry: a second overlapping publish leaves duplicate internal records
        for the same agent (TagResource then fails with ``InvalidParameterValueException:
        A resource with the same resourceName but a different internalId already exists:
        PUBLISHED-<guid>``), the Quick console shows the agent's linked Space as
        "resources unavailable", and chatting with the agent errors out — even though
        DescribeAgent output looks completely healthy. A console remove/re-add of the same
        Space repairs it because the console mutates a SETTLED agent. Every mutating call
        against a freshly created agent must therefore wait for the status to settle first.

        Unlike :meth:`wait_active` this never raises: provenance and tag writes are
        best-effort, so a timeout logs a warning and returns the last-seen agent.

        :param agent_id: the Agent id
        :param timeout: maximum seconds to wait (default 300)
        :param interval: seconds between polls (default 5)
        :returns: the last observed Agent dict (possibly still transitional on timeout),
            or None when the agent does not exist
        """
        elapsed = 0
        while True:
            agent = self.get(agent_id)
            status = (agent or {}).get('AgentStatus')
            if agent is None or status not in ('CREATING', 'UPDATING'):
                return agent
            if elapsed >= timeout:
                logger.warning(
                    f'Agent {agent_id!r} did not settle out of {status!r} within {timeout} seconds.'
                )
                return agent
            logger.debug(f'Agent {agent_id!r} status is {status}; waiting {interval}s to settle ({elapsed}/{timeout}s elapsed).')
            time.sleep(interval)
            elapsed += interval

    def _carry_through_update_params(self, agent: dict, agent_id: str) -> dict:
        """ Base UpdateAgent params carrying the DEPLOYED configuration through.

        UpdateAgent has FULL-REPLACEMENT semantics for the configuration fields
        (observed live): omitting ``StarterPrompts``, ``WelcomeMessage``,
        ``CustomPromptInput`` or ``Description`` on an UpdateAgent call CLEARS them on
        the agent. Every UpdateAgent call must therefore start from the deployed values
        and override only what it means to change.

        :param agent: the DescribeAgent read shape of the deployed agent
        :param agent_id: the Agent id
        :returns: UpdateAgent params pre-filled with the deployed configuration
        """
        params = {
            'AwsAccountId': self.account_id,
            'AgentId': agent_id,
            # UpdateAgent requires Name on every call. The read shape names it 'Name';
            # tolerate 'AgentName' (CreateAgent response key).
            'Name': agent.get('Name') or agent.get('AgentName'),
        }
        description = agent.get('Description')
        if description:
            params['Description'] = description
        starter_prompts = agent.get('StarterPrompts')
        if starter_prompts:
            params['StarterPrompts'] = list(starter_prompts)
        welcome_message = agent.get('WelcomeMessage')
        if welcome_message:
            params['WelcomeMessage'] = welcome_message
        # persona reads back as CustomPromptInterface; map to the NewPrompt write shape
        # (drops server-derived keys like promptSummary/ModelProfileId).
        deployed_prompt = self._persona_to_new_prompt(agent.get('CustomPromptInterface') or {})
        if deployed_prompt:
            params['CustomPromptInput'] = {'NewPrompt': deployed_prompt}
        return params

    def write_provenance(self, agent_arn: str, agent_id: str, timeout: int = 300, interval: int = 5) -> None:
        """ Dual CID_Managed provenance write for an Agent.

        (a) ensure the CID-managed marker is present in the agent Description via
            UpdateAgent. :meth:`_create` already bakes the marker into the CreateAgent
            Description, so for freshly created agents this is a read-only no-op; the
            UpdateAgent path only fires for pre-existing agents missing the marker.
            UpdateAgent has FULL-REPLACEMENT semantics (observed live: a
            Name+Description-only marker write WIPED the deployed persona, starter
            prompts and welcome message), so the whole deployed configuration is
            carried through on every attempt; and
        (b) TagResource on the agent ARN with the CID provenance tag — log-and-continue
            on failure, mirroring QuickSight.set_tags.

        Either mechanism alone suffices for is_cid_managed detection, which is why the
        tag write is tolerated-on-failure while the Description marker is the primary
        signal.

        MUTATION SAFETY (root cause of the live "resources unavailable"/broken-chat
        bug): a freshly created agent runs an asynchronous publish workflow
        (CREATING/UPDATING). Issuing UpdateAgent against it mid-publish is sometimes
        ACCEPTED and spawns a second overlapping publish, leaving duplicate internal
        ``PUBLISHED-<internalId>`` records that break the console's resource
        resolution and the chat runtime, while DescribeAgent still looks healthy.
        TagResource mid-publish is either silently dropped (202-and-ignored) or fails
        with an internalId-conflict error. This method therefore NEVER mutates a
        transitional agent: it waits for the status to settle (:meth:`wait_settled`)
        before the marker write, and settles again after a marker write (which itself
        triggers a republish) before the tag write. If the agent never settles, the
        marker write is skipped with a WARNING instead of risking corruption — the
        tag is still attempted and is_cid_managed accepts either mechanism.

        :param agent_arn: the Agent ARN (tag target)
        :param agent_id: the Agent id (describe/update target)
        :param timeout: maximum total seconds for the settle/retry budget (default 300)
        :param interval: seconds between retries (default 5)
        """
        # (a) CID-managed marker in the Description. The agent is re-described on
        # every attempt; a transitional (CREATING/UPDATING) agent is NEVER mutated —
        # the loop waits for it to settle within the shared elapsed budget.
        marker_written = False
        elapsed = 0
        while True:
            agent = self.get(agent_id)
            if agent is None:
                logger.warning(f'Cannot write the Description provenance marker: agent {agent_id!r} not found.')
                break
            status = agent.get('AgentStatus')
            if status in ('CREATING', 'UPDATING'):
                # NEVER mutate a mid-publish agent (see MUTATION SAFETY above). The
                # settle-wait also protects the tag write below: a fresh agent with
                # the marker already baked in by _create still needs to settle here.
                if elapsed >= timeout:
                    logger.warning(
                        f'Could not verify the provenance of agent {agent_id!r}: the agent was '
                        f'still {status} after {timeout} seconds and a mid-publish UpdateAgent '
                        f'corrupts the agent (observed live). The CID provenance tag write is '
                        f'still attempted, so CID-managed detection may be unaffected.'
                    )
                    break
                logger.debug(
                    f'Agent {agent_id!r} is {status}; waiting {interval}s before the provenance '
                    f'marker write ({elapsed}/{timeout}s elapsed).'
                )
                time.sleep(interval)
                elapsed += interval
                continue
            description = str(agent.get('Description') or '')
            if CID_MANAGED_MARKER in description:
                logger.debug(f'Agent {agent_id!r} Description already carries the CID-managed marker.')
                break
            # Full-replacement semantics: carry the whole deployed configuration
            # through and change ONLY the Description.
            params = self._carry_through_update_params(agent, agent_id)
            params['Description'] = f'{description} {CID_MANAGED_MARKER}'.strip()
            try:
                self.client.update_agent(**params)
                logger.debug(f'Wrote the CID-managed marker into the Description of agent {agent_id!r}.')
                marker_written = True
                break
            except self.client.exceptions.ConflictException as exc:
                if elapsed >= timeout:
                    # Do NOT crash the whole command for a provenance-marker write:
                    # the provenance tag (b) below is still attempted and
                    # is_cid_managed accepts either mechanism.
                    logger.warning(
                        f'Could not write the Description provenance marker on agent {agent_id!r}: '
                        f'the agent was still busy (ConflictException) after {timeout} seconds. '
                        f'The CID provenance tag is still attempted, so CID-managed detection '
                        f'may be unaffected. Last error: {exc}'
                    )
                    break
                logger.debug(
                    f'Agent {agent_id!r} provenance marker write hit ConflictException (status: {status}); '
                    f'retrying in {interval}s ({elapsed}/{timeout}s elapsed).'
                )
                time.sleep(interval)
                elapsed += interval
        if marker_written:
            # The marker UpdateAgent itself triggers a republish (UPDATING). Settle
            # again before tagging: TagResource against a transitional agent is
            # dropped or fails with an internalId conflict (observed live).
            self.wait_settled(agent_id, timeout=timeout, interval=interval)

        # (b) provenance tag: log-and-continue on failure. Only issued against a
        # settled agent (a TagResource call made while the agent is transitional is
        # accepted but dropped, or fails on duplicate internalIds — observed live).
        try:
            self.client.tag_resource(
                ResourceArn=agent_arn,
                Tags=[{'Key': CID_PROVENANCE_TAG_KEY, 'Value': CID_PROVENANCE_TAG_VALUE}],
            )
            logger.debug(f'Tagged agent {agent_id!r} with {CID_PROVENANCE_TAG_KEY}.')
        except self.client.exceptions.AccessDeniedException:
            logger.debug(f'Cannot tag {agent_arn} (AccessDenied). The Description marker is the fallback.')
        except self.client.exceptions.ClientError as exc:
            logger.debug(f'Cannot tag {agent_arn} ({exc}). The Description marker is the fallback.')

    @staticmethod
    def _persona_to_new_prompt(persona: dict) -> dict:
        """ Map a catalog persona dict to the CreateAgent/UpdateAgent write shape.

        The catalog persona.yaml uses camelCase keys (``identity``,
        ``customInstructions``, ``tone``, ``outputStyle``, ``responseLength``) while the
        API write shape ``CustomPromptInput.NewPrompt`` uses PascalCase (``Identity``,
        ``CustomInstructions``, ``Tone``, ``OutputStyle``, ``ResponseLength``); the two
        differ only by letter case, so keys are matched case-insensitively. Any other
        key (notably a server-derived ``promptSummary``) is dropped.

        :param persona: persona dict from the catalog definition (camelCase or PascalCase)
        :returns: the ``NewPrompt`` dict (``CustomPromptInputParameters``) with only the
            fields present in the input
        """
        lowered = {str(key).lower(): value for key, value in (persona or {}).items() if value is not None}
        return {
            field: str(lowered[field.lower()])
            for field in PERSONA_API_FIELDS
            if field.lower() in lowered
        }

    def create_or_update(self, definition: dict, space_arns) -> dict:
        """ Idempotent Agent create-or-update: create, drift-diffed update, or no-op.

        Create path (the agent does not exist per :meth:`get`): one ``CreateAgent`` call
        carrying all 5 persona fields (write shape ``CustomPromptInput.NewPrompt``), the
        starter prompts, the welcome message, the attached Space ARNs, the action
        connectors, and the lifecycle. A ``ResourceExistsException`` is
        treated as success.

        Update path (the agent exists): drift is diffed field-by-field against the
        DescribeAgent read shape — persona via :func:`persona_differs` over the 5 fields
        excluding the server-derived ``promptSummary`` (, the persona reads back
        as ``Agent.CustomPromptInterface``); Space/connector associations via
        :func:`compute_space_delta` sent as delta add/remove lists (``SpacesToAdd`` /
        ``SpacesToRemove``, ``ActionConnectorsToAdd`` / ``ActionConnectorsToRemove``,
); persona / ``StarterPrompts`` / ``WelcomeMessage`` are full-replacement
        values; and ``Name`` is included on EVERY ``UpdateAgent`` call because
        the API requires it. ``UpdateAgent`` has FULL-REPLACEMENT semantics
        (omitted configuration fields are cleared — observed live), so every update
        starts from the deployed configuration (including the ``Description``, which
        carries the CID-managed provenance marker written by :meth:`write_provenance`)
        and overrides only the managed fields. ``lifecycle`` is create-only:
        ``UpdateAgent`` does not accept it, so lifecycle drift is warned about and left
        unchanged. When nothing differs, no mutating API call is made and the agent is
        reported up to date.

        :param definition: agent definition from the catalog: keys ``name``, ``agentId``,
            ``description``, ``persona`` (5 camelCase fields), ``starterPrompts``,
            ``welcomeMessage``, ``lifecycle``, and optional
            ``dependsOn.actionConnectors``
        :param space_arns: iterable of Space ARNs to attach as knowledge sources
        :returns: dict with keys ``agentId``, ``arn`` and ``action`` — one of
            ``'created'`` / ``'updated'`` / ``'unchanged'``
        """
        agent_id = definition.get('agentId')
        name = definition.get('name')
        desired_persona = definition.get('persona') or {}
        starter_prompts = definition.get('starterPrompts')  # None = not managed by this definition
        welcome_message = definition.get('welcomeMessage')  # None = not managed by this definition
        lifecycle = definition.get('lifecycle')             # None = not managed by this definition
        desired_spaces = set(space_arns or ())
        desired_connectors = set((definition.get('dependsOn') or {}).get('actionConnectors') or ())

        agent = self.get(agent_id)
        if agent is None:
            return self._create(
                agent_id=agent_id,
                name=name,
                description=definition.get('description'),
                new_prompt=self._persona_to_new_prompt(desired_persona),
                starter_prompts=starter_prompts,
                welcome_message=welcome_message,
                lifecycle=lifecycle,
                space_arns=desired_spaces,
                connector_arns=desired_connectors,
            )
        return self._update(
            agent=agent,
            agent_id=agent_id,
            name=name,
            desired_persona=desired_persona,
            starter_prompts=starter_prompts,
            welcome_message=welcome_message,
            lifecycle=lifecycle,
            desired_spaces=desired_spaces,
            desired_connectors=desired_connectors,
        )

    def _create(self, agent_id, name, description, new_prompt, starter_prompts,
                welcome_message, lifecycle, space_arns, connector_arns) -> dict:
        """ CreateAgent carrying the full definition, then Space attach via UpdateAgent.

        The CID-managed provenance marker is baked into the Description AT CREATE TIME
: a post-create UpdateAgent marker write against the still-publishing
        agent spawns an overlapping publish workflow that corrupts the service-internal
        resource registry (duplicate ``PUBLISHED-<internalId>`` records — observed live
        as the console showing the linked Space "resources unavailable" and chat
        erroring out, while DescribeAgent looks healthy). Baking the marker in makes
        the post-create :meth:`write_provenance` marker step a read-only no-op.

        SPACE ATTACH SEQUENCING (observed live): passing ``Spaces`` inside the
        CreateAgent call produces a broken Space association — the console shows the
        linked Space as "resources unavailable" and chatting with the agent fails with
        an internal server error, even after the create flow issues zero post-create
        mutations. The ONLY known-good repair is the console's remove/re-add of the
        same Space, which is an UpdateAgent ``SpacesToRemove``/``SpacesToAdd`` pair
        against a SETTLED, ACTIVE agent. The create path therefore mirrors the
        known-good half of that repair: CreateAgent WITHOUT ``Spaces``, wait for the
        publish workflow to settle (:meth:`wait_settled`), then attach the Spaces via
        UpdateAgent ``SpacesToAdd`` carrying the full deployed configuration through
        (full-replacement semantics; ``Name`` required; ``AgentLifecycle`` is
        create-only and never sent on update).
        """
        params = {
            'AwsAccountId': self.account_id,
            'AgentId': agent_id,
            'Name': name,
        }
        description = str(description or '')
        if CID_MANAGED_MARKER not in description:
            description = f'{description} {CID_MANAGED_MARKER}'.strip()
        params['Description'] = description
        if new_prompt:
            params['CustomPromptInput'] = {'NewPrompt': new_prompt}
        if starter_prompts is not None:
            params['StarterPrompts'] = list(starter_prompts)
        if welcome_message is not None:
            params['WelcomeMessage'] = welcome_message
        if lifecycle is not None:
            params['AgentLifecycle'] = lifecycle
        # Spaces are NOT passed at create time (see SPACE ATTACH SEQUENCING above);
        # they are attached below via UpdateAgent SpacesToAdd against the settled agent.
        if connector_arns:
            params['ActionConnectors'] = sorted(connector_arns)
        logger.info(f'Creating agent {agent_id!r}.')
        try:
            response = self.client.create_agent(**params)
            arn = response.get('Arn')
        except self.client.exceptions.ResourceExistsException:
            # Another writer created the agent between our describe and create — the
            # agent exists, which is what we wanted: treat as success.
            logger.info(f'Agent {agent_id!r} already exists (ResourceExistsException). Treating as success.')
            arn = (self.get(agent_id) or {}).get('Arn')
        if space_arns:
            self._attach_spaces_after_create(agent_id, space_arns)
        return {'agentId': agent_id, 'arn': arn, 'action': 'created'}

    def _attach_spaces_after_create(self, agent_id, space_arns, timeout: int = 300, interval: int = 5) -> None:
        """ Attach Spaces to a freshly created agent via UpdateAgent ``SpacesToAdd``.

        Mirrors the known-good console repair path: waits for the agent's initial
        publish workflow to settle (:meth:`wait_settled` — never mutate a
        CREATING/UPDATING agent), then issues one UpdateAgent with ``SpacesToAdd``
        carrying the full deployed configuration through (full-replacement semantics;
        ``Name`` required). Skips ARNs the agent already carries. A residual
        ConflictException retries within the shared budget ( pattern).
        """
        agent = self.wait_settled(agent_id, timeout=timeout, interval=interval)
        if agent is None:
            raise CidError(
                f'Cannot attach Spaces to agent {agent_id!r}: the agent was not found after create.'
            )
        to_add = sorted(set(space_arns) - set(agent.get('Spaces') or ()))
        if not to_add:
            logger.debug(f'Agent {agent_id!r} already carries all desired Spaces.')
            return
        elapsed = 0
        while True:
            params = self._carry_through_update_params(agent, agent_id)
            params['SpacesToAdd'] = to_add
            logger.info(f'Attaching {len(to_add)} Space(s) to agent {agent_id!r} via UpdateAgent.')
            try:
                self.client.update_agent(**params)
                return
            except self.client.exceptions.ConflictException as exc:
                if elapsed >= timeout:
                    raise CidError(
                        f'Could not attach Spaces to agent {agent_id!r}: the agent was still busy '
                        f'(ConflictException) after {timeout} seconds. Last error: {exc}'
                    ) from exc
                logger.debug(
                    f'Space attach on agent {agent_id!r} hit ConflictException; '
                    f'retrying in {interval}s ({elapsed}/{timeout}s elapsed).'
                )
                time.sleep(interval)
                elapsed += interval
                agent = self.wait_settled(agent_id, timeout=max(timeout - elapsed, interval), interval=interval) or agent

    def _update(self, agent, agent_id, name, desired_persona, starter_prompts,
                welcome_message, lifecycle, desired_spaces, desired_connectors) -> dict:
        """ Drift-diffed UpdateAgent, or no-op when nothing changed. """
        arn = agent.get('Arn')
        # Read shapes: persona reads back as CustomPromptInterface (write/read asymmetry);
        # Spaces/ActionConnectors read back as ARN-string lists; the read shape names the
        # agent name 'Name' (tolerate 'AgentName', the CreateAgent response key).
        deployed_interface = agent.get('CustomPromptInterface') or {}
        current_spaces = set(agent.get('Spaces') or ())
        current_connectors = set(agent.get('ActionConnectors') or ())
        deployed_name = agent.get('Name') or agent.get('AgentName')

        spaces_to_add, spaces_to_remove = compute_space_delta(desired_spaces, current_spaces)
        connectors_to_add, connectors_to_remove = compute_space_delta(desired_connectors, current_connectors)

        # Fields the definition does not carry (None) are not managed and never count as drift.
        # 'lifecycle' is create-only: UpdateAgent does not accept an AgentLifecycle
        # parameter, so lifecycle drift cannot be remediated here — warn instead.
        if lifecycle is not None and lifecycle != agent.get('AgentLifecycle'):
            logger.warning(
                f'Agent {agent_id!r} lifecycle is {agent.get("AgentLifecycle")!r} but the '
                f'catalog wants {lifecycle!r}. The lifecycle can only be set at creation '
                f'(UpdateAgent does not accept it) and is left unchanged.'
            )
        drift = {
            'persona': bool(desired_persona) and persona_differs(desired_persona, deployed_interface),
            'spaces': bool(spaces_to_add or spaces_to_remove),
            'connectors': bool(connectors_to_add or connectors_to_remove),
            'starter_prompts': starter_prompts is not None and list(starter_prompts) != list(agent.get('StarterPrompts') or ()),
            'welcome_message': welcome_message is not None and str(welcome_message) != str(agent.get('WelcomeMessage') or ''),
            'name': name is not None and name != deployed_name,
        }
        if not any(drift.values()):
            # Desired configuration matches the deployed one: no API call.
            logger.info(f'Agent {agent_id!r} is up to date. No change needed.')
            return {'agentId': agent_id, 'arn': arn, 'action': 'unchanged'}

        drifted = ', '.join(sorted(field for field, differs in drift.items() if differs))
        logger.info(f'Agent {agent_id!r} configuration drifted ({drifted}). Updating.')
        # UpdateAgent has FULL-REPLACEMENT semantics (observed live): omitted
        # configuration fields are CLEARED. Start from the deployed configuration
        # (including the Description, which carries the CID provenance marker) and
        # override only the managed fields.
        params = self._carry_through_update_params(agent, agent_id)
        if name is not None:
            # UpdateAgent requires Name on EVERY call.
            params['Name'] = name
        # Full-replacement fields: send the complete desired value.
        new_prompt = self._persona_to_new_prompt(desired_persona)
        if new_prompt:
            params['CustomPromptInput'] = {'NewPrompt': new_prompt}
        if starter_prompts is not None:
            params['StarterPrompts'] = list(starter_prompts)
        if welcome_message is not None:
            params['WelcomeMessage'] = welcome_message
        # Delta-based association fields: add/remove lists, never a full replace.
        if spaces_to_add:
            params['SpacesToAdd'] = sorted(spaces_to_add)
        if spaces_to_remove:
            params['SpacesToRemove'] = sorted(spaces_to_remove)
        if connectors_to_add:
            params['ActionConnectorsToAdd'] = sorted(connectors_to_add)
        if connectors_to_remove:
            params['ActionConnectorsToRemove'] = sorted(connectors_to_remove)
        self.client.update_agent(**params)
        return {'agentId': agent_id, 'arn': arn, 'action': 'updated'}
