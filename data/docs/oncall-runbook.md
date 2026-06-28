# Engineering On-Call Runbook

*Owner: Engineering (Marcus Hale) — last updated 15 January 2026*

How on-call works for the engineering team supporting Beacon and the Drift fleet.

## The rotation

On-call runs on a **weekly rotation**, handed over every **Monday at 10:00**. Each week there is a **primary** and a **secondary** on-call engineer. On-call engineers receive an **on-call allowance of £200 per week**.

Alerts are delivered through **AlertHub**, which pages the primary first and the secondary if the primary doesn't acknowledge.

## Severity levels

| Severity | Definition | Acknowledge within |
| --- | --- | --- |
| **SEV1** | Customer-facing outage or any safety risk | 15 minutes |
| **SEV2** | Major degradation, no full outage | 30 minutes |
| **SEV3** | Minor or cosmetic issue | Next business day |

For a **SEV1**, also post in Slack `#incidents` and start an incident channel.

## Escalation path

If you cannot resolve or acknowledge an incident, escalate in order:

1. **Primary** on-call engineer
2. **Secondary** on-call engineer
3. **Marcus Hale**, Engineering Manager
4. **Tom Whitlock**, CTO

## After an incident

For every **SEV1 and SEV2**, run a **blameless post-incident review within 3 business days**. Capture a timeline, root cause, and action items, and track the actions in Jira. The goal is to learn, never to assign blame.
