"""Cost guards shared by the hand-run scripts.

Those scripts run with the operator's credentials, which the Budgets hard-stop doesn't
touch (its deny policy only attaches to the project roles). So each one checks for the
policy itself and refuses to start anything once the cap has been hit.
"""

from __future__ import annotations


def budget_hardstop_active(iam, role_arn: str, project: str) -> bool:
    """True if the Budgets action has attached its deny policy to the project role."""
    role = role_arn.rsplit("/", 1)[-1]
    attached = iam.list_attached_role_policies(RoleName=role)["AttachedPolicies"]
    return any(p["PolicyName"] == f"{project}-budget-hardstop-deny" for p in attached)
