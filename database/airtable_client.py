"""
Airtable client for the Insights agent database operations.

Tables:
- Companies (tblk2Et7RYIVCWRzD): company profiles (employees, HG6M, description, confidential notes)
- Roles: open positions linked to companies
- Insights: user-contributed insights on companies/roles
- Users: community members who have engaged with companies/roles
"""

import os
from typing import Optional
from pyairtable import Api


# Base ID parsed from https://airtable.com/appo2zjaaetcT88Fx/...
AIRTABLE_BASE_ID = "appo2zjaaetcT88Fx"
# Companies table ID parsed from the URL path segment
AIRTABLE_COMPANIES_TABLE_ID = "tblk2Et7RYIVCWRzD"


class AirtableClient:
    def __init__(self):
        api_key = os.environ["AIRTABLE_API_KEY"]
        base_id = os.environ.get("AIRTABLE_BASE_ID", AIRTABLE_BASE_ID)

        self.api = Api(api_key)
        self.base = self.api.base(base_id)

        # Use the table ID directly for Companies – more stable than table name
        self.companies = self.base.table(
            os.environ.get("AIRTABLE_COMPANIES_TABLE", AIRTABLE_COMPANIES_TABLE_ID)
        )
        self.roles = self.base.table(os.environ.get("AIRTABLE_ROLES_TABLE", "Roles"))
        self.insights = self.base.table(os.environ.get("AIRTABLE_INSIGHTS_TABLE", "Insights"))
        self.users = self.base.table(os.environ.get("AIRTABLE_USERS_TABLE", "Users"))

    # -------------------------------------------------------------------------
    # Company lookups
    # -------------------------------------------------------------------------

    def find_company(self, company_name: str) -> Optional[dict]:
        """Search for a company by name (case-insensitive partial match)."""
        formula = f"SEARCH(LOWER('{company_name.lower()}'), LOWER({{Name}}))"
        records = self.companies.all(formula=formula)
        if records:
            return records[0]
        return None

    def get_company_insights(self, company_id: str) -> list[dict]:
        """Return all user-submitted insights for a company."""
        formula = f"FIND('{company_id}', ARRAYJOIN({{Company}}))"
        return self.insights.all(formula=formula)

    def get_company_roles(self, company_id: str) -> list[dict]:
        """Return all roles linked to a company."""
        formula = f"FIND('{company_id}', ARRAYJOIN({{Company}}))"
        return self.roles.all(formula=formula)

    def create_company(self, fields: dict) -> dict:
        """Create a new company record."""
        return self.companies.create(fields)

    def update_company(self, record_id: str, fields: dict) -> dict:
        """Update fields on an existing company record."""
        return self.companies.update(record_id, fields)

    # -------------------------------------------------------------------------
    # Role lookups
    # -------------------------------------------------------------------------

    def find_role(self, role_title: str, company_id: Optional[str] = None) -> Optional[dict]:
        """Search for a role by title, optionally scoped to a company."""
        title_filter = f"SEARCH(LOWER('{role_title.lower()}'), LOWER({{Title}}))"
        if company_id:
            formula = f"AND({title_filter}, FIND('{company_id}', ARRAYJOIN({{Company}})))"
        else:
            formula = title_filter
        records = self.roles.all(formula=formula)
        if records:
            return records[0]
        return None

    def find_role_by_id(self, record_id: str) -> Optional[dict]:
        return self.roles.get(record_id)

    def create_role(self, fields: dict) -> dict:
        """Create a new role record."""
        return self.roles.create(fields)

    def update_role(self, record_id: str, fields: dict) -> dict:
        """Update fields on an existing role record."""
        return self.roles.update(record_id, fields)

    # -------------------------------------------------------------------------
    # Insights (user-contributed notes)
    # -------------------------------------------------------------------------

    def create_insight(self, fields: dict) -> dict:
        """
        Create a new Insights record.

        Expected fields:
          - Company: [company_record_id]  (optional)
          - Role: [role_record_id]        (optional)
          - ContributedBy: slack_user_id
          - Content: free-text insight
          - Timestamp: ISO timestamp
        """
        return self.insights.create(fields)

    def get_insights_contributors(self, company_id: str, role_id: Optional[str] = None) -> list[dict]:
        """
        Return unique Slack user IDs who have contributed insights for a company/role.
        Used to suggest people to chat with.
        """
        if role_id:
            formula = (
                f"AND(FIND('{company_id}', ARRAYJOIN({{Company}})), "
                f"FIND('{role_id}', ARRAYJOIN({{Role}})))"
            )
        else:
            formula = f"FIND('{company_id}', ARRAYJOIN({{Company}}))"
        return self.insights.all(formula=formula)

    # -------------------------------------------------------------------------
    # Users
    # -------------------------------------------------------------------------

    def find_user(self, slack_user_id: str) -> Optional[dict]:
        formula = f"{{SlackID}} = '{slack_user_id}'"
        records = self.users.all(formula=formula)
        if records:
            return records[0]
        return None

    def upsert_user(self, slack_user_id: str, name: str) -> dict:
        """Create or update a user record."""
        existing = self.find_user(slack_user_id)
        if existing:
            return existing
        return self.users.create({"SlackID": slack_user_id, "Name": name})
