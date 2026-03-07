"""
Airtable client for the Insights agent database operations.

READ-ONLY during testing. Write methods are intentionally absent.

Tables:
- Companies (tblk2Et7RYIVCWRzD): company profiles
- Roles: open positions linked to companies
"""

import logging
import os
from typing import Optional
from pyairtable import Api

logger = logging.getLogger(__name__)

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

        self.companies = self.base.table(
            os.environ.get("AIRTABLE_COMPANIES_TABLE", AIRTABLE_COMPANIES_TABLE_ID)
        )
        self.roles = self.base.table(os.environ.get("AIRTABLE_ROLES_TABLE", "Roles"))

    # -------------------------------------------------------------------------
    # Company lookups
    # -------------------------------------------------------------------------

    def find_company(self, company_name: str) -> Optional[dict]:
        """Search for a company by name (case-insensitive partial match)."""
        try:
            formula = f"SEARCH(LOWER('{company_name.lower()}'), LOWER({{Name}}))"
            records = self.companies.all(formula=formula)
            return records[0] if records else None
        except Exception:
            logger.exception("find_company failed for %r", company_name)
            return None

    def get_company(self, record_id: str) -> Optional[dict]:
        try:
            return self.companies.get(record_id)
        except Exception:
            logger.exception("get_company failed for %r", record_id)
            return None

    def get_company_roles(self, company_id: str) -> list[dict]:
        """Return all roles linked to a company."""
        try:
            formula = f"FIND('{company_id}', ARRAYJOIN({{Company}}))"
            return self.roles.all(formula=formula)
        except Exception:
            logger.exception("get_company_roles failed for company %r", company_id)
            return []

    # -------------------------------------------------------------------------
    # Role lookups
    # -------------------------------------------------------------------------

    def find_role(self, role_title: str, company_id: Optional[str] = None) -> Optional[dict]:
        """Search for a role by title, optionally scoped to a company."""
        try:
            title_filter = f"SEARCH(LOWER('{role_title.lower()}'), LOWER({{Title}}))"
            if company_id:
                formula = f"AND({title_filter}, FIND('{company_id}', ARRAYJOIN({{Company}})))"
            else:
                formula = title_filter
            records = self.roles.all(formula=formula)
            return records[0] if records else None
        except Exception:
            logger.exception("find_role failed for %r", role_title)
            return None

    def find_role_for_company(self, role_title: str, company_name: str) -> tuple[Optional[dict], Optional[dict]]:
        """
        Find a role by title scoped to a company.

        Uses ARRAYJOIN({Company}) inside the Airtable formula, which resolves
        linked records to their primary-field display values (i.e. company names)
        server-side — so matching is exact and no separate get_company loop is needed.
        """
        try:
            title_q = role_title.lower().replace("'", "\\'")
            company_q = company_name.lower().replace("'", "\\'")
            formula = (
                f"AND("
                f"SEARCH(LOWER('{title_q}'), LOWER({{Title}})), "
                f"SEARCH(LOWER('{company_q}'), LOWER(ARRAYJOIN({{Company}})))"
                f")"
            )
            records = self.roles.all(formula=formula)
            logger.info(
                "find_role_for_company: %d result(s) for title=%r company=%r",
                len(records), role_title, company_name,
            )
            if not records:
                return None, None

            role = records[0]
            # Resolve the company record for richer synopsis context.
            # Try direct ID lookup first; fall back to formula search by name.
            linked_ids = role["fields"].get("Company", [])
            co = self.get_company(linked_ids[0]) if linked_ids else None
            if co is None:
                co = self.find_company(company_name)
            return role, co
        except Exception:
            logger.exception("find_role_for_company failed for %r / %r", role_title, company_name)
            return None, None

    def find_role_by_id(self, record_id: str) -> Optional[dict]:
        try:
            return self.roles.get(record_id)
        except Exception:
            logger.exception("find_role_by_id failed for %r", record_id)
            return None
