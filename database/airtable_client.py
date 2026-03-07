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
        Search all roles matching role_title, then iterate linked company records
        to find one whose name matches company_name.  Returns (role, company).
        More reliable than a single-record lookup when the Companies table search fails.
        """
        try:
            formula = f"SEARCH(LOWER('{role_title.lower()}'), LOWER({{Title}}))"
            candidates = self.roles.all(formula=formula)
            for role in candidates:
                linked_ids = role["fields"].get("Company", [])
                for co_id in linked_ids:
                    co = self.get_company(co_id)
                    if co:
                        co_name = co["fields"].get("Name", "")
                        if (company_name.lower() in co_name.lower()
                                or co_name.lower() in company_name.lower()):
                            return role, co
            return None, None
        except Exception:
            logger.exception("find_role_for_company failed for %r / %r", role_title, company_name)
            return None, None

    def find_role_by_id(self, record_id: str) -> Optional[dict]:
        try:
            return self.roles.get(record_id)
        except Exception:
            logger.exception("find_role_by_id failed for %r", record_id)
            return None
