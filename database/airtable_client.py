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
        """Search for a company by name (exact match preferred, partial fallback).

        Each strategy has its own try/except so a failure in the exact formula
        (e.g. Airtable 422) still allows the partial fallback to run.
        """
        name_q = company_name.lower().replace("'", "\\'")

        # 1. Exact case-insensitive match — avoids returning "Airtable Enterprise"
        #    when the user asks for "Airtable".
        try:
            records = self.companies.all(formula=f"LOWER({{Name}}) = LOWER('{name_q}')")
            if records:
                logger.info("find_company exact match for %r: %r", company_name, records[0]["fields"].get("Name"))
                return records[0]
        except Exception:
            logger.warning("find_company exact formula failed for %r, trying partial", company_name)

        # 2. Partial / substring match fallback.
        try:
            records = self.companies.all(formula=f"SEARCH(LOWER('{name_q}'), LOWER({{Name}}))")
            if records:
                logger.info("find_company partial match for %r: %r", company_name, records[0]["fields"].get("Name"))
                return records[0]
        except Exception:
            logger.exception("find_company partial search failed for %r", company_name)

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

        Two-step: resolve the company record by name first (exact match preferred),
        then search roles using the company's record ID in the linked Company field.
        Record-ID matching via FIND/ARRAYJOIN is reliable in Airtable filter formulas;
        display-name matching via ARRAYJOIN is not (filterByFormula returns IDs, not names).
        """
        try:
            co = self.find_company(company_name)
            if not co:
                logger.info("find_role_for_company: company %r not found", company_name)
                return None, None

            role = self.find_role(role_title, co["id"])
            logger.info(
                "find_role_for_company: role=%r for company=%r (%r) -> %s",
                role_title, company_name, co["fields"].get("Name"),
                "found" if role else "not found",
            )
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
