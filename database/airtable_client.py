"""
Airtable client for the Insights agent database operations.

READ-ONLY during testing. Write methods are intentionally absent.

Tables:
- Companies (tblk2Et7RYIVCWRzD): company profiles
- Roles: open positions linked to companies
"""

import logging
import os
from difflib import get_close_matches
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

        roles_table_name = os.environ.get("AIRTABLE_ROLES_TABLE", "Roles")
        self.roles = self.base.table(roles_table_name)

        # Resolve the Companies table that the Roles table actually links to.
        # The hardcoded AIRTABLE_COMPANIES_TABLE_ID may differ from the real
        # linked table, causing get_company() to return 404 for all role-linked IDs.
        companies_table_id = self._discover_companies_table(
            company_field_name=os.environ.get("AIRTABLE_COMPANY_FIELD", "Company"),
            fallback_id=os.environ.get("AIRTABLE_COMPANIES_TABLE", AIRTABLE_COMPANIES_TABLE_ID),
        )
        self.companies = self.base.table(companies_table_id)

    def _discover_companies_table(self, company_field_name: str, fallback_id: str) -> str:
        """
        Return the table ID that the Roles table's Company linked field points to.
        Falls back to fallback_id if the schema API fails or the field is not found.
        """
        try:
            schema = self.roles.schema()
            for field in schema.fields:
                if field.name == company_field_name and field.type == "multipleRecordLinks":
                    linked_id = field.options.linked_table_id
                    logger.info(
                        "Discovered Companies table via Roles.%r schema: %s",
                        company_field_name, linked_id,
                    )
                    return linked_id
            logger.warning(
                "Field %r not found or not a link field in Roles schema; falling back to %s",
                company_field_name, fallback_id,
            )
        except Exception:
            logger.warning(
                "Could not fetch Roles schema to discover Companies table; falling back to %s",
                fallback_id,
            )
        return fallback_id

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
            records = self.companies.all(formula=f"LOWER({{Company Name}}) = LOWER('{name_q}')")
            if records:
                logger.info("find_company exact match for %r: %r", company_name, records[0]["fields"].get("Company Name"))
                return records[0]
        except Exception:
            logger.warning("find_company exact formula failed for %r, trying partial", company_name)

        # 2. Partial / substring match fallback.
        try:
            records = self.companies.all(formula=f"SEARCH(LOWER('{name_q}'), LOWER({{Company Name}}))")
            if records:
                logger.info("find_company partial match for %r: %r", company_name, records[0]["fields"].get("Company Name"))
                return records[0]
        except Exception:
            logger.exception("find_company partial search failed for %r", company_name)

        # 3. Fuzzy fallback — catches misspellings (e.g. "monjour" → "Monjur").
        try:
            all_records = self.companies.all(fields=["Company Name"])
            name_to_record = {
                r["fields"]["Company Name"].lower(): r
                for r in all_records
                if r["fields"].get("Company Name")
            }
            matches = get_close_matches(company_name.lower(), name_to_record.keys(), n=1, cutoff=0.6)
            if matches:
                logger.info("find_company fuzzy match for %r → %r", company_name, matches[0])
                return name_to_record[matches[0]]
        except Exception:
            logger.exception("find_company fuzzy fallback failed for %r", company_name)

        return None

    def get_company(self, record_id: str) -> Optional[dict]:
        try:
            return self.companies.get(record_id)
        except Exception:
            logger.exception("get_company failed for %r", record_id)
            return None

    def get_company_roles(self, company_id: str) -> list[dict]:
        """Return all roles linked to a company.

        Airtable resolves ARRAYJOIN({Company}) to display values (company names),
        not record IDs, so we must filter by name rather than by ID.
        """
        try:
            co = self.get_company(company_id)
            company_name = (co.get("fields") or {}).get("Company Name", "") if co else ""
            if company_name:
                name_q = company_name.lower().replace("'", "\\'")
                formula = f"SEARCH(LOWER('{name_q}'), LOWER(ARRAYJOIN({{Company}})))"
                return self.roles.all(formula=formula)
            # Fallback: ID-based (works if Airtable config differs)
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

        Uses ARRAYJOIN({Company}) in the formula which — in Airtable's filterByFormula
        API — resolves linked records to their display (primary-field) values, i.e.
        company names.  Searching by name is therefore correct; searching by record ID
        does NOT work here.
        """
        try:
            title_q = role_title.lower().replace("'", "\\'")
            company_q = company_name.lower().replace("'", "\\'")
            formula = (
                f"AND("
                f"SEARCH(LOWER(\'{title_q}\'), LOWER({{Title}})), "
                f"SEARCH(LOWER(\'{company_q}\'), LOWER(ARRAYJOIN({{Company}})))"
                f")"
            )
            records = self.roles.all(formula=formula)
            logger.info(
                "find_role_for_company: %d result(s) for title=%r company=%r",
                len(records), role_title, company_name,
            )
            co = self.find_company(company_name)

            if not records:
                # Fuzzy fallback 1: resolve the real company name then retry title by ID.
                if co:
                    role = self.find_role(role_title, company_id=co["id"])
                    if role:
                        logger.info("find_role_for_company fuzzy hit: %r at %r", role_title, co["fields"].get("Company Name"))
                        return role, co

                # Fuzzy fallback 2: search Notes field at this company for the role keywords.
                try:
                    co_for_notes = co or self.find_company(company_name)
                    if co_for_notes:
                        notes_formula = (
                            f"AND("
                            f"SEARCH(LOWER('{title_q}'), LOWER({{Notes}})), "
                            f"SEARCH(LOWER('{company_q}'), LOWER(ARRAYJOIN({{Company}})))"
                            f")"
                        )
                        notes_records = self.roles.all(formula=notes_formula)
                        if notes_records:
                            logger.info("find_role_for_company notes hit: %r at %r", role_title, company_name)
                            return notes_records[0], co_for_notes
                except Exception:
                    logger.debug("find_role_for_company notes search failed for %r / %r", role_title, company_name)

                # Return the company even when no role found — callers use it for semantic matching / fallback.
                return None, co

            role = records[0]
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

    def get_location_options(self) -> list[str]:
        """Return valid HQ Location picklist values from the Roles table schema (cached)."""
        if hasattr(self, "_location_options_cache"):
            return self._location_options_cache
        try:
            schema = self.roles.schema()
            for field in schema.fields:
                if field.name == "HQ Location":
                    choices = getattr(getattr(field, "options", None), "choices", None) or []
                    options = [c.name for c in choices]
                    logger.info("Fetched HQ Location options: %r", options)
                    self._location_options_cache = options
                    return options
        except Exception:
            logger.warning("Could not fetch HQ Location schema options")
        self._location_options_cache = []
        return []
