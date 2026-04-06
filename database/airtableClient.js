'use strict';

/**
 * Airtable client for the Insights agent database operations.
 *
 * Tables:
 * - Companies (tblk2Et7RYIVCWRzD): company profiles
 * - Roles: open positions linked to companies
 */

const Airtable = require('airtable');

const AIRTABLE_BASE_ID = 'appo2zjaaetcT88Fx';
const AIRTABLE_COMPANIES_TABLE_ID = 'tblk2Et7RYIVCWRzD';

/**
 * Simple Dice coefficient similarity — approximates Python difflib SequenceMatcher ratio.
 * @param {string} a
 * @param {string} b
 * @returns {number} 0–1
 */
function _similarity(a, b) {
  if (a === b) return 1.0;
  if (a.length < 2 || b.length < 2) return 0.0;
  const bigrams = new Map();
  for (let i = 0; i < a.length - 1; i++) {
    const bg = a.slice(i, i + 2);
    bigrams.set(bg, (bigrams.get(bg) || 0) + 1);
  }
  let intersection = 0;
  for (let i = 0; i < b.length - 1; i++) {
    const bg = b.slice(i, i + 2);
    if (bigrams.has(bg) && bigrams.get(bg) > 0) {
      intersection++;
      bigrams.set(bg, bigrams.get(bg) - 1);
    }
  }
  return (2.0 * intersection) / (a.length + b.length - 2);
}

/**
 * Return up to n strings from possibilities with similarity >= cutoff.
 * Equivalent to Python's difflib.get_close_matches().
 * @param {string} word
 * @param {string[]} possibilities
 * @param {number} n
 * @param {number} cutoff
 * @returns {string[]}
 */
function getCloseMatches(word, possibilities, n = 3, cutoff = 0.6) {
  return possibilities
    .map(p => ({ p, score: _similarity(word, p) }))
    .filter(({ score }) => score >= cutoff)
    .sort((a, b) => b.score - a.score)
    .slice(0, n)
    .map(({ p }) => p);
}

/** Normalise an Airtable Record object to a plain {id, fields} dict. */
function toDict(record) {
  return { id: record.id, fields: record.fields };
}

/** Returns false for roles whose Status contains "confidential" — these are never surfaced. */
function _notConfidential(role) {
  const raw = (role.fields || {}).Status;
  // Status may be a string ("Posted"), an array (["Posted"]), or null/undefined.
  const status = (Array.isArray(raw) ? raw.join(',') : String(raw || '')).toLowerCase();
  return !status.includes('confidential');
}

class AirtableClient {
  constructor({ apiKey, baseId, rolesTableName, companiesTableFallback, companyFieldName }) {
    this._apiKey = apiKey;
    this._baseId = baseId;
    this._rolesTableName = rolesTableName;
    this._companiesTableFallback = companiesTableFallback;
    this._companyFieldName = companyFieldName;

    this._base = new Airtable({ apiKey }).base(baseId);
    this.roles = this._base(rolesTableName);
    this.companies = null; // set after async init
    this._locationOptionsCache = null;
    this._tablesMetaCache = null; // cache for metadata API response
    this._statusNameCache = null; // Map<recordId, statusName> for linked Status field
  }

  /**
   * Factory: creates and fully initialises an AirtableClient.
   * @returns {Promise<AirtableClient>}
   */
  static async create() {
    const client = new AirtableClient({
      apiKey: process.env.AIRTABLE_API_KEY,
      baseId: process.env.AIRTABLE_BASE_ID || AIRTABLE_BASE_ID,
      rolesTableName: process.env.AIRTABLE_ROLES_TABLE || 'Roles',
      companiesTableFallback: process.env.AIRTABLE_COMPANIES_TABLE || AIRTABLE_COMPANIES_TABLE_ID,
      companyFieldName: process.env.AIRTABLE_COMPANY_FIELD || 'Company',
    });
    const companiesTableId = await client._discoverCompaniesTable();
    client.companies = client._base(companiesTableId);
    return client;
  }

  /** Fetch Airtable Metadata API (tables schema) with caching. */
  async _getTablesMeta() {
    if (this._tablesMetaCache) return this._tablesMetaCache;
    const resp = await fetch(
      `https://api.airtable.com/v0/meta/bases/${this._baseId}/tables`,
      { headers: { Authorization: `Bearer ${this._apiKey}` } }
    );
    if (!resp.ok) throw new Error(`Airtable metadata API returned ${resp.status}`);
    const data = await resp.json();
    this._tablesMetaCache = data.tables || [];
    return this._tablesMetaCache;
  }

  /**
   * Build a cache of Status linked-record IDs → status name strings.
   * The Roles table's "Status" field is a linked record field; this resolves
   * those IDs to display values (e.g. "Posted", "Closed", "Unposted - Company").
   */
  async _ensureStatusNameCache() {
    if (this._statusNameCache) return;
    this._statusNameCache = new Map();
    try {
      const tables = await this._getTablesMeta();
      const rolesTable = tables.find(t => t.name === this._rolesTableName || t.id === this._rolesTableName);
      if (!rolesTable) return;
      const statusField = rolesTable.fields.find(
        f => f.name === 'Status' && f.type === 'multipleRecordLinks'
      );
      if (!statusField || !statusField.options || !statusField.options.linkedTableId) return;
      const linkedTable = tables.find(t => t.id === statusField.options.linkedTableId);
      if (!linkedTable) return;
      const statusRecords = await this._base(linkedTable.name).select().all();
      for (const r of statusRecords) {
        const name = Object.values(r.fields || {})[0];
        if (name != null) this._statusNameCache.set(r.id, String(name));
      }
      console.info(`[statusCache] loaded ${this._statusNameCache.size} status options from "${linkedTable.name}"`);
    } catch (e) {
      console.warn(`[statusCache] failed to build: ${e.message}`);
    }
  }

  /** Resolve linked Status record IDs to a plain status string in-place. */
  _resolveStatus(roleDict) {
    const raw = (roleDict.fields || {}).Status;
    if (!Array.isArray(raw) || raw.length === 0) return;
    const name = this._statusNameCache && this._statusNameCache.get(raw[0]);
    if (name) roleDict.fields.Status = name;
  }

  async _discoverCompaniesTable() {
    try {
      const tables = await this._getTablesMeta();
      const rolesTable = tables.find(t => t.name === this._rolesTableName || t.id === this._rolesTableName);
      if (!rolesTable) throw new Error(`Table '${this._rolesTableName}' not found in schema`);
      const field = rolesTable.fields.find(
        f => f.name === this._companyFieldName && f.type === 'multipleRecordLinks'
      );
      if (field) {
        const linkedId = field.options.linkedTableId;
        console.info(`Discovered Companies table via Roles.${this._companyFieldName} schema: ${linkedId}`);
        return linkedId;
      }
      console.warn(`Field '${this._companyFieldName}' not found or not a link field; falling back to ${this._companiesTableFallback}`);
    } catch (err) {
      console.warn(`Could not fetch Roles schema to discover Companies table: ${err.message}; falling back to ${this._companiesTableFallback}`);
    }
    return this._companiesTableFallback;
  }

  // -------------------------------------------------------------------------
  // Company lookups
  // -------------------------------------------------------------------------

  /**
   * Search for a company by name (exact → partial → fuzzy).
   * @param {string} companyName
   * @returns {Promise<object|null>}
   */
  async findCompany(companyName) {
    const nameQ = companyName.toLowerCase().replace(/'/g, "\\'");

    // 1. Exact case-insensitive match
    try {
      const records = await this.companies.select({
        filterByFormula: `LOWER({Company Name}) = LOWER('${nameQ}')`,
      }).all();
      if (records.length > 0) {
        console.info(`findCompany exact match for '${companyName}': '${records[0].fields['Company Name']}'`);
        return toDict(records[0]);
      }
    } catch (err) {
      console.warn(`findCompany exact formula failed for '${companyName}', trying partial`);
    }

    // 2. Partial / substring match
    try {
      const records = await this.companies.select({
        filterByFormula: `SEARCH(LOWER('${nameQ}'), LOWER({Company Name}))`,
      }).all();
      if (records.length > 0) {
        console.info(`findCompany partial match for '${companyName}': '${records[0].fields['Company Name']}'`);
        return toDict(records[0]);
      }
    } catch (err) {
      console.warn(`findCompany partial search failed for '${companyName}': ${err.message}`);
    }

    // 3. Fuzzy fallback — fetch all fields so Domain etc. are present on returned records
    try {
      const allRecords = await this.companies.select().all();
      const nameToRecord = {};
      for (const r of allRecords) {
        const name = (r.fields['Company Name'] || '').toLowerCase();
        if (name) nameToRecord[name] = toDict(r);
      }
      const matches = getCloseMatches(companyName.toLowerCase(), Object.keys(nameToRecord), 1, 0.5);
      if (matches.length > 0) {
        console.info(`findCompany fuzzy match for '${companyName}' → '${matches[0]}'`);
        return nameToRecord[matches[0]];
      }
    } catch (err) {
      console.warn(`findCompany fuzzy fallback failed for '${companyName}': ${err.message}`);
    }

    return null;
  }

  /**
   * @param {string} recordId
   * @returns {Promise<object|null>}
   */
  async getCompany(recordId) {
    try {
      const record = await this.companies.find(recordId);
      return toDict(record);
    } catch (err) {
      console.warn(`getCompany failed for '${recordId}': ${err.message}`);
      return null;
    }
  }

  /**
   * Return all roles linked to a company.
   * Uses exact-element match (comma-boundary trick) to avoid false matches.
   * @param {string} companyId
   * @returns {Promise<Array>}
   */
  async getCompanyRoles(companyId) {
    console.info(`[getCompanyRoles] start — companyId=${companyId}`);
    try {
      const co = await this.getCompany(companyId);
      const companyName = ((co || {}).fields || {})['Company Name'] || '';
      console.info(`[getCompanyRoles] companyName='${companyName}'`);

      // Ensure Status linked-record IDs are resolved to name strings before any filtering
      await this._ensureStatusNameCache();

      if (companyName) {
        const nameEscaped = companyName.replace(/'/g, "\\'");
        const formula = `{Company} = '${nameEscaped}'`;
        console.info(`[getCompanyRoles] trying formula: ${formula}`);
        try {
          const records = await this.roles.select({ filterByFormula: formula }).all();
          console.info(`[getCompanyRoles] formula returned ${records.length} record(s)`);
          if (records.length > 0) {
            console.info(`[getCompanyRoles] first record Company=${JSON.stringify(records[0].fields.Company)}`);
            const dicts = records.map(toDict);
            dicts.forEach(d => this._resolveStatus(d));
            return dicts.filter(_notConfidential);
          }
        } catch (e) {
          console.warn(`[getCompanyRoles] formula error: ${e.message}`);
        }
      }

      // Fallback: full table scan filtered by linked record ID in JS
      console.info(`[getCompanyRoles] falling back to full table scan`);
      const allRecords = await this.roles.select().all();
      console.info(`[getCompanyRoles] full table has ${allRecords.length} record(s)`);
      const dicts = allRecords.map(toDict);
      dicts.forEach(d => this._resolveStatus(d));
      const result = dicts.filter(_notConfidential).filter(r => {
        const linked = (r.fields || {}).Company;
        return Array.isArray(linked) && linked.includes(companyId);
      });
      console.info(`[getCompanyRoles] fallback result: ${result.length} record(s)`);
      return result;
    } catch (err) {
      console.warn(`[getCompanyRoles] failed for '${companyId}': ${err.message}`);
      return [];
    }
  }

  /**
   * Return all company records matching companyName (for disambiguation).
   * @param {string} companyName
   * @returns {Promise<Array>}
   */
  async findCompanies(companyName) {
    const nameQ = companyName.toLowerCase().replace(/'/g, "\\'");

    // 1. Exact match — unambiguous
    try {
      const exact = await this.companies.select({
        filterByFormula: `LOWER({Company Name}) = LOWER('${nameQ}')`,
      }).all();
      if (exact.length > 0) return exact.map(toDict);
    } catch (err) {
      console.warn(`findCompanies exact formula failed for '${companyName}'`);
    }

    // 2. Partial / substring match
    try {
      const records = await this.companies.select({
        filterByFormula: `SEARCH(LOWER('${nameQ}'), LOWER({Company Name}))`,
      }).all();
      if (records.length > 0) return records.map(toDict);
    } catch (err) {
      console.warn(`findCompanies partial search failed for '${companyName}': ${err.message}`);
    }

    // 3. Fuzzy fallback — fetch all fields so Domain etc. are present on returned records
    try {
      const allRecords = await this.companies.select().all();
      const nameToRecord = {};
      for (const r of allRecords) {
        const name = (r.fields['Company Name'] || '').toLowerCase();
        if (name) nameToRecord[name] = toDict(r);
      }
      const matches = getCloseMatches(companyName.toLowerCase(), Object.keys(nameToRecord), 3, 0.5);
      return matches.map(m => nameToRecord[m]);
    } catch (err) {
      console.warn(`findCompanies fuzzy fallback failed for '${companyName}': ${err.message}`);
    }

    return [];
  }

  // -------------------------------------------------------------------------
  // Role lookups
  // -------------------------------------------------------------------------

  /**
   * Search for a role by title, optionally scoped to a company.
   * @param {string} roleTitle
   * @param {string|null} companyId
   * @returns {Promise<object|null>}
   */
  async findRole(roleTitle, companyId = null) {
    try {
      const titleFilter = `SEARCH(LOWER('${roleTitle.toLowerCase()}'), LOWER({Title}))`;
      const formula = companyId
        ? `AND(${titleFilter}, FIND('${companyId}', ARRAYJOIN({Company})))`
        : titleFilter;
      const records = await this.roles.select({ filterByFormula: formula }).all();
      const visible = records.map(toDict).filter(_notConfidential);
      return visible.length > 0 ? visible[0] : null;
    } catch (err) {
      console.warn(`findRole failed for '${roleTitle}': ${err.message}`);
      return null;
    }
  }

  /**
   * Find a role by title scoped to a company.
   * Returns [roleRecord, companyRecord, matchType].
   * matchType: 'title' | 'fuzzy' | 'notes' | 'none'
   * @param {string} roleTitle
   * @param {string} companyName
   * @returns {Promise<[object|null, object|null, string]>}
   */
  async findRoleForCompany(roleTitle, companyName) {
    try {
      const titleQ = roleTitle.toLowerCase().replace(/'/g, "\\'");
      const companyQ = companyName.toLowerCase().replace(/'/g, "\\'");
      const formula = (
        `AND(` +
        `SEARCH(LOWER('${titleQ}'), LOWER({Title})), ` +
        `SEARCH(LOWER('${companyQ}'), LOWER(ARRAYJOIN({Company})))` +
        `)`
      );
      const records = await this.roles.select({ filterByFormula: formula }).all();
      console.info(`findRoleForCompany: ${records.length} result(s) for title='${roleTitle}' company='${companyName}'`);

      const co = await this.findCompany(companyName);

      if (records.length === 0) {
        // Fuzzy fallback 1: resolve real company name then retry title by ID
        if (co) {
          const role = await this.findRole(roleTitle, co.id);
          if (role) {
            console.info(`findRoleForCompany fuzzy hit: '${roleTitle}' at '${(co.fields || {})['Company Name']}'`);
            return [role, co, 'fuzzy'];
          }
        }

        // Fuzzy fallback 2: search Notes field at this company
        try {
          const coForNotes = co || await this.findCompany(companyName);
          if (coForNotes) {
            const notesFormula = (
              `AND(` +
              `SEARCH(LOWER('${titleQ}'), LOWER({Notes})), ` +
              `SEARCH(LOWER('${companyQ}'), LOWER(ARRAYJOIN({Company})))` +
              `)`
            );
            const notesRecords = await this.roles.select({ filterByFormula: notesFormula }).all();
            if (notesRecords.length > 0) {
              console.info(`findRoleForCompany notes hit: '${roleTitle}' at '${companyName}'`);
              return [toDict(notesRecords[0]), coForNotes, 'notes'];
            }
          }
        } catch (err) {
          console.debug(`findRoleForCompany notes search failed for '${roleTitle}' / '${companyName}'`);
        }

        return [null, co, 'none'];
      }

      return [toDict(records[0]), co, 'title'];
    } catch (err) {
      console.warn(`findRoleForCompany failed for '${roleTitle}' / '${companyName}': ${err.message}`);
      return [null, null, 'none'];
    }
  }

  /**
   * @param {string} recordId
   * @returns {Promise<object|null>}
   */
  async findRoleById(recordId) {
    try {
      const record = await this.roles.find(recordId);
      await this._ensureStatusNameCache();
      const d = toDict(record);
      this._resolveStatus(d);
      return d;
    } catch (err) {
      console.warn(`findRoleById failed for '${recordId}': ${err.message}`);
      return null;
    }
  }

  // -------------------------------------------------------------------------
  // Write operations
  // -------------------------------------------------------------------------

  /**
   * Create a new Company record.
   * @param {string} companyName
   * @param {string|null} domainDirty  Raw domain string (e.g. 'acme.com')
   * @returns {Promise<object|null>}
   */
  async createCompany(companyName, domainDirty) {
    try {
      const fields = { 'Company Name': companyName };
      if (domainDirty) fields['Domain Dirty'] = domainDirty;
      const record = await this.companies.create(fields);
      console.info(`createCompany: created '${companyName}' → ${record.id}`);
      return toDict(record);
    } catch (err) {
      console.warn(`createCompany failed for '${companyName}': ${err.message}`);
      return null;
    }
  }

  /**
   * Create a new Role record linked to a company.
   * @param {string|null} companyRecordId  Airtable record ID for the linked company
   * @param {string} roleTitle
   * @param {object} [roleData]  Optional fields: find, notes, location, compensation
   * @returns {Promise<object|null>}
   */
  async createRole(companyRecordId, roleTitle, roleData = {}) {
    try {
      const fields = { 'Title': roleTitle };
      if (companyRecordId) fields['Company'] = [companyRecordId];
      if (roleData.find)             fields['Role - Find']         = roleData.find;
      if (roleData.notes)            fields['Role - Notes']        = roleData.notes;
      if (roleData.location)         fields['Role - Location']     = roleData.location;
      if (roleData.compensation != null) fields['Role - Compensation'] = roleData.compensation;
      const record = await this.roles.create(fields);
      console.info(`createRole: created '${roleTitle}' → ${record.id}`);
      return toDict(record);
    } catch (err) {
      console.warn(`createRole failed for '${roleTitle}': ${err.message}`);
      return null;
    }
  }

  /**
   * Update a role's Status linked-record field to the "Closed" option.
   * Finds the status record ID whose name contains "closed" (case-insensitive).
   * @param {string} roleRecordId
   * @returns {Promise<boolean>} true on success
   */
  async markRoleClosed(roleRecordId) {
    try {
      await this._ensureStatusNameCache();
      let closedId = null;
      for (const [id, name] of this._statusNameCache.entries()) {
        if (/closed/i.test(name)) { closedId = id; break; }
      }
      if (!closedId) {
        console.warn('markRoleClosed: no "Closed" status record found in cache');
        return false;
      }
      await this.roles.update(roleRecordId, { Status: [closedId] });
      console.info(`markRoleClosed: updated role ${roleRecordId} → Status=[${closedId}]`);
      return true;
    } catch (err) {
      console.warn(`markRoleClosed failed for role ${roleRecordId}: ${err.message}`);
      return false;
    }
  }

  /**
   * Return valid Region picklist values from the Roles table schema (cached).
   * @returns {Promise<string[]>}
   */
  async getLocationOptions() {
    if (this._locationOptionsCache !== null) return this._locationOptionsCache;
    try {
      const tables = await this._getTablesMeta();
      const rolesTable = tables.find(t => t.name === this._rolesTableName || t.id === this._rolesTableName);
      if (rolesTable) {
        const regionField = rolesTable.fields.find(f => f.name === 'Region');
        if (regionField && regionField.options && regionField.options.choices) {
          const options = regionField.options.choices.map(c => c.name);
          console.info(`Fetched Region options: ${JSON.stringify(options)}`);
          this._locationOptionsCache = options;
          return options;
        }
      }
    } catch (err) {
      console.warn(`Could not fetch Region schema options: ${err.message}`);
    }
    this._locationOptionsCache = [];
    return [];
  }
}

module.exports = AirtableClient;
