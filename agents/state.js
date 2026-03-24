'use strict';

/**
 * In-memory conversation state manager for the Insights agent.
 */

const Phase = Object.freeze({
  IDENTIFY: 'IDENTIFY',                        // Waiting for user to name a company/role
  CONFIRMING: 'CONFIRMING',                    // Legacy: kept for safety; no longer entered in normal flow
  DISAMBIGUATING: 'DISAMBIGUATING',            // Multiple role matches — waiting for user to pick one
  DISAMBIGUATING_COMPANY: 'DISAMBIGUATING_COMPANY', // Multiple company matches — waiting for user to pick one
  AWAITING_SHARE: 'AWAITING_SHARE',            // Basic mode: waiting for user to share what they know
  COMPANY_FOUND: 'COMPANY_FOUND',              // Synopsis shown, answering follow-ups
  ROLE_FOUND: 'ROLE_FOUND',                    // Role synopsis shown, answering follow-ups
  COLLECTING_NEW_ENTITY: 'COLLECTING_NEW_ENTITY', // Collecting info to add a new company/role to the DB
});

class ConversationState {
  constructor(userId, userName = '', mode = 'premium') {
    this.userId = userId;
    this.userName = userName;
    this.mode = mode; // 'free', 'pro', or 'premium'

    this.phase = Phase.IDENTIFY;

    this.companyRecordId = null;
    this.companyName = null;
    this.companyDomain = null;  // e.g. 'https://maintainx.com'
    this.roleRecordId = null;
    this.roleTitle = null;
    this.roleAppPage = null;    // e.g. 'https://app.whisperedagent.com/roles/rec...'

    // Role candidates waiting for user disambiguation (list of Airtable record IDs)
    this.candidateRoleIds = [];

    // Company candidates waiting for user disambiguation (list of Airtable record IDs)
    this.candidateCompanyIds = [];

    // Suggested field updates accumulated from the conversation (not written to Airtable)
    this.suggestedUpdates = {};

    // Pending new entity being collected for DB insertion
    // Shape: { companyName, roleTitle, domain, find, notes, step, hasExistingCompany }
    // step: 'confirm' | 'get_company' | 'domain' | 'find' | 'notes'
    this.pendingNewEntity = null;

    // Set to true when user signals "I have a new/another role" so the next role name
    // skips DB lookup and goes straight to the collection flow.
    this.pendingNewRoleSignal = false;

    // Full message history for Claude
    this.messages = [];
  }

  addUserMessage(text) {
    this.messages.push({ role: 'user', content: text });
  }

  addAssistantMessage(text) {
    this.messages.push({ role: 'assistant', content: text });
  }
}

class StateManager {
  constructor() {
    this._store = new Map();
  }

  get(userId) {
    return this._store.get(userId) || null;
  }

  getOrCreate(userId, userName = '', mode = 'premium') {
    if (!this._store.has(userId)) {
      this._store.set(userId, new ConversationState(userId, userName, mode));
    }
    return this._store.get(userId);
  }

  reset(userId, userName = '', mode = 'premium') {
    const state = new ConversationState(userId, userName, mode);
    this._store.set(userId, state);
    return state;
  }

  delete(userId) {
    this._store.delete(userId);
  }
}

module.exports = { Phase, ConversationState, StateManager };
