'use strict';

/**
 * Insights Agent – core conversation logic.
 *
 * Tier behaviour
 * ──────────────
 * Premium  Immediately share brief synopsis after confirmation, then dialog to
 *          fill gaps.  Includes contributor attribution in insights.
 *
 * Pro      Ask user to share first (AWAITING_SHARE) — except:
 *            - Posted roles: share existence at Pro-tier level without share gate.
 *            - Closed roles: share all information without share gate.
 *          Then brief synopsis minus contributor attribution; dialog to fill gaps.
 *
 * Free     Roles:     Ask user to share first (AWAITING_SHARE) — except:
 *                       - Posted roles: share existence at Free-tier level without share gate.
 *                       - Closed roles: share all information without share gate.
 *                     Confirm we have it but share no details for other roles;
 *                     ask questions; mention upgrade.
 *          Companies: Immediately share brief public snapshot; then dialog for
 *                     supplemental confidential info.
 *
 * All tiers: data shared by users is accumulated in state.suggestedUpdates
 *            (visible in the UI panel but NOT written to Airtable).
 */

const Anthropic = require('@anthropic-ai/sdk');
const { Phase } = require('./state.js');
const {
  buildCompanySynopsisPrompt,
  buildRoleSynopsisPrompt,
  buildRolesListingPrompt,
} = require('../prompts/synopsis.js');
const {
  buildDataExtractionPrompt,
  buildStructuredMergePrompt,
  buildSimpleFieldMergePrompt,
  getRoleGaps,
  getCompanyGaps,
} = require('../prompts/dataCollection.js');

const CLAUDE_MODEL = process.env.CLAUDE_MODEL || 'claude-sonnet-4-6';

const SYSTEM_PROMPT = `You are the Insights agent for a professional community platform that tracks companies and open roles.
Your goal is to help community members and to learn from them — every conversation is a chance to fill in gaps in our knowledge about roles and companies.

Guidelines:
- Be warm, concise, and conversational. You are a knowledgeable colleague, not a database.
- Keep responses SHORT. 2-4 sentences is usually right. Leave room for dialogue.
- Ask ONE question at a time. Never list questions.
- Ask open, natural questions — not "what is the team size?" but "what's the team setup like there?"
- Bold the sentence containing your question using **double asterisks like this?** — do not bold anything else.
- Do not use any other markdown (no ##, no -, no _).
- CRITICAL: Your questions must always be about gathering intelligence on the role or company — hiring process, team structure, hiring manager, location/remote setup, compensation, company health, GTM motion, culture. NEVER ask why the user wants the role, what draws them to it, or anything about their personal background, motivations, or career goals. Those are not your job.
`;

// Pronoun/reference words that signal "the role we were just talking about"
const PRONOUN_ROLE_TOKENS = new Set([
  'that', 'this', 'it', 'one', 'the', 'same', 'above', 'mentioned',
  'role', 'position', 'job', 'opening',
]);

class InsightsAgent {
  constructor(db, stateManager) {
    this.db = db;
    this.stateManager = stateManager;
    this.client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
  }

  // ------------------------------------------------------------------
  // Public entry points
  // ------------------------------------------------------------------

  async startConversation(userId, userName, mode = 'premium') {
    const state = this.stateManager.reset(userId, userName, mode);
    let greeting;

    if (mode === 'free') {
      greeting = (
        "Ask about any company or role—I'll share public info (upgrade for unposted roles).\n\n" +
        "Contribute roles or insights to help fellow execs and unlock deeper insights"
      );
    } else if (mode === 'pro') {
      greeting = (
        "Tell me about a company or role you're exploring. " +
        "Share what you've learned and I'll pull up what we have — " +
        'your insights help the whole community.'
      );
    } else { // premium
      greeting = (
        "Which company/role are you interested in? I can share what we have and we can compare notes.\n\n" +
        "And, I can capture new insights/roles to help other members."
      );
    }

    state.addAssistantMessage(greeting);
    return greeting;
  }

  async handleMessage(userId, userName, userText, mode = 'premium') {
    const state = this.stateManager.getOrCreate(userId, userName, mode);
    if (state.mode !== mode) state.mode = mode;

    state.addUserMessage(userText);

    // Free tier: gate any request to see unposted/confidential roles upfront,
    // regardless of conversation phase, with the full confidentiality explanation.
    if (mode === 'free' && this._isUnpostedRolesRequest(userText)) {
      const coRef = state.companyName ? ` at ${this._companyRef(state)}` : '';
      const reply = (
        `Unposted roles are shared with us in confidence by recruiters, companies and Whispered paid members. ` +
        `We get these roles because all parties trust that roles shared with Whispered remain confidential and don't spread publicly — so we can only share them with paying members who've agreed to our community standards.\n\n` +
        `If you fit the criteria, apply for Pro/Premium and chat with our team about unlocking all roles and confidential company insights.\n\n` +
        `[Apply for our paid plans](https://www.whispered.com/join)`
      );
      state.addAssistantMessage(reply);
      return reply;
    }

    let reply;
    if (state.phase === Phase.IDENTIFY) {
      reply = await this._handleIdentify(state, userText);
    } else if (state.phase === Phase.CONFIRMING) {
      reply = await this._handleConfirming(state, userText);
    } else if (state.phase === Phase.DISAMBIGUATING) {
      reply = await this._handleDisambiguating(state, userText);
    } else if (state.phase === Phase.DISAMBIGUATING_COMPANY) {
      reply = await this._handleDisambiguatingCompany(state, userText);
    } else if (state.phase === Phase.AWAITING_SHARE) {
      reply = await this._handleAwaitingShare(state, userText);
    } else if (state.phase === Phase.COLLECTING_NEW_ENTITY) {
      reply = await this._handleCollectingNewEntity(state, userText);
    } else if (state.phase === Phase.COMPANY_FOUND || state.phase === Phase.ROLE_FOUND) {
      reply = await this._handleFollowup(state, userText);
    } else {
      state.phase = Phase.IDENTIFY;
      reply = await this._handleIdentify(state, userText);
    }

    state.addAssistantMessage(reply);
    return reply;
  }

  // ------------------------------------------------------------------
  // Phase handlers
  // ------------------------------------------------------------------

  async _handleIdentify(state, userText) {
    const parsed = await this._parseCompanyAndRole(userText);
    const companyName = parsed.company;
    const roleTitle = parsed.role;

    if (!companyName && !roleTitle) {
      // User may be signalling they want to contribute a new role without naming it yet
      if (this._isNewRoleSignal(userText)) {
        state.pendingNewRoleSignal = true;
        return (
          "Great — **what's the company and role title?** For example: " +
          '"VP of Sales at Acme Corp".'
        );
      }
      return (
        "I didn't catch a company or role name there. " +
        '**Could you try again? For example: "Acme Corp" or "Product Manager at Acme Corp".**'
      );
    }

    let companyRecord = null;
    let roleRecord = null;
    let matchType = 'none';

    // Fall back to the company already in state if the message didn't name one
    const effectiveCompany = companyName || state.companyName;

    if (roleTitle && effectiveCompany) {
      [roleRecord, companyRecord, matchType] = await this.db.findRoleForCompany(roleTitle, effectiveCompany);
    } else if (roleTitle) {
      roleRecord = await this.db.findRole(roleTitle);
    } else if (companyName) {
      // Use findCompanies (plural) so we can detect ambiguity.
      const candidates = await this.db.findCompanies(companyName);
      if (candidates.length > 1) {
        return this._askDisambiguateCompany(state, candidates);
      } else if (candidates.length === 1) {
        companyRecord = candidates[0];
        // If the matched name differs from the query (fuzzy match only), confirm before proceeding.
        const matchedName = this._field(companyRecord.fields, 'Company Name').toLowerCase();
        const queriedName = companyName.toLowerCase();
        const isFuzzyOnly = !matchedName.includes(queriedName) && !queriedName.includes(matchedName);
        if (isFuzzyOnly) {
          const confirmedName = this._field(companyRecord.fields, 'Company Name');
          state.pendingCompanyId = companyRecord.id;
          state.pendingCompanyName = confirmedName;
          state.phase = Phase.CONFIRMING;
          return `I don't have **${companyName}** in our database — did you mean **${confirmedName}**?`;
        }
      }
      // else: no match — fall through to not-found handling below
    }

    if (roleRecord && !companyRecord) {
      const linked = (roleRecord.fields.Company || []);
      if (linked.length > 0) {
        companyRecord = await this.db.getCompany(linked[0]);
      }
    }

    // For weak/no DB matches try direct title substring matching.
    if (companyRecord && roleTitle && (matchType === 'notes' || matchType === 'none')) {
      const companyRoles = this._rolesForTier(
        await this.db.getCompanyRoles(companyRecord.id),
        state.mode
      );
      if (companyRoles.length > 0) {
        if (companyRoles.length === 1) {
          roleRecord = companyRoles[0];
        } else {
          const queryLower = roleTitle.toLowerCase();
          const titleHits = companyRoles.filter(r => {
            const title = this._field(r.fields, 'Title').toLowerCase();
            return queryLower.includes(title) || title.includes(queryLower);
          });
          if (titleHits.length === 1) {
            roleRecord = titleHits[0];
          } else if (titleHits.length > 1) {
            return this._askDisambiguate(state, companyRecord, titleHits);
          }
          // titleHits === 0: no title match → fall through to role-not-found handling.
          // (Semantic filter deliberately removed — it kept all GTM-adjacent roles as
          // "plausibly related" which showed false disambiguation for genuinely new roles.)
        }
      }
    }

    if (!roleRecord && !companyRecord) {
      return this._startNewEntityCollection(state, companyName, roleTitle, false);
    }

    // Company found but role not in DB → start collection flow (all tiers)
    if (!roleRecord && companyRecord && roleTitle) {
      return await this._roleNotFoundAtCompany(state, companyRecord, roleTitle);
    }

    if (companyRecord) {
      state.companyRecordId = companyRecord.id;
      state.companyName = this._field(companyRecord.fields, 'Company Name', companyName || '');
      const rawDomain = this._field(companyRecord.fields, 'Domain');
      state.companyDomain = this._ensureHttps(rawDomain.trim());
    } else if (companyName) {
      state.companyName = companyName;
    }

    // Company found but user signalled they want to contribute a new role → ask for the title
    if (companyRecord && !roleRecord && !roleTitle && this._isNewRoleSignal(userText)) {
      state.pendingNewRoleSignal = true;
      const coRef = this._companyRef(state);
      return `Got it — **what's the role title at ${coRef}?**`;
    }

    if (roleRecord) {
      state.roleRecordId = roleRecord.id;
      state.roleTitle = this._field(roleRecord.fields, 'Title', roleTitle || '');
      state.roleAppPage = this._field(roleRecord.fields, 'App Page').trim();
    }

    return await this._dispatchAfterMatch(state, userText);
  }

  async _handleConfirming(state, userText) {
    // Fuzzy company match confirmation: user is responding to "did you mean X?"
    if (state.pendingCompanyId) {
      const lower = userText.toLowerCase().trim();
      const confirmed = /\b(yes|yeah|yep|correct|right|that'?s?\s*(right|it|the one)?|sure|exactly|yup|affirmative)\b/.test(lower);
      const denied = /\b(no|nope|not|wrong|different|other|never mind|nevermind)\b/.test(lower);

      if (confirmed) {
        const companyRecord = await this.db.getCompany(state.pendingCompanyId);
        state.companyRecordId = companyRecord.id;
        state.companyName = this._field(companyRecord.fields, 'Company Name');
        const rawDomain = this._field(companyRecord.fields, 'Domain');
        state.companyDomain = this._ensureHttps(rawDomain.trim());
        state.pendingCompanyId = null;
        state.pendingCompanyName = null;
        return await this._dispatchAfterMatch(state, userText);
      } else if (denied) {
        state.pendingCompanyId = null;
        state.pendingCompanyName = null;
        state.phase = Phase.IDENTIFY;
        return `No problem — which company were you looking for?`;
      } else {
        // Not clear — ask again
        return `Just to confirm — did you mean **${state.pendingCompanyName}**?`;
      }
    }
    // Legacy fallback
    return await this._dispatchAfterMatch(state, userText);
  }

  // ------------------------------------------------------------------
  // Disambiguation helpers
  // ------------------------------------------------------------------

  _askDisambiguate(state, companyRecord, candidates) {
    state.companyRecordId = companyRecord.id;
    state.companyName = this._field(companyRecord.fields, 'Company Name');
    const rawDomain = this._field(companyRecord.fields, 'Domain');
    state.companyDomain = this._ensureHttps(rawDomain.trim());
    state.candidateRoleIds = candidates.map(r => r.id);
    state.phase = Phase.DISAMBIGUATING;

    const coRef = this._companyRef(state);
    const lines = candidates.map((r, i) => {
      const title = (r.fields || {}).Title || 'Untitled';
      const appPage = ((r.fields || {})['App Page'] || '').trim();
      const titleRef = appPage ? `[${title}](${appPage})` : title;
      return `${i + 1}. **${titleRef}**`;
    });

    return (
      `I found a couple of roles at ${coRef} that could be a match — ` +
      `**which of these did you mean?**\n\n${lines.join('\n')}`
    );
  }

  async _handleDisambiguating(state, userText) {
    const candidates = (
      await Promise.all(state.candidateRoleIds.map(id => this.db.findRoleById(id)))
    ).filter(Boolean);

    if (!candidates.length) {
      state.phase = Phase.IDENTIFY;
      state.candidateRoleIds = [];
      return await this._handleIdentify(state, userText);
    }

    const summaries = candidates.map((r, i) => `${i + 1}: ${(r.fields || {}).Title || 'Untitled'}`);
    const prompt = (
      `The user was asked to pick from these roles:\n${summaries.join('\n')}\n\n` +
      `Their reply: "${userText}"\n\n` +
      'Which number did they pick? Return just the integer (1-based), or null if unclear. Valid JSON only.'
    );

    let chosen = null;
    try {
      const raw = await this._callClaude([{ role: 'user', content: prompt }], { maxTokens: 8 });
      const cleaned = raw.trim().replace(/^```json\n?/, '').replace(/^```\n?/, '').replace(/\n?```$/, '').trim();
      const idx = JSON.parse(cleaned);
      if (Number.isInteger(idx) && idx >= 1 && idx <= candidates.length) {
        chosen = candidates[idx - 1];
      }
    } catch (e) {
      console.debug(`_handleDisambiguating parse failed for '${userText}'`);
    }

    if (!chosen) {
      const lines = candidates.map((r, i) => {
        const title = (r.fields || {}).Title || 'Untitled';
        const appPage = ((r.fields || {})['App Page'] || '').trim();
        const titleRef = appPage ? `[${title}](${appPage})` : title;
        return `${i + 1}. **${titleRef}**`;
      });
      return (
        "Sorry, I didn't catch which one — " +
        `**could you pick a number or name from the list?**\n\n${lines.join('\n')}`
      );
    }

    state.candidateRoleIds = [];
    state.roleRecordId = chosen.id;
    state.roleTitle = this._field(chosen.fields, 'Title');
    state.roleAppPage = this._field(chosen.fields, 'App Page').trim();
    return await this._dispatchAfterMatch(state, userText);
  }

  // ------------------------------------------------------------------
  // Company disambiguation
  // ------------------------------------------------------------------

  _askDisambiguateCompany(state, candidates) {
    state.candidateCompanyIds = candidates.map(c => c.id);
    state.phase = Phase.DISAMBIGUATING_COMPANY;

    const lines = candidates.map((c, i) => {
      const name = (c.fields || {})['Company Name'] || 'Unknown';
      const rawDomain = ((c.fields || {}).Domain || '').trim();
      const fullUrl = this._ensureHttps(rawDomain);
      const bareHost = rawDomain.replace(/^https?:\/\//i, '').replace(/\/.*$/, '').trim();
      const domainPart = (fullUrl && bareHost) ? ` ([${bareHost}](${fullUrl}))` : '';
      return `${i + 1}. **${name}**${domainPart}`;
    });

    return (
      `I found a few companies in our database that could match — did you mean one of these or do you have a new company to add?\n\n${lines.join('\n')}`
    );
  }

  async _handleDisambiguatingCompany(state, userText) {
    const candidates = (
      await Promise.all(state.candidateCompanyIds.map(id => this.db.getCompany(id)))
    ).filter(Boolean);

    if (!candidates.length) {
      state.phase = Phase.IDENTIFY;
      state.candidateCompanyIds = [];
      return await this._handleIdentify(state, userText);
    }

    const summaries = candidates
      .map((c, i) => `${i + 1}. ${(c.fields || {})['Company Name'] || 'Unknown'}`)
      .join('\n');
    const prompt = (
      `The user was asked to choose a company from this list:\n${summaries}\n\n` +
      `They replied: "${userText}"\n\n` +
      'Which number did they choose? Reply with just the integer, or 0 if unclear.'
    );

    let chosen = null;
    try {
      const raw = await this._callClaude([{ role: 'user', content: prompt }], { maxTokens: 5 });
      const idx = parseInt(raw.trim().split(/\s+/)[0], 10);
      if (idx >= 1 && idx <= candidates.length) {
        chosen = candidates[idx - 1];
      }
    } catch (e) {
      console.debug(`_handleDisambiguatingCompany parse failed for '${userText}'`);
    }

    if (!chosen) {
      // User didn't pick from the list — they may have said "none of these" or
      // named a different company altogether. Escape disambiguation and re-parse.
      state.candidateCompanyIds = [];
      state.phase = Phase.IDENTIFY;
      return await this._handleIdentify(state, userText);
    }

    state.candidateCompanyIds = [];
    state.companyRecordId = chosen.id;
    state.companyName = this._field(chosen.fields, 'Company Name');
    const rawDomain = this._field(chosen.fields, 'Domain');
    state.companyDomain = this._ensureHttps(rawDomain.trim());
    state.phase = Phase.IDENTIFY; // Let _dispatchAfterMatch set the real phase
    return await this._dispatchAfterMatch(state, userText);
  }

  async _dispatchAfterMatch(state, userText) {
    const mode = state.mode;
    // Fresh entity match — reset the followup counter so we ask at most 2 questions.
    state.insightFollowupsAsked = 0;

    // Roles not visible to this tier — treat as not found.
    if (state.roleRecordId) {
      const roleRec = await this.db.findRoleById(state.roleRecordId);
      if (!this._roleVisibleToTier(roleRec, mode)) {
        state.roleRecordId = null;
        state.roleTitle = null;
        const coRef = this._companyRef(state);
        // Members-only role + free tier: don't even hint it exists
        if (this._roleStatus(roleRec) === 'members-only') {
          return (
            `I don't have that role in our public database. ` +
            `**Upgrade to Pro for access to unposted and whispered roles.**`
          );
        }
        // Confidential: generic not-found response
        if (coRef) {
          return (
            `I don't have that specific role in our database. ` +
            `**Is there another role at ${coRef} you've come across, or would you like to share what you've learned?**`
          );
        }
        return (
          `I don't have that role in our database yet. ` +
          '**Tell me what you know — which company is it at and what have you heard?**'
        );
      }
    }

    const entityRef = this._roleRef(state) || this._companyRef(state);

    const rolesListIntent = (
      !state.roleRecordId &&
      state.companyRecordId &&
      this._isRolesListIntent(userText)
    );

    // FREE + company: role-listing intent → confirm existence only; otherwise show public snapshot
    if (mode === 'free' && !state.roleRecordId) {
      state.phase = Phase.COMPANY_FOUND;
      if (rolesListIntent) {
        return await this._handleFreeRolesListIntent(state);
      }
      const coRec = await this.db.getCompany(state.companyRecordId);
      return await this._generateCompanySynopsis(coRec, 'free', state);
    }

    // FREE + role | PRO (any)
    if (mode === 'free' || mode === 'pro') {
      if (mode === 'pro' && rolesListIntent) {
        state.phase = Phase.COMPANY_FOUND;
        const roles = this._rolesForTier(
          await this.db.getCompanyRoles(state.companyRecordId), 'pro'
        );
        const count = roles.length;
        const noun = count === 1 ? 'role' : 'roles';
        const coRef = this._companyRef(state);
        if (count) {
          return (
            `We do have ${count} ${noun} tracked for ${coRef}. ` +
            '**Upgrade to Premium to see the full breakdown with hiring manager and location details — ' +
            "or is there a specific role you've already heard about?**"
          );
        }
        return `I don't have any roles tracked for ${coRef} at the moment.`;
      }

      // Status-based exceptions: skip the share gate for Posted and Closed roles.
      // Posted → share existence at the user's tier level.
      // Closed → share all information regardless of tier.
      if (state.roleRecordId) {
        const roleRec = await this.db.findRoleById(state.roleRecordId);
        const roleStatus = this._field((roleRec || {}).fields, 'Status').toLowerCase();
        if (roleStatus.includes('closed')) {
          const coRec = state.companyRecordId ? await this.db.getCompany(state.companyRecordId) : null;
          state.phase = Phase.ROLE_FOUND;
          return await this._generateRoleSynopsis(coRec, roleRec, 'premium', state);
        }
        if (roleStatus.includes('posted')) {
          const coRec = state.companyRecordId ? await this.db.getCompany(state.companyRecordId) : null;
          state.phase = Phase.ROLE_FOUND;
          return await this._generateRoleSynopsis(coRec, roleRec, mode, state);
        }
      }

      state.phase = Phase.AWAITING_SHARE;
      if (mode === 'free') {
        return (
          `We do have ${entityRef} in our database. ` +
          "**What have you learned about it from your conversations or research — share anything you know and I'll confirm what we have.**"
        );
      } else { // pro
        return (
          `We have ${entityRef} in our database. ` +
          "**Before I share what we know, what have you already heard or learned about it?**"
        );
      }
    }

    // PREMIUM: role-listing intent → show full roles list
    if (rolesListIntent) {
      state.phase = Phase.COMPANY_FOUND;
      return await this._listCompanyRoles(state);
    }

    // PREMIUM: role matched → show synopsis
    if (state.roleRecordId) {
      const roleRec = await this.db.findRoleById(state.roleRecordId);
      const coRec = state.companyRecordId ? await this.db.getCompany(state.companyRecordId) : null;
      state.phase = Phase.ROLE_FOUND;
      return await this._generateRoleSynopsis(coRec, roleRec, 'premium', state);
    }

    const coRec = await this.db.getCompany(state.companyRecordId);
    state.phase = Phase.COMPANY_FOUND;
    return await this._generateCompanySynopsis(coRec, 'premium', state);
  }

  async _handleAwaitingShare(state, userText) {
    const mode = state.mode;
    const entity = state.roleTitle || state.companyName || 'this';

    await this._extractAndAccumulate(state, userText);

    let synopsis;
    if (state.roleRecordId) {
      const roleRec = await this.db.findRoleById(state.roleRecordId);
      const coRec = state.companyRecordId ? await this.db.getCompany(state.companyRecordId) : null;

      if (mode === 'free') {
        state.phase = Phase.ROLE_FOUND;
        state.insightFollowupsAsked = 1; // This response contains the one followup we ask
        const ackPrompt = (
          `The user shared this about ${entity}: "${userText}"\n\n` +
          '1. Warmly acknowledge their contribution in 1 sentence — be specific about what they shared, not generic.\n' +
          '2. Confirm that we do have this role in our database, but do NOT reveal any details we have.\n' +
          '3. Ask ONE follow-up question that builds naturally on what they shared — clarify something they mentioned or ask if they have any other information. Do NOT jump to a new unrelated topic.\n' +
          '4. In a final sentence, mention they can upgrade to Pro to see what we know.\n' +
          'Bold only the question sentence using **double asterisks**. No other markdown.'
        );
        return await this._callClaude([{ role: 'user', content: ackPrompt }]);
      }

      synopsis = roleRec ? await this._generateRoleSynopsis(coRec, roleRec, mode, state) : '';
      state.phase = Phase.ROLE_FOUND;
    } else {
      const coRec = await this.db.getCompany(state.companyRecordId);

      if (mode === 'free') {
        state.phase = Phase.COMPANY_FOUND;
        state.insightFollowupsAsked = 1; // This response contains the one followup we ask
        const ackPrompt = (
          `The user shared this about ${entity}: "${userText}"\n\n` +
          '1. Warmly thank them for the insight in 1 sentence — be specific about what they shared.\n' +
          '2. Ask ONE follow-up question that builds naturally on what they shared — clarify something they mentioned or ask if they have any other information. Do NOT jump to a completely unrelated topic.\n' +
          'Bold only the question sentence using **double asterisks**. No other markdown.'
        );
        return await this._callClaude([{ role: 'user', content: ackPrompt }]);
      }

      synopsis = coRec ? await this._generateCompanySynopsis(coRec, mode, state) : '';
      state.phase = Phase.COMPANY_FOUND;
    }

    // The synopsis already contains an embedded question (Q2), so mark one followup spent.
    state.insightFollowupsAsked = 1;
    const ackPrompt = (
      `The user just shared this about ${entity}: "${userText}"\n\n` +
      'Acknowledge what they shared in 1 sentence (be specific and appreciative), ' +
      'then transition naturally into the synopsis below. ' +
      'Bold only the question sentence using **double asterisks**. No other markdown.\n\n' +
      `Synopsis:\n${synopsis}`
    );
    return await this._callClaude([{ role: 'user', content: ackPrompt }]);
  }

  _isRoleInfoRequest(userText) {
    const lower = userText.toLowerCase();
    const refPhrases = [
      'that role', 'that one', 'that position', 'that job', 'that opening',
      'tell me more', 'more about', 'more info', 'more detail',
      'what about it', 'and the role', 'about the role',
    ];
    if (refPhrases.some(p => lower.includes(p))) return true;

    const tokens = lower.split(/\s+/);
    const stopWords = new Set([
      'a', 'an', 'of', 'on', 'in', 'for', 'me', 'us', 'more', 'some', 'any',
      ...PRONOUN_ROLE_TOKENS,
    ]);
    const nonStop = tokens.filter(t => !stopWords.has(t));
    return tokens.length <= 6 && nonStop.length === 0;
  }

  _identifyRoleFromContext(state, roles) {
    for (let i = state.messages.length - 1; i >= 0; i--) {
      const msg = state.messages[i];
      if (msg.role !== 'assistant') continue;
      const content = msg.content.toLowerCase();
      for (const role of roles) {
        const title = this._field(role.fields || {}, 'Title').toLowerCase();
        if (title && content.includes(title)) return role;
      }
    }
    return null;
  }

  async _isContinuationReply(state, userText) {
    let lastAgent = '';
    for (let i = state.messages.length - 1; i >= 0; i--) {
      if (state.messages[i].role === 'assistant') {
        lastAgent = state.messages[i].content.substring(0, 400);
        break;
      }
    }
    if (!lastAgent) return false;

    const prompt = (
      `The assistant just said: "${lastAgent}"\n` +
      `The user replied: "${userText}"\n\n` +
      "Is the user directly answering/continuing the assistant's question, " +
      'or are they asking about a completely new company or role?\n' +
      'Reply with exactly one word: "reply" or "new_request".'
    );
    try {
      const raw = await this._callClaude([{ role: 'user', content: prompt }], { maxTokens: 5 });
      return raw.trim().toLowerCase().startsWith('reply');
    } catch (e) {
      return false;
    }
  }

  async _handleFollowup(state, userText) {
    // Role selection by number or name: user replied with a number or partial role name after a roles listing.
    if (state.companyRecordId && !state.roleRecordId) {
      const roles = this._rolesForTier(
        await this.db.getCompanyRoles(state.companyRecordId), state.mode
      );
      let picked = null;
      const numMatch = userText.trim().match(/^#?(\d+)$/);
      if (numMatch) {
        const idx = parseInt(numMatch[1], 10) - 1;
        if (idx >= 0 && idx < roles.length) picked = roles[idx];
      } else if (roles.length > 0) {
        const needle = userText.trim().toLowerCase();
        picked = roles.find(r => {
          const title = (this._field((r.fields || {}), 'Title') || '').toLowerCase();
          return title.includes(needle);
        }) || null;
      }
      if (picked) {
        state.roleRecordId = picked.id;
        state.roleTitle = this._field((picked.fields || {}), 'Title');
        state.roleAppPage = this._field((picked.fields || {}), 'App Page').trim();
        state.phase = Phase.ROLE_FOUND;
        const coRec = await this.db.getCompany(state.companyRecordId);
        return await this._generateRoleSynopsis(coRec, picked, state.mode, state);
      }
    }

    // Role listing intent must be checked first — it takes priority over continuation detection.
    // "are there any other roles at maintainx" is still a continuation but needs a listing response.
    if (state.companyRecordId && this._isRolesListIntent(userText)) {
      // If the message mentions a different company (e.g. "what roles do we have at Gong"
      // while currently on Cursor), switch context first so we look up the right company.
      const parsedForSwitch = await this._parseCompanyAndRole(userText);
      const mentionedCo = (parsedForSwitch && parsedForSwitch.company || '').toLowerCase().trim();
      const currentCo = (state.companyName || '').toLowerCase().trim();
      const isDifferentCompany = mentionedCo &&
        mentionedCo !== currentCo &&
        !currentCo.includes(mentionedCo) &&
        !mentionedCo.includes(currentCo);
      if (isDifferentCompany) {
        // Reset and re-dispatch — _handleIdentify → _dispatchAfterMatch will
        // look up the new company and re-evaluate rolesListIntent there.
        state.companyRecordId = null;
        state.companyName = null;
        state.roleRecordId = null;
        state.roleTitle = null;
        state.phase = Phase.IDENTIFY;
        return await this._handleIdentify(state, userText);
      }

      if (state.mode === 'premium') {
        return await this._listCompanyRoles(state);
      }
      if (state.mode === 'free') {
        return await this._handleFreeRolesListIntent(state);
      }
      // pro
      const roles = this._rolesForTier(
        await this.db.getCompanyRoles(state.companyRecordId), 'pro'
      );
      const count = roles.length;
      const noun = count === 1 ? 'role' : 'roles';
      const coRef = this._companyRef(state);
      if (!count) return `I don't have any roles tracked for ${coRef} at the moment.`;
      return (
        `We do have ${count} ${noun} tracked for ${coRef}. ` +
        '**Upgrade to Premium to see the full breakdown — ' +
        "or is there a specific role you've already come across?**"
      );
    }

    // Detect "I have a new/another role" intent — respond warmly and set a flag so the
    // next message (the role title) skips DB lookup and goes straight to collection.
    if (state.companyRecordId && !state.roleRecordId && this._isNewRoleSignal(userText)) {
      state.pendingNewRoleSignal = true;
      const coRef = this._companyRef(state);
      return (
        `That's great — sounds like you know about a role at ${coRef} that we're not tracking yet. ` +
        `**What's the role title?**`
      );
    }

    // When pendingNewRoleSignal is set, handle the next message carefully:
    // the user may provide a role title, a company correction, or both.
    if (state.pendingNewRoleSignal && !this._isRolesListIntent(userText)) {
      const parsed = await this._parseCompanyAndRole(userText);

      // Company correction (with or without role title) — update state first
      if (parsed.company && parsed.company.toLowerCase() !== (state.companyName || '').toLowerCase()) {
        await this._updateStateCompany(state, parsed.company);
      }

      if (parsed.role && parsed.role.length >= 3) {
        state.pendingNewRoleSignal = false;
        return this._startNewEntityCollection(state, state.companyName, parsed.role, !!state.companyRecordId);
      }

      if (parsed.company) {
        // Company correction with no role title yet — re-ask for the role
        const coRef = this._companyRef(state);
        return `Got it — **${coRef}**. **What's the role title?**`;
      }
    }

    // When in company-found mode with no active role, a role name always takes priority
    // so that "SDR leader" (answering "What's the role?") routes to collection rather than gap-fill.
    if (state.companyRecordId && !state.roleRecordId && !this._isRolesListIntent(userText)) {
      const parsed = await this._parseCompanyAndRole(userText);
      if (parsed.role && parsed.role.length >= 3) {
        state.phase = Phase.IDENTIFY;
        return this._handleIdentify(state, userText);
      }
    }

    // Gate general comp benchmark questions for non-premium members.
    if (state.mode !== 'premium' && this._isCompBenchmarkIntent(userText)) {
      return (
        'We don\'t share general compensation benchmarks — that\'s available to **Premium members**. ' +
        'If you\'re asking about comp data we have on file for this specific role, I\'ll include it when it\'s available. ' +
        '**Upgrade to Premium to access comp benchmarks and deeper hiring insights.**'
      );
    }

    if (await this._isContinuationReply(state, userText)) {
      const clarification = await this._attributionClarification(state, userText);
      if (clarification) return clarification;
      await this._extractAndAccumulate(state, userText);
      // After the user has answered our followup question, wrap up instead of peppering
      // them with more questions. The synopsis already asked Q1; we ask one more (Q2).
      if (state.insightFollowupsAsked >= 1) {
        return this._buildInsightWrapUp(state);
      }
      state.insightFollowupsAsked++;
      return await this._callClaude(state.messages, { system: await this._buildFollowupSystem(state) });
    }

    const parsed = await this._parseCompanyAndRole(userText);
    const newCompany = parsed.company;
    const newRole = parsed.role;
    const currentCompany = (state.companyName || '').toLowerCase();
    const currentRole = (state.roleTitle || '').toLowerCase();

    const MIN_LEN = 4;
    const switchingCompany = (
      newCompany &&
      newCompany.length >= MIN_LEN &&
      newCompany.toLowerCase() !== currentCompany &&
      !newCompany.toLowerCase().includes(currentCompany)
    );
    const sameCompanyMentioned = (
      newCompany &&
      (newCompany.toLowerCase() === currentCompany || currentCompany.includes(newCompany.toLowerCase()))
    );
    const switchingRole = (
      newRole &&
      newRole.length >= MIN_LEN &&
      !switchingCompany &&
      (!newCompany || sameCompanyMentioned) &&
      newRole.toLowerCase() !== currentRole
    );

    if (switchingCompany) {
      state.phase = Phase.IDENTIFY;
      state.companyRecordId = null;
      state.companyName = null;
      state.roleRecordId = null;
      state.roleTitle = null;
      state.suggestedUpdates = {};
      return await this._handleIdentify(state, userText);
    }

    if (switchingRole) {
      state.phase = Phase.IDENTIFY;
      state.roleRecordId = null;
      state.roleTitle = null;
      return await this._handleIdentify(state, userText);
    }

    // "Tell me more about that role" — pronominal reference
    if (!state.roleRecordId && state.companyRecordId && this._isRoleInfoRequest(userText)) {
      const companyRoles = this._rolesForTier(
        await this.db.getCompanyRoles(state.companyRecordId), state.mode
      );
      let matched = null;
      if (companyRoles.length === 1) {
        matched = companyRoles[0];
      } else if (companyRoles.length > 1) {
        matched = this._identifyRoleFromContext(state, companyRoles);
      }
      if (matched) {
        state.roleRecordId = matched.id;
        state.roleTitle = this._field(matched.fields || {}, 'Title');
        state.phase = Phase.ROLE_FOUND;
        const coRec = await this.db.getCompany(state.companyRecordId);
        return await this._generateRoleSynopsis(coRec, matched, state.mode, state);
      }
    }

    const clarification = await this._attributionClarification(state, userText);
    if (clarification) return clarification;
    await this._extractAndAccumulate(state, userText);
    if (state.insightFollowupsAsked >= 1) {
      return this._buildInsightWrapUp(state);
    }
    state.insightFollowupsAsked++;
    return await this._callClaude(state.messages, { system: await this._buildFollowupSystem(state) });
  }

  /**
   * Before extracting intel, check whether the message clearly refers to the
   * current role/company in focus or might be about a different entity.
   * Returns a clarifying question string if attribution is ambiguous, or null
   * if the message is clearly about the current focus.
   */
  async _attributionClarification(state, userText) {
    if (!state.roleTitle && !state.companyName) return null;

    const focus = state.roleTitle
      ? `"${state.roleTitle}" at ${state.companyName}`
      : state.companyName;

    // Include the last few conversation turns so Claude has established context
    const recentHistory = (state.messages || []).slice(-6);
    const historyText = recentHistory
      .map(m => `${m.role === 'user' ? 'User' : 'Agent'}: ${m.content}`)
      .join('\n');

    const prompt = (
      `The conversation so far has been about ${focus}.\n\n` +
      `Recent conversation:\n${historyText}\n\n` +
      `Latest user message: "${userText}"\n\n` +
      'Is this message clearly a continuation about the same entity, or does it EXPLICITLY and UNAMBIGUOUSLY reference a completely different role or company?\n' +
      'Default to clear=true unless there is strong explicit evidence (e.g. the user names a different company/role by name) that they are talking about something else.\n' +
      'Reply with JSON only: {"clear": true} or {"clear": false, "entity": "<the different entity they explicitly named>"}'
    );

    try {
      const raw = await this._callClaude([{ role: 'user', content: prompt }], { maxTokens: 80 });
      const cleaned = raw.trim().replace(/^```json\n?/, '').replace(/^```\n?/, '').replace(/\n?```$/, '').trim();
      const result = JSON.parse(cleaned);
      if (!result.clear && result.entity) {
        const currentRef = this._roleRef(state) || this._companyRef(state);
        return (
          `Just to make sure I capture this in the right place — ` +
          `**is that about ${result.entity}, or about ${currentRef}?**`
        );
      }
    } catch (e) {
      // Fail safe: proceed without clarification
    }
    return null;
  }

  async _buildFollowupSystem(state) {
    let roleFields = {};
    let companyFields = {};
    if (state.roleRecordId) {
      const roleRec = await this.db.findRoleById(state.roleRecordId);
      roleFields = (roleRec || {}).fields || {};
    }
    if (state.companyRecordId) {
      const coRec = await this.db.getCompany(state.companyRecordId);
      companyFields = (coRec || {}).fields || {};
    }

    const sugRole = Object.fromEntries(
      Object.entries((state.suggestedUpdates.role || {})).filter(([, v]) => v)
    );
    const sugCompany = Object.fromEntries(
      Object.entries((state.suggestedUpdates.company || {})).filter(([, v]) => v)
    );
    const mergedRole = { ...roleFields, ...sugRole };
    const mergedCompany = { ...companyFields, ...sugCompany };

    const roleGaps = state.roleRecordId ? getRoleGaps(mergedRole) : [];
    const companyGaps = getCompanyGaps(mergedCompany);

    let system;
    const allGapDescs = [...roleGaps, ...companyGaps].map(([, desc]) => desc);
    const gapContext = allGapDescs.length > 0
      ? '\n\nInformation we are still looking to learn (use as background context only, not a questionnaire):\n' +
        allGapDescs.map((d, i) => `${i + 1}. ${d}`).join('\n')
      : '';

    system = (
      SYSTEM_PROMPT +
      gapContext +
      '\n\nRespond in 2-3 sentences. First, briefly acknowledge or build on something specific the user just shared — ' +
      'be genuine and specific, not generic. ' +
      'Then end with ONE broad, friendly follow-up question. ' +
      'If there are information gaps listed above, name 2-3 of them as examples of things we would love to know, ' +
      'and invite the user to share anything they know on any of those areas — do NOT ask a single narrow question about just one gap. ' +
      'If there are no gaps, simply ask whether they have anything else to share about the company or role. ' +
      'Do NOT ask why the user wants the role or anything about their personal background or motivations.'
    );

    // Premium + role: ensure Claude always gives a brief overview of what we know
    if (state.mode === 'premium' && Object.keys(roleFields).length > 0) {
      const keyFacts = {};
      for (const k of ['Title', 'Function', 'HM Name', 'Region', 'Remote', 'Find', 'Notes']) {
        if (mergedRole[k] != null) keyFacts[k] = mergedRole[k];
      }
      system += (
        '\n\nIMPORTANT: You are discussing the role below with a premium member. ' +
        'Always reference what we know about it (briefly) before asking your question.\n' +
        `ROLE DATA: ${JSON.stringify(keyFacts)}`
      );
    }

    return system;
  }

  // ------------------------------------------------------------------
  // Link helpers
  // ------------------------------------------------------------------

  _ensureHttps(url) {
    if (!url) return url;
    return (url.startsWith('http://') || url.startsWith('https://')) ? url : 'https://' + url;
  }

  /** Thank the user and invite them to bring up another company or role. */
  _buildInsightWrapUp(state) {
    const entity = (state.roleTitle ? `${state.roleTitle} at ` : '') + (state.companyName || 'that');
    return (
      `Thanks so much for sharing — this is really helpful for ${entity}! ` +
      `**Are there other companies or roles I can help you with?**`
    );
  }

  /** Look up companyName in the DB and update state; falls back to name-only if not found. */
  async _updateStateCompany(state, companyName) {
    const companyRecord = await this.db.findCompany(companyName);
    if (companyRecord) {
      state.companyRecordId = companyRecord.id;
      state.companyName = this._field(companyRecord.fields, 'Company Name');
      state.companyDomain = this._ensureHttps(this._field(companyRecord.fields, 'Domain').trim());
    } else {
      state.companyRecordId = null;
      state.companyName = companyName;
      state.companyDomain = null;
    }
  }

  _companyRef(state) {
    const name = state.companyName || 'the company';
    const url = state.companyDomain || '';
    return url ? `[${name}](${url})` : name;
  }

  _roleRef(state) {
    const title = state.roleTitle || '';
    const url = state.roleAppPage || '';
    if (!title) return '';
    return url ? `[${title}](${url})` : title;
  }

  // ------------------------------------------------------------------
  // Roles listing (premium only)
  // ------------------------------------------------------------------

  /** Returns true when the user signals they know about a new/unlisted role. */
  _isNewRoleSignal(text) {
    const lower = text.toLowerCase();
    return (
      /\b(new|another|different|also)\b.{0,35}\b(role|position|job|opening)\b/.test(lower) ||
      /\b(role|position|job|opening)\b.{0,25}\b(new|another|not (tracked|listed|in your|in the))\b/.test(lower) ||
      /\bi (have|know (about|of)|heard (about|of)|came across|found|saw)\b.{0,25}\b(role|position)\b/.test(lower) ||
      /\bawar(e|eness) of\b.{0,25}\b(role|position)\b/.test(lower) ||
      /\b(add|submit|contribute|share)\b.{0,25}\b(role|position|job|opening)\b/.test(lower) ||
      /\b(role|position|job|opening)\b.{0,25}\b(to add|to submit|to contribute|to share)\b/.test(lower)
    );
  }

  /**
   * Shared handler for "tell me about the roles" intent on free tier.
   * Posted roles are public — show full details. Unposted roles are gated.
   */
  async _handleFreeRolesListIntent(state) {
    const allRoles = await this.db.getCompanyRoles(state.companyRecordId);
    const publicOpenRoles = allRoles.filter(r =>
      this._roleStatus(r) === 'public' && this._roleIsActive(r)
    );
    const unpostedActiveCount = allRoles.filter(r =>
      this._roleStatus(r) === 'members-only' && this._roleIsActive(r)
    ).length;
    const closedRoles = allRoles.filter(r =>
      /\bclosed\b/.test(this._normalizeStatus(this._field((r.fields || {}), 'Status', '')))
    );
    const coRef = this._companyRef(state);

    if (publicOpenRoles.length > 0) {
      const coRec = await this.db.getCompany(state.companyRecordId);
      const companyName = (coRec && coRec.fields) ? (coRec.fields['Company Name'] || coRef) : coRef;
      const companyUrl = state.companyDomain || '';
      const companyRef = companyUrl ? `[${companyName}](${companyUrl})` : companyName;

      // Build count line: "We have X posted role(s) at Company [and Y unposted roles]"
      const postedNoun = publicOpenRoles.length === 1 ? 'posted role' : 'posted roles';
      let countLine = `We have ${publicOpenRoles.length} ${postedNoun} at ${companyRef}`;
      if (unpostedActiveCount > 0) {
        const unpostedNoun = unpostedActiveCount === 1 ? 'unposted role' : 'unposted roles';
        countLine += ` and ${unpostedActiveCount} ${unpostedNoun}`;
      }
      countLine += ':';

      // Numbered list of public roles
      const roleLines = publicOpenRoles.map((r, i) => {
        const rf = r.fields || {};
        const title = rf.Title || 'Untitled';
        const appPage = (rf['App Page'] || '').trim();
        const titleRef = appPage ? `[${title}](${appPage})` : title;
        const regionArr = rf.Region || [];
        const region = (Array.isArray(regionArr) ? regionArr.join(', ') : regionArr) || '';
        const remoteFlag = rf.Remote ? 'FULLY REMOTE' : '';
        const location = [region, remoteFlag].filter(Boolean).join(' | ');
        const hmPart = rf['HM Name'] ? `HM: ${rf['HM Name']}` : '';
        const details = [hmPart, location].filter(Boolean).join(' | ');
        return `${i + 1}. ${titleRef}${details ? ' — ' + details : ''}`;
      }).join('\n');

      // Closing question — non-presumptuous
      const closingQ = publicOpenRoles.length === 1
        ? '\n\n**Are you interested in this role, or do you have insights on the company?**'
        : '\n\n**Are you interested in one of these roles, or do you have insights on the company?**';

      return `${countLine}\n\n${roleLines}${closingQ}`;
    }

    if (unpostedActiveCount > 0) {
      const noun = unpostedActiveCount === 1 ? 'role' : 'roles';
      return (
        `We do have ${unpostedActiveCount} active ${noun} tracked for ${coRef}. ` +
        'These are roles shared with us in confidence — executives and recruiters trust our community\'s talent bar and discretion with sensitive, unannounced openings, so we only share them with paid members. ' +
        '**Become a paid member to see the role titles and hiring details.**'
      );
    }
    if (closedRoles.length > 0) {
      const noun = closedRoles.length === 1 ? 'role' : 'roles';
      return `We have ${closedRoles.length} previously tracked ${noun} for ${coRef}, but they're all closed at the moment. **Become a paid member to get notified when new roles open up.**`;
    }
    return `I don't have any roles tracked for ${coRef} at the moment.`;
  }

  /** Returns true when a free user is asking to see unposted/confidential/gated roles. */
  _isUnpostedRolesRequest(text) {
    const low = text.toLowerCase();
    return (
      // Explicit unposted/confidential role keywords
      /\b(unposted|whispered|confidential|hidden|private|members.only|gated|restricted|exclusive)\b.{0,30}\broles?\b/.test(low) ||
      /\broles?\b.{0,30}\b(unposted|whispered|confidential|hidden|private|members.only|gated)\b/.test(low) ||
      // "show/see/access/share the other roles / those roles / all roles"
      /\b(show|see|access|view|get|share|reveal|unlock|tell me about).{0,25}\b(those|the other|other|all|remaining|rest of the|the rest).{0,20}\broles?\b/.test(low) ||
      // "what are the other/those roles"
      /\bwhat (are|about) (the other|those other|those|all the|the remaining|the rest of the) roles?\b/.test(low) ||
      // "can I see the X unposted / other roles"
      /\b(can i|could i|how do i|how can i).{0,30}\b(see|access|view|get|unlock).{0,30}\broles?\b/.test(low) ||
      // "upgrade" / "paid" / "pro" mentioned together with roles
      /\b(upgrade|paid|pro|premium).{0,30}\broles?\b/.test(low) ||
      /\broles?.{0,30}\b(upgrade|paid plan|pro plan|premium plan)\b/.test(low)
    );
  }

  /** Returns true when the user is asking about general comp benchmarks (not specific role data). */
  _isCompBenchmarkIntent(text) {
    const low = text.toLowerCase();
    // General benchmark signals
    if (/\b(benchmark|market rate|going rate|industry standard|industry average)\b/.test(low)) return true;
    if (/\btypical (comp|compensation|salary|pay|package|base|equity|bonus)\b/.test(low)) return true;
    if (/\b(average|usual|normally|generally|typically)\b.{0,30}\b(comp|compensation|salary|pay|earn|make)\b/.test(low)) return true;
    if (/\bwhat (do|does|would|should).{0,20}\b(roles?|positions?|directors?|vps?|heads?)\b.{0,20}\b(make|earn|pay|get paid|comp)\b/.test(low)) return true;
    if (/\b(salary|comp|compensation).{0,20}\b(range|ranges|band|bands|expectations?)\b/.test(low)) return true;
    if (/\bcomp.{0,25}\broles? like (this|these)\b/.test(low)) return true;
    if (/\bwhat.{0,20}\bpay(s)? (for|like)\b/.test(low)) return true;
    return false;
  }

  _isRolesListIntent(text) {
    const low = text.toLowerCase();
    const phrases = [
      'what roles', 'which roles', 'list roles', 'any roles', 'open roles',
      'roles do you have', 'roles you have', 'roles we have', 'tell me roles',
      'tell me about the roles', 'tell me the roles', 'tell me about those roles',
      'about those roles', 'about the roles', 'those roles', 'what are those roles',
      'more about the roles', 'more on the roles', 'more about those',
      'roles in our', 'roles in the', 'roles at', 'roles for',
      'what positions', 'any positions', 'open positions',
      'what openings', 'any openings',
      'the roles', 'the positions',
    ];
    if (phrases.some(p => low.includes(p))) return true;
    // Catch verb-led requests: "show me the roles", "see those roles", "give me the positions", etc.
    return /\b(show|see|view|give|get|find|check|learn about|hear about)\b.{0,20}\b(roles?|positions?|openings?)\b/.test(low);
  }

  async _listCompanyRoles(state) {
    const coRec = await this.db.getCompany(state.companyRecordId);
    const roles = await this.db.getCompanyRoles(state.companyRecordId);

    // Resolve the actual company name for each role from its linked record,
    // so a role at PayScale or AnyScale is never labelled as "Scale".
    const companyCache = {};
    for (const role of roles) {
      const linkedIds = (role.fields || {}).Company || [];
      if (linkedIds.length > 0) {
        const cid = linkedIds[0];
        if (!(cid in companyCache)) {
          const c = await this.db.getCompany(cid);
          companyCache[cid] = (c && c.fields) ? (c.fields['Company Name'] || '') : '';
        }
        role.fields._company_name = companyCache[cid];
      }
    }

    const openRoles = roles.filter(
      r => !/\bclosed\b/.test(this._normalizeStatus(this._field(r.fields, 'Status')))
    );
    const closedRoles = roles.filter(
      r => /\bclosed\b/.test(this._normalizeStatus(this._field(r.fields, 'Status')))
    );
    const companyUrl = state.companyDomain || '';
    const prompt = buildRolesListingPrompt(coRec || {}, openRoles, closedRoles, companyUrl);
    return await this._callClaude([{ role: 'user', content: prompt }]);
  }

  // ------------------------------------------------------------------
  // Synopsis generators
  // ------------------------------------------------------------------

  async _generateCompanySynopsis(companyRecord, mode = 'premium', state = null) {
    const allRoles = await this.db.getCompanyRoles(companyRecord.id);
    const roles = this._rolesForTier(allRoles, mode);
    const companyUrl = state ? (state.companyDomain || '') : '';
    // For free synopsis, pass posted role titles + unposted count so the prompt
    // can say "We have 1 posted role (Title) and 3 unposted roles at Company."
    const rolesSummary = mode === 'free' ? {
      postedActive: allRoles.filter(r => this._roleStatus(r) === 'public' && this._roleIsActive(r)),
      unpostedActiveCount: allRoles.filter(r => this._roleStatus(r) === 'members-only' && this._roleIsActive(r)).length,
      closedCount: allRoles.filter(r =>
        /\bclosed\b/.test(this._normalizeStatus(this._field((r.fields || {}), 'Status', '')))
      ).length,
    } : null;
    const prompt = buildCompanySynopsisPrompt(companyRecord, roles, [], mode, companyUrl, rolesSummary);
    return await this._callClaude([{ role: 'user', content: prompt }]);
  }

  async _generateRoleSynopsis(companyRecord, roleRecord, mode = 'premium', state = null) {
    const roleFields = (roleRecord || {}).fields || {};
    const companyFields = (companyRecord || {}).fields || {};
    const roleGaps = getRoleGaps(roleFields);
    const companyGaps = getCompanyGaps(companyFields);
    const allGaps = [...roleGaps, ...companyGaps];
    const topGap = allGaps.length > 0 ? allGaps[0][1] : null;
    const companyUrl = state ? (state.companyDomain || '') : '';
    const roleUrl = state ? (state.roleAppPage || '') : '';
    const prompt = buildRoleSynopsisPrompt(
      roleRecord, companyRecord || {}, [], mode, topGap, companyUrl, roleUrl
    );
    return await this._callClaude([{ role: 'user', content: prompt }]);
  }

  // ------------------------------------------------------------------
  // Data extraction
  // ------------------------------------------------------------------

  async _extractAndAccumulate(state, userText) {
    const roleName = state.roleTitle || '';
    const companyName = state.companyName || '';
    if (!roleName && !companyName) return;

    const prompt = buildDataExtractionPrompt(userText, roleName, companyName);
    const raw = await this._callClaude([{ role: 'user', content: prompt }], { maxTokens: 512 });

    let extracted;
    try {
      const cleaned = raw.trim().replace(/^```json\n?/, '').replace(/^```\n?/, '').replace(/\n?```$/, '').trim();
      extracted = JSON.parse(cleaned);
    } catch (e) {
      console.debug(`data extraction parse failed: ${raw.substring(0, 200)}`);
      return;
    }

    const updates = state.suggestedUpdates;
    if (!updates.role) updates.role = {};
    if (!updates.company) updates.company = {};
    updates.role_name = roleName;
    updates.company_name = companyName;

    // Fetch current Airtable field values so the merge baseline includes stored data
    let airtableRoleFields = {};
    let airtableCompanyFields = {};
    if (state.roleRecordId) {
      const roleRec = await this.db.findRoleById(state.roleRecordId);
      airtableRoleFields = (roleRec || {}).fields || {};
    }
    if (state.companyRecordId) {
      const coRec = await this.db.getCompany(state.companyRecordId);
      airtableCompanyFields = (coRec || {}).fields || {};
    }

    for (const [field, value] of Object.entries(extracted.role || {})) {
      if (value == null) continue;
      if (field === 'Compensation') {
        const num = parseInt(value, 10);
        if (!isNaN(num)) updates.role.Compensation = num;
      } else if (field === 'Notes') {
        const existing = updates.role.Notes || airtableRoleFields.Notes || '';
        updates.role.Notes = await this._structuredMerge('role_notes', existing, value);
      } else if (field === 'Region') {
        const locationText = String(value);
        const existingRegion = String(updates.role.Region || airtableRoleFields.Region || '');
        updates.role.Region = existingRegion
          ? await this._simpleMerge('Region', existingRegion, locationText)
          : locationText;
      } else {
        const existing = String(updates.role[field] || airtableRoleFields[field] || '');
        updates.role[field] = existing
          ? await this._simpleMerge(field, existing, value)
          : value;
      }
    }

    for (const [field, value] of Object.entries(extracted.company || {})) {
      if (!value) continue;
      if (field === 'Confidential Notes') {
        const existing = updates.company['Confidential Notes'] || airtableCompanyFields['Confidential Notes'] || '';
        updates.company['Confidential Notes'] = await this._structuredMerge('company_notes', existing, value);
      } else {
        const existing = String(updates.company[field] || airtableCompanyFields[field] || '');
        updates.company[field] = existing
          ? await this._simpleMerge(field, existing, value)
          : value;
      }
    }
  }

  async _semanticRoleFilter(roleQuery, roles) {
    const summaries = roles.map((r, i) => {
      const rf = r.fields || {};
      const title = rf.Title || 'Untitled';
      const fn = rf.Function || '';
      const notesSnippet = (rf.Notes || '').substring(0, 200);
      let line = `${i}: ${title}`;
      if (fn) line += ` (${fn})`;
      if (notesSnippet) line += ` — ${notesSnippet}`;
      return line;
    });

    const prompt = (
      `A user in a GTM/RevOps professional community is asking about: "${roleQuery}"\n\n` +
      `Roles at this company:\n${summaries.join('\n')}\n\n` +
      'Remove only roles that are CLEARLY in a completely different function ' +
      '(e.g. a Finance, HR, or pure Engineering role when the query is about RevOps/GTM). ' +
      'Keep everything that could plausibly be what the user means — including ' +
      'adjacent roles like GTM AI, Revenue Intelligence, Sales/Marketing Ops, ' +
      'GTM Strategy, or Growth. When in doubt, KEEP the role. ' +
      'Return a JSON array of the indices to KEEP. ' +
      'Return [] only if nothing is remotely related. Valid JSON only — no preamble.'
    );

    try {
      const raw = await this._callClaude([{ role: 'user', content: prompt }], { maxTokens: 32 });
      const cleaned = raw.trim().replace(/^```json\n?/, '').replace(/^```\n?/, '').replace(/\n?```$/, '').trim();
      const result = JSON.parse(cleaned);
      if (Array.isArray(result)) {
        const kept = result
          .filter(i => Number.isInteger(i) && i >= 0 && i < roles.length)
          .map(i => roles[i]);
        console.info(`_semanticRoleFilter: '${roleQuery}' → kept ${JSON.stringify(kept.map(r => r.fields.Title))}`);
        return kept;
      }
    } catch (e) {
      console.debug(`_semanticRoleFilter failed for '${roleQuery}'`);
    }
    return roles; // on failure, return all roles (safest)
  }

  async _roleNotFoundAtCompany(state, companyRecord, roleQuery) {
    state.companyRecordId = companyRecord.id;
    state.companyName = this._field(companyRecord.fields, 'Company Name');
    const rawDomain = this._field(companyRecord.fields, 'Domain');
    state.companyDomain = this._ensureHttps(rawDomain.trim());

    // Fetch existing roles visible to this tier so we can tell the user what we DO have
    const existingRoles = this._rolesForTier(
      await this.db.getCompanyRoles(companyRecord.id),
      state.mode
    );
    const coRef = this._companyRef(state);

    // Focus the summary on open roles only — closed roles accumulate over time and
    // would overwhelm the message. We still surface them if explicitly asked.
    const openRoles = existingRoles.filter(
      r => !/\bclosed\b/.test(this._normalizeStatus(this._field(r.fields, 'Status')))
    );
    const openCount = openRoles.length;

    let roleSummary;
    if (openCount === 0) {
      roleSummary = "we're not tracking any open roles there yet";
    } else if (openCount === 1) {
      const title = this._field(openRoles[0].fields, 'Title');
      roleSummary = `we're tracking 1 open role (${title})`;
    } else if (openCount <= 3) {
      const titles = openRoles.map(r => this._field(r.fields, 'Title')).join(', ');
      roleSummary = `we're tracking ${openCount} open roles (${titles})`;
    } else {
      roleSummary = `we're tracking ${openCount} open roles`;
    }

    // Set up collection state (hasExistingCompany=true so domain isn't re-asked)
    state.pendingNewEntity = {
      companyName: state.companyName,
      roleTitle: roleQuery,
      domain: null,
      find: null,
      notes: null,
      location: null,
      compensation: null,
      step: 'confirm',
      hasExistingCompany: true,
    };
    state.phase = Phase.COLLECTING_NEW_ENTITY;

    return (
      `We have **${state.companyName}** in our database — ${roleSummary} but no **${roleQuery}** role. ` +
      `Do you want to add the **${roleQuery}** role at ${coRef}?`
    );
  }

  // ------------------------------------------------------------------
  // New entity collection (company/role not in DB)
  // ------------------------------------------------------------------

  /**
   * Initialise the COLLECTING_NEW_ENTITY phase and return the opening confirmation message.
   * @param {object} state
   * @param {string|null} companyName
   * @param {string|null} roleTitle
   * @param {boolean} hasExistingCompany  True when the company IS in the DB but the role is not
   */
  _startNewEntityCollection(state, companyName, roleTitle, hasExistingCompany) {
    state.pendingNewEntity = {
      companyName: companyName || null,
      roleTitle: roleTitle || null,
      domain: null,
      find: null,
      notes: null,
      location: null,
      compensation: null,
      step: 'confirm',
      hasExistingCompany,
    };
    state.phase = Phase.COLLECTING_NEW_ENTITY;

    if (companyName && roleTitle) {
      return (
        `We don't have **${companyName}** or a **${roleTitle}** role in our database yet. ` +
        `**Just to confirm — are you talking about the ${roleTitle} role at ${companyName}?**`
      );
    } else if (companyName) {
      return (
        `We don't have **${companyName}** in our database yet. ` +
        `**Just to confirm — is that the company you're referring to?**`
      );
    } else {
      return (
        `We don't have **${roleTitle}** in our database yet. ` +
        `**Just to confirm — is that the role you're asking about, and which company is it at?**`
      );
    }
  }

  async _handleCollectingNewEntity(state, userText) {
    const pe = state.pendingNewEntity;
    if (!pe) {
      state.phase = Phase.IDENTIFY;
      return this._handleIdentify(state, userText);
    }

    const lower = userText.toLowerCase().trim();
    const isYes = /\b(yes|yeah|yep|correct|right|sure|exactly|yup|affirmative|that'?s?\s*(right|it|correct)?)\b/.test(lower);
    const isNo = /\b(no|nope|not|wrong|different|other|never mind|nevermind)\b/.test(lower);

    // ── Step: confirm ────────────────────────────────────────────────
    if (pe.step === 'confirm') {
      if (isNo) {
        state.pendingNewEntity = null;
        state.phase = Phase.IDENTIFY;
        return `No problem — which company or role were you looking for?`;
      }

      // If we're missing the company name, the user may have provided it inline
      if (!pe.companyName && !isYes) {
        const parsed = await this._parseCompanyAndRole(userText);
        if (parsed.company) pe.companyName = parsed.company;
        if (parsed.role) pe.roleTitle = parsed.role;
      }

      if (!pe.companyName) {
        pe.step = 'get_company';
        return `**Which company is this role at?**`;
      }

      // Try to auto-lookup the domain from Claude's training knowledge
      if (!pe.hasExistingCompany) {
        pe.domain = await this._lookupDomain(pe.companyName);
      }

      // Company already in DB — skip domain, go straight to role info
      if (pe.hasExistingCompany) {
        if (!pe.roleTitle) {
          pe.step = 'get_role';
          return `**What's the title of the role you're asking about?**`;
        }
        pe.step = 'find_and_notes';
        return (
          `Got it — **${pe.roleTitle}** at **${pe.companyName}**. We don't have that role yet. ` +
          `**What do you know about it — how would someone find or apply, and what's the role like?**`
        );
      }

      // Domain found automatically — skip asking for it
      if (pe.domain) {
        if (pe.roleTitle) {
          pe.step = 'find_and_notes';
          return (
            `Got it — **${pe.roleTitle}** at **${pe.companyName}**. We'll add both. ` +
            `**What do you know about this role — how would someone find or apply, and what's it like?**`
          );
        }
        // Company-only and domain resolved — save now
        return this._saveNewEntity(state);
      }

      // Domain unknown — ask the user
      pe.step = pe.roleTitle ? 'domain_then_role' : 'domain_only';
      const companyRef = pe.companyName;
      return `Got it — **${companyRef}**. **What's their website or domain?**`;
    }

    // ── Step: get_company ─────────────────────────────────────────────
    if (pe.step === 'get_company') {
      const parsed = await this._parseCompanyAndRole(userText);
      pe.companyName = parsed.company || userText.trim();
      pe.domain = await this._lookupDomain(pe.companyName);
      if (pe.domain && pe.roleTitle) {
        pe.step = 'find_and_notes';
        return `**What do you know about the ${pe.roleTitle} role — how would someone find or apply, and what's it like?**`;
      }
      pe.step = pe.roleTitle ? 'domain_then_role' : 'domain_only';
      return `**What's ${pe.companyName}'s website or domain?**`;
    }

    // ── Step: get_role ────────────────────────────────────────────────
    if (pe.step === 'get_role') {
      const parsed = await this._parseCompanyAndRole(userText);
      pe.roleTitle = parsed.role || userText.trim();
      pe.step = 'find_and_notes';
      return `**What do you know about the ${pe.roleTitle} role — how would someone find or apply, and what's it like?**`;
    }

    // ── Step: domain (user-provided) ──────────────────────────────────
    if (pe.step === 'domain_only' || pe.step === 'domain_then_role') {
      const raw = userText.trim().replace(/^https?:\/\//, '').split('/')[0].trim();
      pe.domain = raw || userText.trim();

      if (pe.step === 'domain_then_role' && pe.roleTitle) {
        pe.step = 'find_and_notes';
        return `**What do you know about the ${pe.roleTitle} role — how would someone find or apply, and what's it like?**`;
      }
      // Company-only — save now
      return this._saveNewEntity(state);
    }

    // ── Step: find_and_notes ──────────────────────────────────────────
    if (pe.step === 'find_and_notes') {
      const extracted = await this._extractRoleFields(userText, pe.roleTitle);
      pe.find = extracted.find || null;
      pe.notes = extracted.notes || null;
      pe.location = extracted.location || null;
      pe.compensation = extracted.compensation ?? null;
      pe.step = 'follow_up';
      return await this._generateNewRoleFollowUp(pe);
    }

    // ── Step: follow_up (one targeted question, then done) ─────────────
    if (pe.step === 'follow_up') {
      const extracted = await this._extractRoleFields(userText, pe.roleTitle);
      if (extracted.find) pe.find = pe.find ? `${pe.find}; ${extracted.find}` : extracted.find;
      if (extracted.notes) pe.notes = pe.notes ? `${pe.notes}\n${extracted.notes}` : extracted.notes;
      if (extracted.location) pe.location = extracted.location;
      if (extracted.compensation != null) pe.compensation = extracted.compensation;
      return this._saveNewEntity(state);
    }

    // Fallback
    state.phase = Phase.IDENTIFY;
    return this._handleIdentify(state, userText);
  }

  /**
   * Try to resolve a company's domain using Claude's training knowledge.
   * Returns the bare domain string (e.g. 'acme.com') or null if unknown.
   * @param {string} companyName
   * @returns {Promise<string|null>}
   */
  async _lookupDomain(companyName) {
    const prompt = (
      `What is the primary website domain for the company "${companyName}"?\n` +
      'Reply with ONLY the bare domain (e.g. "acme.com") with no protocol or path. ' +
      'If you are not confident, reply with exactly: null'
    );
    try {
      const raw = await this._callClaude(
        [{ role: 'user', content: prompt }],
        { maxTokens: 32, system: 'You are a factual assistant. Return only what is asked, nothing else.' }
      );
      const domain = raw.trim().toLowerCase().replace(/^https?:\/\//, '').split('/')[0].trim();
      if (domain && domain !== 'null' && domain.includes('.')) return domain;
    } catch (e) {
      console.debug(`_lookupDomain failed for '${companyName}'`);
    }
    return null;
  }

  /**
   * Generate one targeted follow-up question for a new role, phrased generally.
   * Priority: Location → Scope/team size → Criteria (reason for hire) → Details (HM)
   * Never ask about compensation.
   * @param {object} pe  pendingNewEntity
   * @returns {Promise<string>}
   */
  async _generateNewRoleFollowUp(pe) {
    const known = [
      pe.find         ? `How to find/apply (Role - Find): ${pe.find}` : null,
      pe.location     ? `Location (Role - Location): ${pe.location}` : null,
      pe.compensation != null ? `Compensation (Role - Compensation): $${pe.compensation}` : null,
      pe.notes        ? `Notes so far (Role - Notes):\n${pe.notes}` : null,
    ].filter(Boolean).join('\n');

    const prompt = (
      `We're capturing a new role: "${pe.roleTitle}" at "${pe.companyName}".\n` +
      (known ? `What we know so far:\n${known}\n\n` : '\n') +
      'Write ONE warm follow-up question to gather the most important missing detail.\n' +
      'Priority (check what is NOT yet covered above and pick the top gap):\n' +
      '1. Location / remote setup (Role - Location)\n' +
      '2. Compensation — OTE total cash + bonus in USD (Role - Compensation)\n' +
      '3. Scope — responsibilities, team size, who the role reports to\n' +
      '4. Criteria — key skills, reason for hire, interview panel\n' +
      '5. Details — hiring manager or who it reports to\n' +
      'Ask generally — hint at a few possibilities rather than drilling on one thing ' +
      '(e.g. "Do you have any more details — like the location, comp range, or team size?").\n' +
      'Max 2 sentences. Bold only the question with **double asterisks**. No other markdown.'
    );
    try {
      return await this._callClaude([{ role: 'user', content: prompt }], { maxTokens: 128 });
    } catch (e) {
      return `**Do you have any more details on the role — like the location, comp range, team size, or what they're looking for in a candidate?**`;
    }
  }

  /**
   * Use Claude to extract all role fields from a free-form user response.
   * Maps to Airtable fields: Role - Find, Role - Notes, Role - Location, Role - Compensation.
   * @param {string} userText
   * @param {string} roleTitle
   * @returns {Promise<{find: string|null, notes: string|null, location: string|null, compensation: number|null}>}
   */
  async _extractRoleFields(userText, roleTitle) {
    const prompt = (
      `The user was asked what they know about the "${roleTitle}" role — ` +
      'including how to find/apply and what the role is like.\n\n' +
      `They replied: "${userText}"\n\n` +
      'Extract the following fields:\n' +
      '- "find" (Role - Find): how to find or apply — who at the company is leading the search, recruiter name, referral contact, LinkedIn URL, or job posting URL. null if not mentioned.\n' +
      '- "notes" (Role - Notes): structured role details as plain text using these section labels where info is available:\n' +
      '    Scope: responsibilities, team size, who the role reports to\n' +
      '    Criteria: key skills, interview panel, reason for hire\n' +
      '    Details: hiring manager, reporting structure\n' +
      '  Omit any section with no information. null if nothing to note.\n' +
      '- "location" (Role - Location): office location and hybrid/remote setup. null if not mentioned.\n' +
      '- "compensation" (Role - Compensation): total OTE cash + bonus in USD as a plain number (e.g. 250000). null if not mentioned.\n' +
      'Return JSON only: {"find": "...", "notes": "...", "location": "...", "compensation": number|null}'
    );
    try {
      const raw = await this._callClaude(
        [{ role: 'user', content: prompt }],
        { maxTokens: 400, system: 'You are a concise data extractor. Return valid JSON only.' }
      );
      const cleaned = raw.trim().replace(/^```json\n?/, '').replace(/^```\n?/, '').replace(/\n?```$/, '').trim();
      return JSON.parse(cleaned);
    } catch (e) {
      console.debug(`_extractRoleFields parse failed for '${roleTitle}'`);
      return { find: null, notes: userText.trim(), location: null, compensation: null };
    }
  }

  _saveNewEntity(state) {
    const pe = state.pendingNewEntity;
    if (!pe) {
      state.phase = Phase.IDENTIFY;
      return `Something went wrong — what company or role would you like to explore?`;
    }

    // Update conversation state
    state.companyName = pe.companyName || state.companyName;
    state.companyDomain = pe.domain ? this._ensureHttps(pe.domain) : state.companyDomain;
    if (pe.roleTitle) state.roleTitle = pe.roleTitle;

    // Push collected data to suggestedUpdates under distinct new_role / new_company keys
    // so the UI panel can render them as "additions" rather than updates to existing records.
    if (!state.suggestedUpdates) state.suggestedUpdates = {};

    if (pe.companyName && !pe.hasExistingCompany) {
      state.suggestedUpdates.new_company = {
        'Company Name': pe.companyName,
        ...(pe.domain ? { 'Domain Dirty': pe.domain } : {}),
      };
    }

    if (pe.roleTitle) {
      state.suggestedUpdates.new_role = {
        Title: pe.roleTitle,
        Company: pe.companyName,
        ...(pe.find         ? { 'Role - Find':         pe.find         } : {}),
        ...(pe.notes        ? { 'Role - Notes':        pe.notes        } : {}),
        ...(pe.location     ? { 'Role - Location':     pe.location     } : {}),
        ...(pe.compensation != null ? { 'Role - Compensation': pe.compensation } : {}),
      };
    }

    state.pendingNewEntity = null;

    // Transition to COMPANY_FOUND so the normal gap-filling loop doesn't restart.
    state.phase = Phase.COMPANY_FOUND;
    const companyRef = this._companyRef(state);

    if (pe.roleTitle) {
      return (
        `Captured — **${pe.roleTitle}** at **${pe.companyName}** has been queued for our database. ` +
        'Thanks for the contribution!'
      );
    }

    return (
      `Captured — **${pe.companyName}** has been queued for our database. ` +
      `**Is there a specific role at ${companyRef} you're exploring?**`
    );
  }

  async _simpleMerge(fieldName, existing, newInfo) {
    if (!existing) return newInfo;
    const prompt = buildSimpleFieldMergePrompt(fieldName, existing, newInfo);
    try {
      const result = await this._callClaude(
        [{ role: 'user', content: prompt }],
        { maxTokens: 256, system: 'You are a concise database editor. Follow instructions exactly.' }
      );
      return result.trim();
    } catch (e) {
      console.debug(`_simpleMerge failed for field '${fieldName}'; falling back to concat`);
      return (existing + '; ' + newInfo).trim();
    }
  }

  async _structuredMerge(schemaType, existing, newInfo) {
    const prompt = buildStructuredMergePrompt(schemaType, existing, newInfo);
    try {
      const result = await this._callClaude(
        [{ role: 'user', content: prompt }],
        { maxTokens: 512, system: 'You are a concise database editor. Follow instructions exactly.' }
      );
      return result.trim();
    } catch (e) {
      console.debug('_structuredMerge failed; falling back to concat');
      return existing ? (existing + '\n' + newInfo).trim() : newInfo;
    }
  }

  // ------------------------------------------------------------------
  // Claude helpers
  // ------------------------------------------------------------------

  /**
   * Classify a role's Status field into one of three visibility tiers:
   *   'public'       — Posted (any variant) or Closed: share name with all tiers
   *   'members-only' — Whispered Role™, Unposted-Rumor/Recruiter/Company/Future: Pro/Premium only
   *   'confidential' — Unposted-Confidential: never share with anyone
   */
  /** Strip emoji/non-ASCII so status values like "🟥 Posted" compare as "posted". */
  _normalizeStatus(raw) {
    const s = (Array.isArray(raw) ? raw.join(',') : String(raw));
    // Remove non-ASCII characters (emoji, special symbols) then normalise whitespace
    return s.replace(/[^\x00-\x7F]+/g, '').trim().toLowerCase();
  }

  _roleStatus(role) {
    const raw = this._field((role || {}).fields, 'Status', '');
    const s = this._normalizeStatus(raw);
    if (s.includes('confidential')) return 'confidential';
    // \bposted\b matches "Posted" but NOT "Unposted"
    if (/\bclosed\b/.test(s)) return 'public';
    if (/\bposted\b/.test(s)) return 'public';
    // Whispered Role™, Unposted-Rumor, Unposted-Recruiter, Unposted-Company, Unposted-Future
    return 'members-only';
  }

  /** Returns true if the role is active (not closed, not confidential). */
  _roleIsActive(role) {
    if (this._roleStatus(role) === 'confidential') return false;
    const raw = this._field((role.fields || {}), 'Status', '');
    return !/\bclosed\b/.test(this._normalizeStatus(raw));
  }

  /** Returns true if a role should be surfaced to a user at the given tier. */
  _roleVisibleToTier(role, mode) {
    const s = this._roleStatus(role);
    if (s === 'confidential') return false;
    if (s === 'members-only') return mode === 'pro' || mode === 'premium';
    return true; // public: Posted + Closed visible to all tiers
  }

  /** Filter a roles array to only those visible at the given tier. */
  _rolesForTier(roles, mode) {
    return roles.filter(r => this._roleVisibleToTier(r, mode));
  }

  _isConfidential(role) {
    return this._roleStatus(role) === 'confidential';
  }

  _field(recordFields, key, defaultVal = '') {
    if (!recordFields || !(key in recordFields)) return defaultVal;
    const val = recordFields[key];
    if (Array.isArray(val)) return val.length > 0 ? String(val[0]) : defaultVal;
    return (val != null && typeof val === 'string') ? val : defaultVal;
  }

  async _parseCompanyAndRole(userText) {
    const prompt = (
      'Extract the company name and job role title from this message.\n' +
      'Rules:\n' +
      "- 'company': extract any business, organisation, or product name, even if " +
      "lowercase or abbreviated (e.g. 'maintainx', 'openai', '11x', 'acme corp'). " +
      'Return null only if no company is mentioned.\n' +
      "- 'role': extract only a specific job title (e.g. 'VP of Sales', 'Head of GTM AI'). " +
      "Do NOT extract generic words like 'role', 'roles', 'job', 'position', or " +
      "pronoun/reference phrases like 'that role', 'that one', 'this role', 'it', 'that position', " +
      "'tell me more about that', 'the role'. Return null if no real job title is explicitly stated.\n" +
      "Return a JSON object with keys 'company' and 'role'.\n\n" +
      `Message: ${userText}\n\nJSON:`
    );
    const raw = await this._callClaude([{ role: 'user', content: prompt }]);
    try {
      const cleaned = raw.trim().replace(/^```json\n?/, '').replace(/^```\n?/, '').replace(/\n?```$/, '').trim();
      const data = JSON.parse(cleaned);
      for (const key of ['company', 'role']) {
        if (Array.isArray(data[key])) {
          data[key] = data[key].length > 0 ? data[key][0] : null;
        }
      }
      return data;
    } catch (e) {
      return { company: null, role: null };
    }
  }

  async _callClaude(messages, options = {}) {
    const { maxTokens = 1024, system = null } = options;
    const response = await this.client.messages.create({
      model: CLAUDE_MODEL,
      max_tokens: maxTokens,
      system: system || SYSTEM_PROMPT,
      messages,
    });
    return response.content[0].text;
  }
}

module.exports = InsightsAgent;
