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
        `Hey ${userName}! I'm the Insights agent.\n\n` +
        "Ask me about any company or role you're exploring and I'll share what public information we have. " +
        'Upgrade to Pro to unlock full community insights.'
      );
    } else if (mode === 'pro') {
      greeting = (
        `Hey ${userName}! I'm the Insights agent.\n\n` +
        "Tell me about a company or role you're exploring. " +
        "Share what you've learned and I'll pull up what we have — " +
        'your insights help the whole community.'
      );
    } else { // premium
      greeting = (
        `Hey ${userName}! I'm the Insights agent.\n\n` +
        "Which company or role do you want to know about? " +
        "I'll share what we have and we can compare notes.\n\n" +
        'For example: "Airtable" or "VP of RevOps at Airtable".'
      );
    }

    state.addAssistantMessage(greeting);
    return greeting;
  }

  async handleMessage(userId, userName, userText, mode = 'premium') {
    const state = this.stateManager.getOrCreate(userId, userName, mode);
    if (state.mode !== mode) state.mode = mode;

    state.addUserMessage(userText);

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
      }
      // else: no match — fall through to not-found handling below
    }

    if (roleRecord && !companyRecord) {
      const linked = (roleRecord.fields.Company || []);
      if (linked.length > 0) {
        companyRecord = await this.db.getCompany(linked[0]);
      }
    }

    // For weak/no DB matches run semantic matching.
    if (companyRecord && roleTitle && (matchType === 'notes' || matchType === 'none')) {
      const companyRoles = await this.db.getCompanyRoles(companyRecord.id);
      if (companyRoles.length > 0) {
        if (companyRoles.length === 1) {
          roleRecord = companyRoles[0];
        } else {
          // Direct title substring match first
          const queryLower = roleTitle.toLowerCase();
          const titleHits = companyRoles.filter(r => {
            const title = this._field(r.fields, 'Title').toLowerCase();
            return queryLower.includes(title) || title.includes(queryLower);
          });
          if (titleHits.length === 1) {
            roleRecord = titleHits[0];
          } else {
            const candidates = await this._semanticRoleFilter(roleTitle, companyRoles);
            if (candidates.length > 1) {
              return this._askDisambiguate(state, companyRecord, candidates);
            } else if (candidates.length === 1) {
              roleRecord = candidates[0];
            }
          }
        }
      }
    }

    if (!roleRecord && !companyRecord) {
      const entity = companyName || roleTitle;
      return (
        `I don't have "${entity}" in our database yet. ` +
        "Tell me what you know about it — what's the company, the role, " +
        'and what have you learned from your conversations?'
      );
    }

    // Premium: company found but no matching role → show other roles at this company
    if (!roleRecord && companyRecord && roleTitle && state.mode === 'premium') {
      return await this._premiumRoleNotFoundResponse(state, companyRecord, roleTitle);
    }

    if (companyRecord) {
      state.companyRecordId = companyRecord.id;
      state.companyName = this._field(companyRecord.fields, 'Company Name', companyName || '');
      const rawDomain = this._field(companyRecord.fields, 'Domain');
      state.companyDomain = this._ensureHttps(rawDomain.trim());
    } else if (companyName) {
      state.companyName = companyName;
    }

    if (roleRecord) {
      state.roleRecordId = roleRecord.id;
      state.roleTitle = this._field(roleRecord.fields, 'Title', roleTitle || '');
      state.roleAppPage = this._field(roleRecord.fields, 'App Page').trim();
    }

    return await this._dispatchAfterMatch(state, userText);
  }

  async _handleConfirming(state, userText) {
    // Legacy phase handler — state should no longer enter CONFIRMING in normal flow.
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
      const desc = (c.fields || {}).Description || '';
      const blurb = desc ? `: ${desc.substring(0, 80)}...` : '';
      return `${i + 1}. **${name}**${blurb}`;
    });

    return (
      `I found a few companies that match — which one did you mean?\n\n${lines.join('\n')}\n\n` +
      '**Just reply with the number or the company name.**'
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
      return "I didn't catch which one you meant. **Could you reply with the number or the full company name?**";
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
        const roles = await this.db.getCompanyRoles(state.companyRecordId);
        const count = roles.length;
        const noun = count === 1 ? 'role' : 'roles';
        const coRef = this._companyRef(state);
        if (count) {
          return (
            `We do have ${count} ${noun} tracked for ${coRef}, ` +
            'but the details are only available on Pro and above. ' +
            '**Upgrade to Pro to see the role titles and hiring details.**'
          );
        }
        return `I don't have any roles tracked for ${coRef} at the moment.`;
      }
      const coRec = await this.db.getCompany(state.companyRecordId);
      return await this._generateCompanySynopsis(coRec, 'free', state);
    }

    // FREE + role | PRO (any)
    if (mode === 'free' || mode === 'pro') {
      if (mode === 'pro' && rolesListIntent) {
        state.phase = Phase.COMPANY_FOUND;
        const roles = await this.db.getCompanyRoles(state.companyRecordId);
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
        const roleStatus = (((roleRec || {}).fields || {}).Status || '').toLowerCase();
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
          `Tell me what you already know about ${entityRef} from your research or conversations. ` +
          "**Share what you've learned and I'll let you know what we have.**"
        );
      } else { // pro
        return (
          `Great — before I pull up what we have on ${entityRef}, ` +
          "**what have you already learned from your conversations or research?**"
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
        const ackPrompt = (
          `The user shared this about ${entity}: "${userText}"\n\n` +
          '1. Warmly acknowledge their contribution in 1 sentence.\n' +
          '2. Confirm that we do have this role in our database.\n' +
          '3. Do NOT reveal any details we have about the role.\n' +
          '4. Ask ONE natural follow-up question to learn more.\n' +
          '5. In a final sentence, mention they can upgrade to Pro to see what we know.\n' +
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
        const ackPrompt = (
          `The user shared this about ${entity}: "${userText}"\n\n` +
          '1. Warmly thank them for the insight in 1 sentence.\n' +
          '2. Ask ONE focused follow-up question to gather more insider info ' +
          '(culture, hiring process, leadership, recent changes).\n' +
          'Bold only the question sentence using **double asterisks**. No other markdown.'
        );
        return await this._callClaude([{ role: 'user', content: ackPrompt }]);
      }

      synopsis = coRec ? await this._generateCompanySynopsis(coRec, mode, state) : '';
      state.phase = Phase.COMPANY_FOUND;
    }

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
    // Guard: if the user is clearly replying to the agent's last question, skip entity-switching.
    if (await this._isContinuationReply(state, userText)) {
      const clarification = await this._attributionClarification(state, userText);
      if (clarification) return clarification;
      await this._extractAndAccumulate(state, userText);
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
      const companyRoles = await this.db.getCompanyRoles(state.companyRecordId);
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

    // Role listing intent
    if (state.companyRecordId && this._isRolesListIntent(userText)) {
      if (state.mode === 'premium') {
        return await this._listCompanyRoles(state);
      } else {
        const roles = await this.db.getCompanyRoles(state.companyRecordId);
        const count = roles.length;
        const noun = count === 1 ? 'role' : 'roles';
        const coRef = this._companyRef(state);
        if (!count) return `I don't have any roles tracked for ${coRef} at the moment.`;
        if (state.mode === 'free') {
          return (
            `We do have ${count} ${noun} tracked for ${coRef}, ` +
            'but the details are only available on Pro and above. ' +
            '**Upgrade to Pro to unlock the role titles and hiring details.**'
          );
        }
        // pro
        return (
          `We do have ${count} ${noun} tracked for ${coRef}. ` +
          '**Upgrade to Premium to see the full breakdown — ' +
          "or is there a specific role you've already come across?**"
        );
      }
    }

    const clarification = await this._attributionClarification(state, userText);
    if (clarification) return clarification;
    await this._extractAndAccumulate(state, userText);
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

    const prompt = (
      `We are capturing intel about ${focus}.\n` +
      `User message: "${userText}"\n\n` +
      'Does this message contain intel that is clearly about the entity above, ' +
      'or does it reference a different role or company in a way that makes attribution ambiguous?\n' +
      'Reply with JSON only: {"clear": true} or {"clear": false, "entity": "<what they seem to be referring to>"}'
    );

    try {
      const raw = await this._callClaude([{ role: 'user', content: prompt }], { maxTokens: 60 });
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
    if (roleGaps.length > 0 || companyGaps.length > 0) {
      const allGapDescs = [...roleGaps, ...companyGaps].map(([, desc]) => desc);
      const gapList = allGapDescs.map((d, i) => `${i + 1}. ${d}`).join('\n');
      system = (
        SYSTEM_PROMPT +
        '\n\nGaps we still want to fill (listed in priority order):\n' +
        gapList +
        '\n\nEnd with ONE question about whichever gap is most natural to ask about given the conversation so far. ' +
        'If the user has already mentioned something relevant to a gap in this conversation, skip that gap and move to the next. ' +
        'Frame it as a natural, conversational question — not a form field. ' +
        'Do NOT ask why the user wants the role or anything about their personal motivations or background.'
      );
    } else {
      system = (
        SYSTEM_PROMPT +
        '\n\nEnd with ONE question about the hiring process, timeline, or how the search is being run. ' +
        'Do NOT ask why the user wants the role or anything about their personal background or motivations.'
      );
    }

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

  _isRolesListIntent(text) {
    const low = text.toLowerCase();
    return [
      'what roles', 'which roles', 'list roles', 'any roles', 'open roles',
      'roles do you have', 'roles you have', 'roles we have', 'tell me roles',
      'tell me about the roles', 'tell me the roles',
      'roles in our', 'roles in the', 'roles at', 'roles for',
      'what positions', 'any positions', 'open positions',
      'what openings', 'any openings',
    ].some(p => low.includes(p));
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
      r => this._field(r.fields, 'Status', 'open').toLowerCase() !== 'closed'
    );
    const closedRoles = roles.filter(
      r => this._field(r.fields, 'Status').toLowerCase() === 'closed'
    );
    const companyUrl = state.companyDomain || '';
    const prompt = buildRolesListingPrompt(coRec || {}, openRoles, closedRoles, companyUrl);
    return await this._callClaude([{ role: 'user', content: prompt }]);
  }

  // ------------------------------------------------------------------
  // Synopsis generators
  // ------------------------------------------------------------------

  async _generateCompanySynopsis(companyRecord, mode = 'premium', state = null) {
    const roles = await this.db.getCompanyRoles(companyRecord.id);
    const companyUrl = state ? (state.companyDomain || '') : '';
    const prompt = buildCompanySynopsisPrompt(companyRecord, roles, [], mode, companyUrl);
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

  async _premiumRoleNotFoundResponse(state, companyRecord, roleQuery) {
    state.companyRecordId = companyRecord.id;
    state.companyName = this._field(companyRecord.fields, 'Company Name');
    const rawDomain = this._field(companyRecord.fields, 'Domain');
    state.companyDomain = this._ensureHttps(rawDomain.trim());
    state.phase = Phase.COMPANY_FOUND;

    const roles = await this.db.getCompanyRoles(companyRecord.id);
    const coRef = this._companyRef(state);

    if (!roles.length) {
      return (
        `I couldn't find a **${roleQuery}** role at ${coRef} — ` +
        "and we don't have any other roles tracked there right now. " +
        "Tell me what you know about it and I'll capture it."
      );
    }

    const openRoles = roles.filter(
      r => this._field(r.fields, 'Status', 'open').toLowerCase() !== 'closed'
    );
    const displayRoles = openRoles.length > 0 ? openRoles : roles;
    const roleLines = displayRoles.map(r => {
      const rf = r.fields || {};
      let line = `- **${rf.Title || 'Untitled'}**`;
      if (rf.Function) line += ` (${rf.Function})`;
      return line;
    });

    return (
      `I couldn't find a **${roleQuery}** role at ${coRef}, ` +
      `but here are the other roles we're tracking there:\n\n${roleLines.join('\n')}\n\n` +
      'Is one of these what you were looking for?'
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
