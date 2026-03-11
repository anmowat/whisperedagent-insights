'use strict';

/**
 * Data collection: field schema, gap analysis, and extraction prompts.
 */

// ── Role gap descriptions (in conversational language for Claude) ─────────────

// Ordered by collection priority: Find → Location → Notes → Company Confidential Notes → Compensation
const ROLE_GAP_DESCRIPTIONS = {
  Find: (
    'how to find or reach the hiring manager — the internal sponsor, ' +
    'the external recruiter or search firm, and any community member who can ' +
    'help make an introduction'
  ),
  Region: "where the role is based (region/city) and whether it's remote, hybrid, or in-office",
  Notes: (
    'role details across three areas — Scope (responsibilities and team size); ' +
    'Criteria (key skills, interview panel, reason for hire); ' +
    'Details (hiring manager, who the role reports to)'
  ),
  // Compensation is lowest priority — only surfaced once everything else is known
  Compensation: (
    'total compensation — only worth asking if the user is deep in the process ' +
    'or volunteers it; frame naturally, e.g. asking about the overall package'
  ),
};

// ── Company confidential gap descriptions ─────────────────────────────────────

const COMPANY_GAP_DESCRIPTIONS = {
  Status: (
    'company health — ARR, employee count, growth rate, competitive dynamics, ' +
    'when they last raised and what their runway looks like'
  ),
  'GTM Motion': (
    'how they go to market — inbound vs outbound mix, ACV, who they sell to ' +
    '(buyer personas), and whether GRR/NRR are healthy'
  ),
  'GTM Org': (
    'how the GTM org is structured — who leads each function, and whether ' +
    'GTM decisions are made by a CRO or are CEO-driven'
  ),
  'CEO/Culture': (
    'the CEO and culture — how the CEO is perceived (trusted, founder-mode, ' +
    'hands-off), and what the work culture is like (pace, intensity, values)'
  ),
};


// ── Gap analysis ──────────────────────────────────────────────────────────────

/**
 * Return [[fieldName, description], ...] for role fields that are empty.
 * @param {object} roleFields
 * @returns {Array<[string, string]>}
 */
function getRoleGaps(roleFields) {
  return Object.entries(ROLE_GAP_DESCRIPTIONS).filter(
    ([field]) => !String(roleFields[field] || '').trim()
  );
}

/**
 * Return [[topic, description], ...] for company confidential topics not yet covered.
 * @param {object} companyFields
 * @returns {Array<[string, string]>}
 */
function getCompanyGaps(companyFields) {
  const notes = (companyFields['Confidential Notes'] || '').toLowerCase();
  const keywords = {
    Status: ['arr', 'revenue', 'growth', 'runway', 'raise', 'funding', 'employees'],
    'GTM Motion': ['inbound', 'outbound', 'acv', 'grr', 'nrr', 'gtm', 'channel'],
    'GTM Org': ['cro', 'vp sales', 'vp marketing', 'gtm org', 'reporting'],
    'CEO/Culture': ['ceo', 'founder', 'culture', 'work style', 'pace'],
  };
  return Object.entries(COMPANY_GAP_DESCRIPTIONS).filter(([topic]) => {
    const covered = (keywords[topic] || []).some(kw => notes.includes(kw));
    return !covered;
  });
}


// ── Extraction prompt ─────────────────────────────────────────────────────────

/**
 * Prompt Claude to pull structured field values out of a natural-language message.
 * @param {string} userText
 * @param {string} roleName
 * @param {string} companyName
 * @returns {string}
 */
function buildDataExtractionPrompt(userText, roleName, companyName) {
  return `Extract any database-worthy information from this message about role "${roleName}" at "${companyName}".

Role fields (only include if explicitly mentioned):
- Find: who leads the hiring search (internal sponsor, recruiter name/firm, helpful community contacts)
- Notes: structured as three sections — Scope (responsibilities, team size); Criteria (key skills, interview panel, reason for hire); Details (hiring manager, who the role reports to, compensation context — always capture comp figures or qualitative comp info here e.g. "$180-200k base + 25% bonus", "comp reportedly low", "strong equity component").
- Region: free-text description capturing all location detail — cities, regions, days in office, remote/hybrid/in-office policy (e.g. "SF or NYC, 3 days in office, open to remote for right candidate")
- Compensation: INTEGER only — the annual USD cash total (base + bonus / OTE) as a plain number with no symbols, words, or punctuation (e.g. 180000). If the user gives a range use the midpoint. If the figure is vague or only qualitative (e.g. "low", "competitive") set this to null and capture the context in Notes instead.

Company field (only include if mentioned):
- Confidential Notes: any non-public intel — ARR, growth, GTM strategy, org structure, leadership, CEO, culture

Message: "${userText}"

Rules:
- Only populate fields where the user actually shared relevant info
- Combine related details into the right field (e.g. "HM is Sarah, team of 12" both go in Notes)
- For Confidential Notes collect all company intel in one string
- Compensation must be a plain integer or null — never a string
- Return ONLY valid JSON, nothing else

{"role":{"Find":null,"Notes":null,"Region":null,"Compensation":null},"company":{"Confidential Notes":null}}`;
}


// ── Structured field schemas ──────────────────────────────────────────────────

const ROLE_NOTES_SCHEMA = `Scope: <responsibilities and team size>
Criteria: <key skills, interview panel, reason for hire>
Details: <hiring manager, who the role reports to, compensation context e.g. "$180-200k base + 25% bonus" or "comp reportedly low">`;

const COMPANY_NOTES_SCHEMA = `Status: ARR ~$Xm, ~Y employees, growth rate Z%, <competitive dynamics>, last raise <series / date>, runway <X months>
GTM Motion: <inbound/outbound mix>, ACV $X, buyers <personas>, GRR X% / NRR X%
GTM Org: <structure>, function leads: <names/titles>, GTM decisions: <CRO-led / CEO-driven>
CEO/Culture: CEO <name> — <reputation/style>, culture: <description>`;


/**
 * Prompt Claude to merge newInfo into existing structured notes.
 * @param {string} schemaType - 'role_notes' | 'company_notes'
 * @param {string} existing
 * @param {string} newInfo
 * @returns {string}
 */
function buildStructuredMergePrompt(schemaType, existing, newInfo) {
  const isRole = schemaType === 'role_notes';
  const schema = isRole ? ROLE_NOTES_SCHEMA : COMPANY_NOTES_SCHEMA;
  const label = isRole ? 'Role.Notes' : 'Company.Confidential Notes';
  const existingBlock = (existing || '').trim() || '(empty)';

  return `You are updating the ${label} field for a tracked company/role.

TARGET SCHEMA (output must follow this exact structure):
---
${schema}
---

EXISTING CONTENT:
---
${existingBlock}
---

NEW INFORMATION TO MERGE IN:
---
${(newInfo || '').trim()}
---

Instructions:
- Produce a clean merged version that follows the schema above exactly.
- Each line starts with the label in bold (e.g. "Scope & Responsibilities:") followed by the value.
- NEVER remove or summarise away information from existing content — every fact already there must be preserved unless the new information explicitly contradicts it (e.g. a correction).
- Incorporate all new information from "NEW INFORMATION TO MERGE IN".
- Remove exact duplicates only; keep the most specific / recent value when there is a direct conflict.
- If a section has no information at all (existing or new), omit that line entirely — do NOT write "Unknown".
- Be concise: one line per section unless the content genuinely needs more.
- Return ONLY the merged field content — no preamble, no JSON, no markdown fences.`;
}


/**
 * Prompt Claude to synthesize newInfo into existing for a plain text field.
 * @param {string} fieldName
 * @param {string} existing
 * @param {string} newInfo
 * @returns {string}
 */
function buildSimpleFieldMergePrompt(fieldName, existing, newInfo) {
  const existingBlock = (existing || '').trim() || '(empty)';
  return `You are updating the "${fieldName}" field in a recruiting database.

EXISTING VALUE:
---
${existingBlock}
---

NEW INFORMATION TO INCORPORATE:
---
${(newInfo || '').trim()}
---

Instructions:
- Write a single concise sentence or phrase that captures ALL the information from both.
- NEVER remove any unique facts from the existing value.
- Eliminate redundancy: if the new information is already expressed in the existing value (even partially or in different words), do not repeat it — just keep the best phrasing.
- If the new information contradicts the existing value, keep both with a note (e.g. "X or Y").
- Return ONLY the merged value — no preamble, no labels, no quotes, no markdown.`;
}


// ── Follow-up question prompt ─────────────────────────────────────────────────

/**
 * Prompt Claude to ask ONE natural follow-up question targeting the highest-priority gap.
 * @param {string} roleName
 * @param {string} companyName
 * @param {Array<[string,string]>} roleGaps
 * @param {Array<[string,string]>} companyGaps
 * @returns {string}
 */
function buildGapQuestionPrompt(roleName, companyName, roleGaps, companyGaps) {
  const allGaps = [...roleGaps.map(([, d]) => d), ...companyGaps.map(([, d]) => d)];
  if (!allGaps.length) {
    return (
      `You already have good coverage on ${roleName || companyName}. ` +
      'Ask the user if they have any recent updates or additional observations to share. ' +
      'One sentence, warm and casual.'
    );
  }

  const topTwo = allGaps.slice(0, 2);
  const gapsText = topTwo.map(g => `- ${g}`).join('\n');

  return `You are chatting with a community member about "${roleName || companyName}".

Things we'd love to learn (pick the most natural one to ask about):
${gapsText}

Ask ONE open, conversational question that might uncover this information.
- Don't ask for specific fields by name ("what is the team size?" is too robotic)
- Ask naturally ("What's the team setup like there?" is better)
- One question only, 1-2 sentences, warm and curious tone.`;
}


module.exports = {
  ROLE_GAP_DESCRIPTIONS,
  COMPANY_GAP_DESCRIPTIONS,
  getRoleGaps,
  getCompanyGaps,
  buildDataExtractionPrompt,
  buildStructuredMergePrompt,
  buildSimpleFieldMergePrompt,
  buildGapQuestionPrompt,
};
