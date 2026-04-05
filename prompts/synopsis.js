'use strict';

/**
 * Prompt builders for the Insights agent synopsis generation.
 *
 * mode values:
 *   'free'    – public info only; no confidential notes, no insights
 *   'pro'     – full info minus contributor attribution; insights anonymised
 *   'premium' – full info including who contributed each insight
 */

/**
 * Build a prompt for a company synopsis.
 * @param {object} company  - Airtable record {id, fields}
 * @param {Array}  roles    - Array of role records
 * @param {Array}  insights - Array of insight records
 * @param {string} mode
 * @param {string} companyUrl
 * @returns {string}
 */
function buildCompanySynopsisPrompt(company, roles, insights, mode = 'premium', companyUrl = '', rolesSummary = null, hasInsights = false) {
  const fields = company.fields || {};
  const companyName = fields['Company Name'] || 'Unknown';
  const companyRef = companyUrl ? `[${companyName}](${companyUrl})` : companyName;

  let investors = fields.Investors || '';
  if (Array.isArray(investors)) investors = investors.join(', ');

  let companySection = `COMPANY: ${companyName}

Description:
${fields.Description || 'No description on file.'}

Headcount Growth % (last 6 months) — negative means headcount shrank, positive means it grew:
${fields.HG6M != null && fields.HG6M !== '' ? fields.HG6M + '%' : 'Not available.'}

Number of Employees: ${fields.Employees || 'Unknown'}
Investors: ${investors || 'Unknown'}`.trim();

  if (mode === 'pro' || mode === 'premium') {
    const conf = (fields['Confidential Notes'] || '').trim();
    if (conf) {
      companySection += `\n\nConfidential Notes (internal only – do NOT share verbatim, summarise tastefully):\n${conf}`;
    }
  }

  let rolesSection;
  let freeRolesLine = '';   // pre-built line for free tier (used verbatim in prompt)
  let freeClosingQ = '';    // pre-built closing question for free tier
  if (mode === 'free' && rolesSummary) {
    const { postedActive = [], unpostedActiveCount = 0, closedCount = 0 } = rolesSummary;
    const postedCount = postedActive.length;
    if (postedCount > 0 || unpostedActiveCount > 0) {
      // Build the "We have X posted role(s) (Title) and Y unposted roles at Company" line
      const parts = [];
      if (postedCount === 1) {
        const title = (postedActive[0].fields || {}).Title || 'Untitled';
        parts.push(`1 posted role (${title})`);
      } else if (postedCount > 1) {
        parts.push(`${postedCount} posted roles`);
      }
      if (unpostedActiveCount > 0) {
        const noun = unpostedActiveCount === 1 ? 'unposted role' : 'unposted roles';
        parts.push(`${unpostedActiveCount} ${noun}`);
      }
      freeRolesLine = `We have ${parts.join(' and ')} at ${companyName}.`;
      if (postedCount > 1) {
        const numbered = postedActive.map((r, i) => `${i + 1}. ${(r.fields || {}).Title || 'Untitled'}`).join('\n');
        freeRolesLine += '\n' + numbered;
        freeClosingQ = '**Are you interested in one of these roles, or do you have insights on the company?**';
      } else if (postedCount === 1) {
        freeClosingQ = '**Are you interested in this role, or do you have insights on the company?**';
      } else {
        // Only unposted roles — user can't see the role title so don't ask if they're interested in it
        freeClosingQ = `**Do you have any insights on the team or business at ${companyName}, or are there other roles or companies you're interested in?**`;
      }
    } else if (closedCount > 0) {
      const noun = closedCount === 1 ? 'role' : 'roles';
      freeRolesLine = `No open roles right now — ${closedCount} previously tracked ${noun} are closed.`;
      freeClosingQ = '**Do you have any insights on the company?**';
    } else {
      freeRolesLine = 'No roles currently tracked.';
      freeClosingQ = '**Do you have any insights on the company?**';
    }
    rolesSection = `ROLES LINE (use this verbatim): ${freeRolesLine}`;
  } else if (roles && roles.length > 0) {
    const roleLines = roles.map(r => {
      const rf = r.fields || {};
      const regionArr = rf.Region || [];
      const region = Array.isArray(regionArr) ? regionArr.join(', ') : regionArr;
      return (
        `- ${rf.Title || 'Untitled'} | ` +
        (rf.Function ? `Function: ${rf.Function} | ` : '') +
        `Hiring Manager: ${rf['HM Name'] || 'Unknown'} | ` +
        `Region: ${region || fields.HQ || 'Unknown'} | ` +
        `Remote: ${rf.Remote ? 'Yes' : 'No'} | ` +
        `Find: ${rf.Find || ''} | ` +
        `Notes: ${rf.Notes || ''}`
      );
    });
    rolesSection = 'OPEN ROLES:\n' + roleLines.join('\n');
  } else {
    rolesSection = 'OPEN ROLES: No open roles currently tracked.';
  }

  let insightsSection = '';
  if (mode === 'free') {
    insightsSection = '';
  } else if (mode === 'pro') {
    if (insights && insights.length > 0) {
      const lines = insights.map(i => {
        const fi = i.fields || {};
        return `- [${fi.Timestamp || ''}] ${fi.Content || ''}`;
      });
      insightsSection = 'COMMUNITY INSIGHTS (do NOT attribute to specific contributors):\n' + lines.join('\n');
    } else {
      insightsSection = 'COMMUNITY INSIGHTS: No community insights yet.';
    }
  } else { // premium
    if (insights && insights.length > 0) {
      const lines = insights.map(i => {
        const fi = i.fields || {};
        return `- [${fi.Timestamp || ''}] ${fi.Content || ''} — shared by ${fi.Contributor || 'anonymous'}`;
      });
      insightsSection = 'COMMUNITY INSIGHTS:\n' + lines.join('\n');
    } else {
      insightsSection = 'COMMUNITY INSIGHTS: No community insights yet.';
    }
  }

  let modeInstruction = '';
  if (mode === 'free') {
    modeInstruction = (
      'IMPORTANT: This is a Free-tier response. Share only public information ' +
      '(description, headcount, investors, role counts). Do NOT mention ' +
      'confidential notes, community insights, or role titles.'
    );
  } else if (mode === 'pro') {
    modeInstruction = (
      'IMPORTANT: This is a Pro-tier response. You may reference community insights ' +
      'but do NOT name or attribute them to specific contributors.'
    );
  }

  const sections = [companySection, rolesSection];
  if (insightsSection) sections.push(insightsSection);
  const body = sections.join('\n\n');

  const linkInstruction = companyUrl
    ? `Formatting: whenever you mention the company name in your response, write it as the markdown hyperlink ${companyRef} — do not write the plain name.\n\n`
    : '';

  return `You are the Insights agent for a professional community. A member just asked about this company.

${modeInstruction ? modeInstruction + '\n\n' : ''}${linkInstruction}DATA ON FILE:
---
${body}
---

${freeRolesLine ? `Write a SHORT response that:
1. Gives the most useful snapshot of the company — what makes it interesting right now (1-2 sentences).
2. Include the ROLES LINE above exactly as written — do not rephrase it.
3. End with this EXACT closing question: ${freeClosingQ}

IMPORTANT: Only reference facts that are explicitly present in the DATA ON FILE above. Do NOT invent, estimate, or compute any metric, score, or figure that does not appear verbatim in the data. If a field says "Not available" or "N/A", omit it entirely.
Do not use any markdown other than the bold in the closing question above.` : `Write a SHORT response that:
1. Gives the most useful snapshot of the company — what makes it interesting right now (1-2 sentences).
2. If there are open roles, introduce them like this (use the exact company name and adapt count naturally): "${hasInsights ? 'We have confidential insights from other members and [N] open role(s) at [Company Name]: [Title]' : 'We have [N] open role(s) at [Company Name]: [Title]'}". List each role on its own numbered line if there are multiple.
3. Ends with ONE short, generic question: "**Are you interested in one of these roles, or do you have insights on the company?**" (adapt wording naturally if there's only one role or no roles). Do NOT ask a specific industry or market question.

IMPORTANT: Only reference facts that are explicitly present in the DATA ON FILE above. Do NOT invent, estimate, or compute any metric, score, or figure that does not appear verbatim in the data. If a field says "Not available" or "N/A", omit it entirely.
Bold only the closing question using **double asterisks**. Do not use any other markdown outside the numbered role list. Do NOT try to share everything — leave room for dialogue.`}`;
}


/**
 * Build a prompt for a role-specific synopsis.
 * @param {object} role       - Airtable record {id, fields}
 * @param {object} company    - Airtable record {id, fields}
 * @param {Array}  insights
 * @param {string} mode
 * @param {string|null} topGap
 * @param {string} companyUrl
 * @param {string} roleUrl
 * @returns {string}
 */
function buildRoleSynopsisPrompt(role, company, insights, mode = 'premium', topGap = null, companyUrl = '', roleUrl = '') {
  const rf = (role || {}).fields || {};
  const cf = (company || {}).fields || {};

  const roleTitle = rf.Title || 'Unknown';
  const roleRef = roleUrl ? `[${roleTitle}](${roleUrl})` : roleTitle;
  const companyName = cf['Company Name'] || 'Unknown';
  const companyRef = companyUrl ? `[${companyName}](${companyUrl})` : companyName;

  const regionArr = rf.Region || [];
  const region = Array.isArray(regionArr) ? regionArr.join(', ') : regionArr;

  const functionLine = rf.Function ? `\nFunction: ${rf.Function}` : '';
  const compLine = rf['Role - Compensation'] != null && rf['Role - Compensation'] !== ''
    ? `\nCompensation: ${rf['Role - Compensation']}`
    : '';
  let roleSection = `ROLE: ${roleTitle}${functionLine}
Company: ${companyName}
Hiring Manager: ${rf['HM Name'] || 'Unknown'}
Region: ${region || cf.HQ || 'Unknown'}
Remote: ${rf.Remote ? 'Yes' : 'No'}
How to Find / Apply: ${rf.Find || 'Unknown'}
Notes: ${rf.Notes || 'None.'}${compLine}`.trim();

  let insightsSection = '';
  if (mode === 'free') {
    insightsSection = '';
  } else if (mode === 'pro') {
    if (insights && insights.length > 0) {
      const lines = insights.map(i => {
        const fi = i.fields || {};
        return `- [${fi.Timestamp || ''}] ${fi.Content || ''}`;
      });
      insightsSection = 'COMMUNITY INSIGHTS ON THIS ROLE (do NOT attribute to specific contributors):\n' + lines.join('\n');
    } else {
      insightsSection = 'COMMUNITY INSIGHTS ON THIS ROLE: No insights yet – you could be the first!';
    }
  } else { // premium
    if (insights && insights.length > 0) {
      const lines = insights.map(i => {
        const fi = i.fields || {};
        return `- [${fi.Timestamp || ''}] ${fi.Content || ''} — shared by ${fi.Contributor || 'anonymous'}`;
      });
      insightsSection = 'COMMUNITY INSIGHTS ON THIS ROLE:\n' + lines.join('\n');
    } else {
      insightsSection = 'COMMUNITY INSIGHTS ON THIS ROLE: No insights yet – you could be the first!';
    }
  }

  const hg6m = cf.HG6M != null && cf.HG6M !== '' ? cf.HG6M + '%' : 'N/A';
  const companyContext = `Company Description: ${cf.Description || 'N/A'}
Headcount Growth % (last 6 months) — negative means headcount shrank, positive means it grew: ${hg6m}`.trim();

  let modeInstruction = '';
  if (mode === 'free') {
    modeInstruction = 'IMPORTANT: This is a Free-tier response. Do NOT share insights or confidential details.';
  } else if (mode === 'pro') {
    modeInstruction = 'IMPORTANT: This is a Pro-tier response. Do NOT name or attribute insights to specific contributors.';
  }

  const sections = [roleSection, companyContext];
  if (insightsSection) sections.push(insightsSection);
  const body = sections.join('\n\n');

  const canAskMore = mode === 'premium'
    ? 'Let them know they can ask you more questions for additional details.'
    : '';

  let endingInstruction;
  if (mode === 'free') {
    endingInstruction = (
      'End with ONE open, welcoming question inviting them to share what they\'ve learned ' +
      'about this role from their conversations or research — make it easy to share anything, ' +
      'not just a specific detail. Do NOT ask why the user wants the role or anything about their personal motivations.'
    );
  } else {
    endingInstruction = (
      'End with ONE open, welcoming question inviting them to share what they\'ve learned ' +
      'about this role from their conversations or research — make it easy for them to share ' +
      'anything they know, not just a specific detail. ' +
      'Do NOT ask why the user wants the role or anything about their personal background or motivations.'
    );
  }

  const linkParts = [];
  if (roleUrl) linkParts.push(`the role name as ${roleRef}`);
  if (companyUrl) linkParts.push(`the company name as ${companyRef}`);
  const linkInstruction = linkParts.length > 0
    ? `Formatting: whenever you mention ${linkParts.join(' and ')} in your response, use those exact markdown hyperlinks — do not write the plain names.\n\n`
    : '';

  const strategiesStep = canAskMore ? '4' : '3';
  const endingStep = canAskMore ? '5' : '4';
  return `You are the Insights agent for a professional community. A member just asked about this role.

${modeInstruction ? modeInstruction + '\n\n' : ''}${linkInstruction}DATA ON FILE:
---
${body}
---

Write a SHORT response (3-4 sentences max) that:
1. Summarises what we know about the role — the 1-2 most useful things.
2. Mentions anything notable about the hiring process if we have it.
${canAskMore ? `3. ${canAskMore}\n` : ''}${strategiesStep}. Mention that we have hundreds of playbooks on how to find and win ${roleTitle} roles and that they can click on the **Strategies** agent for advice — keep this to one sentence.
${endingStep}. ${endingInstruction}

IMPORTANT: Only reference facts that are explicitly present in the DATA ON FILE above. Do NOT invent, estimate, or compute any metric, score, or figure (such as a "growth outlook score") that does not appear verbatim in the data. If a field says "Not available" or "N/A", omit it entirely.

Bold only the question sentence using **double asterisks**. Do not use any other markdown. Do NOT try to share everything at once — the goal is to start a dialogue.`;
}


/**
 * Build a prompt for a premium user asking what roles we have at a company.
 * @param {object} company
 * @param {Array}  openRoles
 * @param {Array}  closedRoles
 * @param {string} companyUrl
 * @returns {string}
 */
function buildRolesListingPrompt(company, openRoles, closedRoles, companyUrl = '') {
  const cf = (company || {}).fields || {};
  const companyName = cf['Company Name'] || 'this company';
  const companyRef = companyUrl ? `[${companyName}](${companyUrl})` : companyName;

  function formatRole(r) {
    const rf = r.fields || {};
    const regionArr = rf.Region || [];
    const region = (Array.isArray(regionArr) ? regionArr.join(', ') : regionArr) || cf.HQ || 'Unknown';
    const remoteFlag = rf.Remote ? 'Remote' : '';
    const location = [region, remoteFlag].filter(Boolean).join(' | ');
    const title = rf.Title || 'Untitled';
    const appPage = (rf['App Page'] || '').trim();
    const titleRef = appPage ? `[${title}](${appPage})` : title;
    // Use the role's own resolved company name (set by _listCompanyRoles),
    // falling back to the queried companyName only when not available.
    const roleCompany = rf._company_name || companyName;
    const parts = [`${titleRef} at ${roleCompany}`];
    if (rf.Function) parts.push(`Function: ${rf.Function}`);
    if (rf['HM Name']) parts.push(`HM: ${rf['HM Name']}`);
    if (location && location !== 'Unknown') parts.push(`Location: ${location}`);
    if (rf.Find) parts.push(`Find: ${rf.Find}`);
    if (rf.Notes) parts.push(`Notes: ${rf.Notes}`);
    return '- ' + parts.join(' | ');
  }

  const openSection = openRoles && openRoles.length > 0
    ? openRoles.map(formatRole).join('\n')
    : 'None tracked.';
  const closedSection = closedRoles && closedRoles.length > 0
    ? closedRoles.map(formatRole).join('\n')
    : 'None tracked.';

  const multipleOpen = openRoles && openRoles.length > 1;
  const endingInstruction = multipleOpen
    ? 'End by asking the user which of these roles they\'d like to explore further — do NOT ask a specific question about one role before knowing which one they care about.'
    : 'End with ONE question about what the user has heard regarding the hiring process or timeline.';

  const linkInstruction = companyUrl
    ? `Formatting: when mentioning the company name use ${companyRef}, and when listing role titles use the markdown links provided in OPEN ROLES above — do not write plain names.\n\n`
    : '';

  return `You are the Insights agent for a professional community. A premium member asked about the roles we have tracked for ${companyRef}.

${linkInstruction}OPEN ROLES:
${openSection}

RECENTLY CLOSED ROLES:
${closedSection}

Write a response that:
1. Opens with one short sentence (e.g. "We have X open roles at Company:").
2. Lists each open role as a numbered line in this exact format (one role per line):
   #1 [Role Title](link if available) — HM: Name | Location
   Use the role's company name, hiring manager, and location. If a value is unknown, omit that part.
3. Mentions recently closed roles briefly in a sentence after the list, if any exist.
4. ${endingInstruction}

IMPORTANT: Never attribute a role to a company other than the one stated in the data above.
Bold only the question/prompt sentence using **double asterisks**. Do not use any other markdown outside the numbered list lines.`;
}


/**
 * Build a prompt that guides Claude to ask for missing company/role information.
 * @param {string} entityType  - 'company' | 'role'
 * @param {string} entityName
 * @param {object} existingFields
 * @returns {string}
 */
function buildInfoCollectionPrompt(entityType, entityName, existingFields) {
  let desired;
  if (entityType === 'company') {
    desired = {
      Description: 'a general description of what the company does',
      HG6M: 'their growth outlook / momentum over the next 6 months',
      Employees: 'approximate headcount',
      'Confidential Notes': 'any insider intel or confidential context',
    };
  } else {
    desired = {
      'HM Name': 'who the hiring manager is',
      'HQ Location': 'where the role is based (remote/hybrid/on-site, city)',
      Find: 'how to find or apply for the role',
      Notes: 'any other notes about the role or interview process',
    };
  }

  const missing = Object.entries(desired)
    .filter(([field]) => !existingFields[field])
    .map(([, desc]) => desc);

  if (!missing.length) {
    return (
      `We actually have pretty good coverage on ${entityName}. ` +
      'Ask the user if they have any additional insights or recent updates to share.'
    );
  }

  const missingStr = missing.map(m => `- ${m}`).join('\n');
  return `You are collecting insights from a community member about ${entityType} "${entityName}".

We are missing the following information:
${missingStr}

Ask the user for this information in a natural, conversational way. Ask one or two questions at a time – don't overwhelm them. Be warm and appreciative of their contribution.`;
}


module.exports = {
  buildCompanySynopsisPrompt,
  buildRoleSynopsisPrompt,
  buildRolesListingPrompt,
  buildInfoCollectionPrompt,
};
