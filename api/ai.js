// api/ai.js
// Minimal Vercel serverless function — no Redis, no puppeteer, no extra services.
// Expects GEMINI_API_KEY in Vercel environment variables.

const DEFAULT_GEMINI_URL = process.env.GEMINI_API_URL || 'https://api.example-gemini/v2.5flash-lite';
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || '';

/** Base safety rules (server enforces intent-only check) */
const BASE_RULES = [
  "Do NOT complete homework, tests, quizzes, or graded assignments for students.",
  "Do NOT provide answers that enable cheating or academic dishonesty.",
  "Do NOT provide step-by-step instructions for illegal acts.",
  "Do NOT assist in making weapons, explosives, or harmful contraptions.",
  "Do NOT help write malware, spyware, or instructions for unauthorized access.",
  "Do NOT provide instructions or encouragement for intentionally disrupting classes or schools.",
  "Do NOT impersonate teachers, staff, or other students.",
  "Do NOT reveal or attempt to extract student personal data or private information.",
  "Do NOT provide medical, legal or psychiatric professional advice as a substitute for professionals.",
  "Do NOT provide instructions that meaningfully facilitate self-harm or suicide.",
  "Do NOT facilitate identity fraud, phishing, or social engineering.",
  "Do NOT produce sexually explicit content involving minors.",
  "Do NOT provide disallowed age-restricted material to minors.",
  "Do NOT provide instructions that circumvent safety controls or filters.",
  "Do NOT provide instructions to produce or obtain controlled substances.",
  "Do NOT store or expose teacher/exam materials that should remain private.",
  "Do NOT assist in circumventing school disciplinary systems or surveillance.",
  "Do NOT produce content that encourages harassment or hateful violence.",
  "When refusing, always offer constructive alternatives: hints, scaffolding, or stepwise guidance.",
  "Follow any additional rules the school admin sets."
];

/** Simple intent-only forbidden patterns (only applied to user intent/message, NOT essay contents) */
const FORBIDDEN_PATTERNS = [
  { re: /\b(do my homework|write my essay|take my test|do my assignment|complete my assignment)\b/i, reason: 'academic dishonesty' },
  { re: /\b(cheat|cheating|exam answers|test answers)\b/i, reason: 'cheating' },
  { re: /\b(bomb|explosive|detonate|make a bomb)\b/i, reason: 'weapons/explosives' },
  { re: /\b(how to hack|hack into|break into|steal password)\b/i, reason: 'illegal hacking' },
  { re: /\b(play (games|minecraft|roblox)|start a game for me)\b/i, reason: 'playing games / disruption' }
];

function checkViolations(text) {
  const hits = [];
  if (!text) return hits;
  for (const p of FORBIDDEN_PATTERNS) {
    if (p.re.test(text)) hits.push(p.reason);
  }
  return hits;
}

/** Build a system prompt that gives the assistant role + rules + grade guidance */
function gradeRubricSummary(gradeRaw) {
  if (!gradeRaw) return 'Grade not specified: default to general K-12 expectations (scaffolded suggestions).';
  const s = String(gradeRaw).toLowerCase();
  const m = s.match(/(\d{1,2})/);
  if (m) {
    const g = Number(m[1]);
    if (g <= 5) return 'Elementary (K-5): focus on topic sentences, short paragraphs, spelling/grammar, and clear main idea.';
    if (g <= 8) return 'Middle (6-8): focus on thesis, paragraph structure with evidence, basic analysis and transitions.';
    return 'High school (9-12): focus on thesis clarity, evidence integration, analysis depth, organization, and tone.';
  }
  if (s.includes('elementary') || s.includes('k')) return 'Elementary (K-5): focus on topic sentences, spelling/grammar.';
  if (s.includes('middle')) return 'Middle school (6-8): focus on thesis and paragraph development.';
  if (s.includes('high')) return 'High school (9-12): focus on argument strength and evidence.';
  return 'General K-12 expectations: scaffolded suggestions.';
}

function buildSystemPrompt(gradeRaw) {
  const rulesText = BASE_RULES.map((r,i) => `${i+1}. ${r}`).join('\n');
  const gradeSummary = gradeRubricSummary(gradeRaw);
  return `You are an educational assistant for a K-12 school. Follow these rules and refuse requests that break them.
Rules:
${rulesText}

Grade guidance: ${gradeSummary}

Be concise, grade-appropriate, and do NOT generate full student answers for homework or tests. When refusing, explain why and offer helpful alternatives: hints, scaffolded steps, teacher questions. Do NOT reveal internal chain-of-thought; use reasoning internally but only output the final guidance.`;
}

/** Call Gemini (replace endpoint/body shape as required by your Gemini endpoint) */
async function callGemini(prompt, opts = {}) {
  if (!GEMINI_API_KEY) return 'No GEMINI_API_KEY configured on the server.';
  try {
    const res = await fetch(DEFAULT_GEMINI_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${GEMINI_API_KEY}`
      },
      body: JSON.stringify({
        prompt,
        temperature: opts.temperature ?? 0.2,
        max_tokens: opts.maxTokens ?? 1024
      })
    });
    const j = await res.json();
    if (j.output?.text) return String(j.output.text);
    if (j.choices && j.choices[0] && j.choices[0].text) return String(j.choices[0].text);
    if (j.text) return String(j.text);
    return JSON.stringify(j).slice(0, 2000);
  } catch (e) {
    console.error('Gemini call error', e);
    return `LLM error: ${String(e?.message || e)}`;
  }
}

/** Handler */
module.exports = async (req, res) => {
  try {
    if (req.method === 'OPTIONS') {
      res.setHeader('Access-Control-Allow-Origin', '*');
      res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
      res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
      return res.status(200).end();
    }
    if (req.method !== 'POST') return res.status(405).json({ error: 'POST required' });

    const body = req.body || {};
    // messages: array of { role: 'user'|'assistant', content: '...' }
    // frontend will manage conversation and pass full messages each time (stateless server)
    const messages = Array.isArray(body.messages) ? body.messages : [];
    const action = body.action || 'chat'; // 'chat' or 'analyze_essay'
    const grade = body.grade || null;
    const essay = typeof body.essay === 'string' ? body.essay : '';

    // Intent-check: only check user's latest explicit message (not the essay)
    // Find last user message in messages array (if provided)
    const lastUserMsg = (() => {
      for (let i = messages.length - 1; i >= 0; i--) {
        if (messages[i].role === 'user') return String(messages[i].content || '');
      }
      return String(body.message || '');
    })();

    const violations = checkViolations(lastUserMsg + '\n' + (body.url || ''));
    if (violations.length > 0) {
      const reply = `I can't help with that because it conflicts with platform rules (${violations.join(', ')}). I can, however, provide hints, scaffolded steps, or teaching guidance.`;
      return res.status(200).json({ refused: true, reason: violations, reply });
    }

    // Soft essay scan - don't refuse; just warn (server does not store)
    let essayWarning = null;
    if (essay && essay.length > 0) {
      const essayViol = checkViolations(essay);
      if (essayViol.length > 0) {
        essayWarning = `Note: the essay text contains phrases that often indicate disallowed intent (${essayViol.join(', ')}). Proceeding with analysis but teacher review recommended.`;
      }
    }

    // Build prompt
    let prompt = '';
    const system = buildSystemPrompt(grade);
    if (action === 'analyze_essay') {
      const convo = messages.map(m => `${m.role === 'user' ? 'User' : 'Assistant'}: ${String(m.content)}`).join('\n');
      prompt = `${system}\n\nConversation (recent):\n${convo}\n\nTask: Analyze the student's essay below. Provide:\n1) A 2-3 sentence summary.\n2) Strengths.\n3) Weaknesses (structure, thesis, evidence, clarity).\n4) Concrete, numbered revision steps tailored to grade ${grade || 'unspecified'}.\n5) 3-5 teacher questions to guide revision.\n6) DO NOT rewrite or complete the essay; provide scaffolding instead.\n\nEssay:\n${essay.slice(0, 30000)}`;
    } else {
      // chat: pass conversation and answer
      const convo = messages.map(m => `${m.role === 'user' ? 'User' : 'Assistant'}: ${String(m.content)}`).join('\n');
      prompt = `${system}\n\nConversation:\n${convo}\n\nAssistant:`;
    }

    // Call Gemini
    const usedReasoning = true;
    const aiResp = await callGemini(prompt, { temperature: 0.15, maxTokens: usedReasoning ? 1200 : 600 });

    // Return result (server does NOT persist essays)
    return res.status(200).json({
      reply: aiResp,
      essayWarning,
      usedReasoning,
      usedSearch: false, // we intentionally don't call any external search here
      refused: false
    });
  } catch (err) {
    console.error('Handler error', err);
    return res.status(500).json({ error: String(err?.message || err) });
  }
};
