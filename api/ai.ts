// api/ai.ts
/**
 * Vercel serverless handler (TypeScript).
 *
 * Features:
 *  - Gemini 2.5 Flash-Lite proxy (set GEMINI_API_KEY and optionally GEMINI_API_URL)
 *  - Brave Search integration (BRAVE_SUBSCRIPTION_TOKEN) used only when needed
 *  - "Reasoning-on-demand" heuristics (useReasoning flag or automatic detection)
 *  - Persistent memory (Redis via REDIS_URL) or in-memory fallback
 *  - Context caching to reduce re-work
 *  - Safer violation checking: only checks user message + url (not essay text)
 *  - Soft warning if essay contains suspicious phrases (teacher review)
 *  - Strong base rules (20 items) + EXTRA_RULES env support
 *  - Grade-aware rubric guidance: pass `grade` or `gradeLevel` in request body
 *
 * Required env vars:
 *   - GEMINI_API_KEY (required to call Gemini)
 *   - GEMINI_API_URL (optional override for Gemini endpoint)
 * Optional env vars:
 *   - BRAVE_SUBSCRIPTION_TOKEN (Brave Search key)
 *   - REDIS_URL (redis://... for persistence)
 *   - EXTRA_RULES (JSON array string or newline-separated additional rules)
 *   - SAVE_COOKIES=1 (if you want server to store cookies in memory/redis when used)
 *
 * Request body JSON (examples):
 *  - Chat: { userId, message }
 *  - Essay analysis: { userId, action: "analyze_essay", essay, grade: "9" }
 *  - Force flags: { userId, message, useBraveSearch: true, useReasoning: true }
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';

// Optional deps — try require to allow running even if not installed
let Redis: any = null;
let LRU: any = null;
try {
  // If you want Redis / LRU, install: npm i ioredis lru-cache
  Redis = require('ioredis');
  LRU = require('lru-cache');
} catch (e) {
  // fallback to in-memory implementations below
}

// Config
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || '';
const DEFAULT_GEMINI_URL = process.env.GEMINI_API_URL || 'https://api.example-gemini/v2.5flash-lite';
const BRAVE_TOKEN = process.env.BRAVE_SUBSCRIPTION_TOKEN || '';
const USE_SAVE_COOKIES = process.env.SAVE_COOKIES === '1';

// --------- Base rules (20) + extras ----------
const BASE_RULES = [
  "Do NOT complete homework, tests, quizzes, or graded assignments for students.",
  "Do NOT provide answers that enable cheating, plagiarism, or academic dishonesty.",
  "Do NOT provide step-by-step instructions to commit illegal acts.",
  "Do NOT assist in making weapons, explosives, or harmful contraptions.",
  "Do NOT help write malware, spyware, or instructions for unauthorized access.",
  "Do NOT provide instructions or encouragement for intentionally disrupting classes or schools.",
  "Do NOT impersonate teachers, staff, or other students.",
  "Do NOT reveal or attempt to extract student personal data or private information.",
  "Do NOT provide medical, legal or psychiatric professional advice as a substitute for professionals.",
  "Do NOT provide instructions that meaningfully facilitate self-harm or suicide.",
  "Do NOT facilitate identity fraud, phishing, or social engineering.",
  "Do NOT produce sexually explicit content involving minors or facilitate sexual exploitation.",
  "Do NOT provide disallowed age-restricted material to minors.",
  "Do NOT provide instructions that circumvent safety controls or filters.",
  "Do NOT provide instructions to produce or obtain controlled substances.",
  "Do NOT store or expose teacher/exam materials that should remain private.",
  "Do NOT assist in circumventing school disciplinary systems or surveillance.",
  "Do NOT produce content that encourages harassment, hateful violence, or harassment of individuals.",
  "When refusing, always offer constructive alternatives: hints, scaffolding, or stepwise guidance.",
  "Follow any additional school-provided rules from environment configuration (EXTRA_RULES)."
];

function loadExtraRules(): string[] {
  const out: string[] = [];
  if (process.env.EXTRA_RULES) {
    try {
      const parsed = JSON.parse(process.env.EXTRA_RULES);
      if (Array.isArray(parsed)) parsed.forEach((r) => out.push(String(r)));
    } catch {
      (process.env.EXTRA_RULES || '').split(/\r?\n/).map(s => s.trim()).filter(Boolean).forEach(s => out.push(s));
    }
  }
  return out;
}
const EXTRA_RULES = loadExtraRules();
const RULES = BASE_RULES.concat(EXTRA_RULES);

// --------- Memory & Cache (Redis optional) ----------
const useRedis = Boolean(process.env.REDIS_URL && Redis);
let redis: any = null;
if (useRedis) {
  try { redis = new Redis(process.env.REDIS_URL); } catch (e) { console.warn('Redis init failed, falling back to memory.'); redis = null; }
}
const IN_MEMORY_MEMORY = new Map<string, any[]>();
const MAX_MESSAGES_PER_USER = 150;

let contextCache: any;
if (LRU) contextCache = new LRU({ max: 1000, ttl: 1000 * 60 * 10 });
else {
  const map = new Map<string, { value: any; expiresAt: number }>();
  contextCache = {
    get(k: string) {
      const e = map.get(k);
      if (!e) return undefined;
      if (Date.now() > e.expiresAt) { map.delete(k); return undefined; }
      return e.value;
    },
    set(k: string, v: any, ttl = 1000 * 60 * 10) {
      map.set(k, { value: v, expiresAt: Date.now() + ttl });
    }
  };
}

async function appendMemory(userId: string, role: 'user' | 'assistant', content: string) {
  if (redis) {
    const key = `mem:${userId}`;
    await redis.rpush(key, JSON.stringify({ role, content, ts: Date.now() }));
    await redis.ltrim(key, -MAX_MESSAGES_PER_USER, -1);
    await redis.expire(key, 60 * 60 * 24 * 30);
    return;
  }
  const arr = IN_MEMORY_MEMORY.get(userId) || [];
  arr.push({ role, content, ts: Date.now() });
  if (arr.length > MAX_MESSAGES_PER_USER) arr.splice(0, arr.length - MAX_MESSAGES_PER_USER);
  IN_MEMORY_MEMORY.set(userId, arr);
}

async function readMemory(userId: string) {
  if (redis) {
    const key = `mem:${userId}`;
    const list = await redis.lrange(key, 0, -1);
    return list.map((s: string) => JSON.parse(s));
  }
  return IN_MEMORY_MEMORY.get(userId) || [];
}

// --------- Simple violation checker (message/url only) ----------
const FORBIDDEN_PATTERNS: { re: RegExp; reason: string }[] = [
  { re: /\b(do my homework|write my essay|take my test|do my assignment|complete my assignment)\b/i, reason: 'academic dishonesty' },
  { re: /\b(cheat|cheating|exam answers|test answers)\b/i, reason: 'cheating' },
  { re: /\b(bomb|explosive|detonate|make a bomb)\b/i, reason: 'weapons/explosives' },
  { re: /\b(how to hack|hack into|break into)\b/i, reason: 'illegal hacking' },
  { re: /\b(play (games|minecraft|roblox)|start a game for me)\b/i, reason: 'playing games / disruption' },
];

function checkViolations(text: string) {
  const hits: string[] = [];
  for (const p of FORBIDDEN_PATTERNS) if (p.re.test(text)) hits.push(p.reason);
  return hits;
}

// --------- Brave Search helper (called only when needed) ----------
async function braveWebSearch(query: string, size = 6) {
  if (!BRAVE_TOKEN) throw new Error('BRAVE_SUBSCRIPTION_TOKEN not configured');
  const url = `https://api.search.brave.com/res/v1/web/search?q=${encodeURIComponent(query)}&size=${size}`;
  const r = await fetch(url, { headers: { 'X-Subscription-Token': BRAVE_TOKEN, 'Accept': 'application/json' } });
  if (!r.ok) {
    const body = await r.text();
    throw new Error(`Brave Search error ${r.status}: ${body}`);
  }
  const json = await r.json();
  const limit = r.headers.get('X-RateLimit-Limit');
  return { json, limit };
}

// --------- Gemini wrapper ----------
async function callGemini(prompt: string, opts: { temperature?: number; maxTokens?: number } = {}) {
  if (!GEMINI_API_KEY) return 'No GEMINI_API_KEY configured.';
  try {
    const r = await fetch(DEFAULT_GEMINI_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${GEMINI_API_KEY}` },
      body: JSON.stringify({
        prompt,
        temperature: opts.temperature ?? 0.2,
        max_tokens: opts.maxTokens ?? 1024
      })
    });
    const j = await r.json();
    if (j.output?.text) return String(j.output.text);
    if (j.choices && j.choices[0] && j.choices[0].text) return String(j.choices[0].text);
    if (j.text) return String(j.text);
    return JSON.stringify(j).slice(0, 2000);
  } catch (e: any) {
    return `LLM call error: ${String(e?.message || e)}`;
  }
}

// --------- Heuristics (search + reasoning) ----------
function needsSearchHeuristic(body: any) {
  if (body.useBraveSearch === true) return true;
  if (typeof body.useBraveSearch === 'string' && ['1', 'true', 'yes'].includes(body.useBraveSearch)) return true;
  if (/\b(search for|find|sources|verify|ground|cite|references?)\b/i.test(String(body.message || ''))) return true;
  if (body.url) return true;
  return false;
}
function needsReasoningHeuristic(body: any) {
  if (body.useReasoning === true) return true;
  if (typeof body.useReasoning === 'string' && ['1', 'true', 'yes'].includes(body.useReasoning)) return true;
  if (body.action === 'analyze_essay') return true;
  const content = String(body.message || body.essay || '');
  if (content.length > 800) return true;
  if (/\b(analyze|critique|evaluate|grade|feedback|rubric)\b/i.test(content)) return true;
  return false;
}

// --------- Grade rubric helper ----------
function normalizeGrade(g: any) {
  if (!g) return null;
  const s = String(g).trim().toLowerCase();
  // try to extract numeric grade
  const m = s.match(/(\d{1,2})/);
  if (m) return parseInt(m[1], 10);
  if (s.includes('k')) return 0;
  if (s.includes('elementary')) return 3;
  if (s.includes('middle')) return 7;
  if (s.includes('high')) return 10;
  return null;
}
function gradeRubricSummary(gradeRaw: any) {
  const grade = normalizeGrade(gradeRaw);
  if (grade === null) {
    return 'Grade not specified: default to general K-12 expectations (be conservative; prefer simpler language and scaffolded suggestions).';
  }
  if (grade === 0 || grade <= 5) {
    return 'Elementary (K-5) rubric: focus on clear topic sentences, basic paragraph structure, short sentences, simple transitions, spelling/grammar, and clear main idea. Suggestions should be concrete and step-by-step.';
  }
  if (grade >= 6 && grade <= 8) {
    return 'Middle school (6-8) rubric: expect clearer thesis statements, paragraphs with evidence and explanation, improved transitions, basic citation/attribution, and logical flow. Suggest revisions that teach paragraph development and evidence explanation.';
  }
  if (grade >= 9 && grade <= 12) {
    return 'High school (9-12) rubric: expect a clear thesis, structured paragraphs with claims and textual evidence, deeper analysis, varied sentence structure, and appropriate tone. Suggest revisions for argument strength, evidence integration, sophistication of analysis, and organization.';
  }
  // default for higher levels
  return 'Grade level indicates advanced expectations: use college-prep rubric focusing on argument coherence, evidence evaluation, and polished style.';
}

// --------- Prompt builders (system prompt includes role + rules + grade guidance) ----------
function buildSystemPrompt(gradeRaw: any) {
  const rulesText = RULES.map((r, i) => `${i+1}. ${r}`).join('\n');
  const gradeSummary = gradeRubricSummary(gradeRaw);
  return `You are an educational assistant for a K-12 school platform. Your job:
- Provide helpful, scaffolded feedback and lessons for students and teachers.
- Focus only on the educational task at hand (essay analysis, tutoring, explanations).
- Do NOT discuss unrelated subjects unless explicitly required by the essay topic and it is age-appropriate.
- Respect and enforce the following rules (refuse actions that violate them and always offer helpful alternatives such as hints, scaffolding, and stepwise guidance):

${rulesText}

Grade guidance (use to adapt tone and difficulty): ${gradeSummary}

When you refuse to do something because it violates rules, briefly explain why, and give practical, educational alternatives (explain steps, provide questions to guide the student, or give partial scaffolding). Do NOT reveal internal chain-of-thought reasoning. Use reasoning internally to produce well-structured outputs, but return only the user-facing guidance.`;
}

async function buildConversationContext(userId: string, extra = '') {
  const mem = await readMemory(userId);
  const recent = mem.slice(-40).map(m => `${m.role === 'user' ? 'User' : 'Assistant'}: ${m.content}`).join('\n');
  return `${buildSystemPrompt(null)}\n\nRecent conversation (most recent last):\n${recent}\n\n${extra}`;
}

// --------- Handler ----------
export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'POST only' });
    return;
  }

  try {
    const body = req.body || {};
    const userId: string = String(body.userId || body.user || 'anon');
    const action: string = body.action || 'chat'; // 'chat' | 'analyze_essay' | 'search_and_analyze'
    const message: string = String(body.message || '');
    const essay: string = String(body.essay || '');
    const url: string | undefined = body.url;
    const gradeRaw = body.grade || body.gradeLevel || body.gradeLevelRaw || null;

    // Violation check: only inspect user's explicit message and URL (not essay)
    const checkText = `${message || ''}\n${url || ''}`.trim();
    const violations = checkViolations(checkText);
    if (violations.length > 0) {
      const reply = `I can't help with that because it conflicts with platform rules (${violations.join(', ')}). I can help with hints, scaffolding, or step-by-step explanations instead.`;
      await appendMemory(userId, 'assistant', reply);
      return res.status(200).json({ refused: true, reason: violations, reply });
    }

    // Soft essay warning (essay body scanned but doesn't auto-refuse)
    let essayWarning: string | null = null;
    if (essay && essay.length > 0) {
      const essayViol = checkViolations(essay);
      if (essayViol.length > 0) {
        essayWarning = `Note: essay text contains potentially concerning phrases (${essayViol.join(', ')}). Proceeding with analysis, but teacher review recommended.`;
        await appendMemory(userId, 'assistant', essayWarning);
      }
    }

    // Decide heuristics
    const doSearch = needsSearchHeuristic(body);
    const doReasoning = needsReasoningHeuristic(body);

    // Brave search (only if needed and token present)
    let searchResults: any = null;
    if (doSearch && BRAVE_TOKEN) {
      try {
        const query = body.searchQuery || (essay ? essay.slice(0, 300) : message.slice(0, 300));
        const key = `brave:${query}`;
        const cached = contextCache.get(key);
        if (cached) searchResults = cached;
        else {
          const sr = await braveWebSearch(query, 3);
          contextCache.set(key, sr, 1000 * 60 * 5);
          searchResults = sr;
        }
      } catch (e: any) {
        console.warn('Brave search error', e?.message || e);
        searchResults = { error: String(e?.message || e) };
      }
    }

    // Build prompt context (include grade guidance specifically)
    const gradeGuidance = gradeRubricSummary(gradeRaw);
    const extraContext = [
      doSearch ? `Search results (top): ${JSON.stringify(searchResults?.json?.web?.results?.slice(0,3) || []).slice(0,8000)}` : '',
      `Grade guidance: ${gradeGuidance}`
    ].filter(Boolean).join('\n');

    const ctx = await buildConversationContext(userId, extraContext);
    // Compose task-specific prompt
    let prompt = '';
    if (action === 'analyze_essay') {
      const essayText = essay || message;
      prompt = `${buildSystemPrompt(gradeRaw)}\n\nConversation summary:\n${ctx}\n\nTask: Analyze the student's essay below and provide:\n1) 2-3 sentence summary\n2) Strengths\n3) Weaknesses (structure, thesis, evidence, clarity)\n4) Concrete, numbered revision steps tailored to the student's grade level (${String(gradeRaw || 'unspecified')}). Make sure suggestions match typical rubrics for that grade.\n5) A short list of scaffolded teacher questions the student can answer to improve.\n6) DO NOT rewrite the essay; provide scaffolding and sample sentences only if explicitly requested.\n\nEssay:\n${essayText.slice(0, 30000)}`;
    } else if (action === 'search_and_analyze') {
      prompt = `${buildSystemPrompt(gradeRaw)}\n\nContext:\n${ctx}\n\nUsing the search grounding above, summarize the main points and produce teacher-friendly and student-friendly guidance tailored to grade ${String(gradeRaw || 'unspecified')}.`;
    } else {
      // general chat
      const internalNote = doReasoning ? '/* Use internal reasoning to produce a high-quality, grade-appropriate answer. Do not reveal chain-of-thought. */' : '';
      prompt = `${buildSystemPrompt(gradeRaw)}\n\n${internalNote}\nConversation:\n${ctx}\nUser: ${message}\nAssistant:`;
    }

    // Call Gemini
    const tokenBudget = doReasoning ? 1200 : 600;
    const temp = doReasoning ? 0.15 : 0.2;
    // record user message into memory (short)
    await appendMemory(userId, 'user', message || essay || (body.searchQuery || ''));

    const aiResponse = await callGemini(prompt, { temperature: temp, maxTokens: tokenBudget });

    // Save AI response to memory
    await appendMemory(userId, 'assistant', aiResponse);

    // Return response with metadata (essayWarning if any)
    return res.status(200).json({
      reply: aiResponse,
      usedSearch: !!doSearch && !!BRAVE_TOKEN,
      usedReasoning: !!doReasoning,
      searchResults: doSearch ? (searchResults?.json || searchResults) : undefined,
      essayWarning
    });
  } catch (err: any) {
    console.error('Handler error', err);
    return res.status(500).json({ error: String(err?.message || err) });
  }
}
