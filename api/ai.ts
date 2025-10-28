// server.ts
/**
 * School AI backend (Express, TypeScript) - single file
 *
 * Features:
 *  - Gemini 2.5 Flash-Lite integration (GEMINI_API_KEY + GEMINI_API_URL)
 *  - Brave Search integration (BRAVE_SUBSCRIPTION_TOKEN) used only when heuristics demand it
 *  - Optional headless browsing via puppeteer-core + BRAVE_PATH (best-effort; may fail on some hosts)
 *  - Per-user persistent memory with Redis (REDIS_URL) or in-memory fallback
 *  - LRU context cache to avoid redundant heavy ops
 *  - Grade-aware rubric adaptation (grade / gradeLevel)
 *  - Reasoning-on-demand (heuristics or explicit flag)
 *  - Safety rules (20 base rules + EXTRA_RULES from env). Intent-check applies to user message + URL only.
 *  - Soft warning if essay contains suspicious phrases (essay not auto-refused)
 *  - Endpoints: POST /ai (main), GET /health
 *
 * Usage:
 * 1) Install deps (recommended):
 *    npm install express ioredis lru-cache puppeteer-core
 *
 *    Note: puppeteer-core is optional. If you don't want browsing, skip installing it.
 *
 * 2) Set env vars:
 *    - GEMINI_API_KEY (required to call Gemini)
 *    - GEMINI_API_URL (optional override - provider endpoint)
 *    - BRAVE_SUBSCRIPTION_TOKEN (optional; required to use Brave Search)
 *    - REDIS_URL (optional; example: redis://:password@host:6379/0)
 *    - EXTRA_RULES (optional JSON array or newline-separated string)
 *    - BRAVE_PATH (optional - path to Brave/Chromium executable for puppeteer-core)
 *    - SAVE_COOKIES=1 (optional; stores cookie metadata in memory/Redis)
 *
 * 3) Run:
 *    npx ts-node server.ts   (if ts-node installed)
 *    or compile to JS and run `node server.js`
 *
 * Endpoint:
 *   POST /ai with JSON body { userId, action?, message?, essay?, grade?, useBraveSearch?, useReasoning?, url?, cookies? }
 */

import express from 'express';
import process from 'process';
import crypto from 'crypto';

let RedisClient: any = null;
let LRU: any = null;
let puppeteer: any = null;

// Try to require optional libs (so file still runs if they aren't installed)
try {
  // Use require() for optional imports to avoid TS compile-time errors if packages missing
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  RedisClient = require('ioredis');
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  LRU = require('lru-cache');
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  puppeteer = require('puppeteer-core');
} catch (e) {
  // optional libs not installed — code will fall back to in-memory or no browsing
  // console.warn('Optional libs not loaded:', e?.message || e);
}

const app = express();
app.use(express.json({ limit: '1mb' })); // avoid giant uploads
const PORT = Number(process.env.PORT || 3000);

// Config / env
const GEMINI_API_KEY = String(process.env.GEMINI_API_KEY || '');
const DEFAULT_GEMINI_URL = String(process.env.GEMINI_API_URL || 'https://api.example-gemini/v2.5flash-lite'); // replace with real endpoint
const BRAVE_TOKEN = String(process.env.BRAVE_SUBSCRIPTION_TOKEN || '');
const BRAVE_PATH = String(process.env.BRAVE_PATH || '');
const SAVE_COOKIES = process.env.SAVE_COOKIES === '1';

// ------------------------------
// Base rules (20) + extra rules
// ------------------------------
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
  "Do NOT produce content that encourages harassment, hateful violence, or malicious harassment of individuals.",
  "When refusing, always offer constructive alternatives: hints, scaffolding, or stepwise guidance.",
  "Follow any additional school-provided rules from environment configuration (EXTRA_RULES)."
];

function loadExtraRules(): string[] {
  const out: string[] = [];
  const env = process.env.EXTRA_RULES;
  if (!env) return out;
  try {
    const parsed = JSON.parse(env);
    if (Array.isArray(parsed)) {
      parsed.forEach((r) => out.push(String(r)));
      return out;
    }
  } catch {
    // not valid JSON -> split by newline
  }
  env.split(/\r?\n/).map(s => s.trim()).filter(Boolean).forEach(s => out.push(s));
  return out;
}
const EXTRA_RULES = loadExtraRules();
const RULES = BASE_RULES.concat(EXTRA_RULES);

// ------------------------------
// Memory & cache (Redis optional)
// ------------------------------
const useRedis = Boolean(process.env.REDIS_URL && RedisClient);
let redis: any = null;
if (useRedis) {
  try {
    redis = new RedisClient(process.env.REDIS_URL);
    redis.on('error', (err: any) => console.warn('Redis error', err?.message || err));
  } catch (e) {
    console.warn('Redis init failed, falling back to memory. Error:', e?.message || e);
    redis = null;
  }
}

// In-memory memory fallback (not persistent across restarts)
const IN_MEMORY_MEMORY = new Map<string, Array<{ role: string; content: string; ts: number }>>();
const MAX_MESSAGES_PER_USER = 200;

async function appendMemory(userId: string, role: 'user' | 'assistant', content: string) {
  if (redis) {
    const key = `mem:${userId}`;
    try {
      await redis.rpush(key, JSON.stringify({ role, content, ts: Date.now() }));
      await redis.ltrim(key, -MAX_MESSAGES_PER_USER, -1);
      await redis.expire(key, 60 * 60 * 24 * 30); // 30 days TTL
      return;
    } catch (e) {
      console.warn('Redis appendMemory error', e?.message || e);
    }
  }
  const arr = IN_MEMORY_MEMORY.get(userId) || [];
  arr.push({ role, content, ts: Date.now() });
  if (arr.length > MAX_MESSAGES_PER_USER) arr.splice(0, arr.length - MAX_MESSAGES_PER_USER);
  IN_MEMORY_MEMORY.set(userId, arr);
}

async function readMemory(userId: string) {
  if (redis) {
    const key = `mem:${userId}`;
    try {
      const list = await redis.lrange(key, 0, -1);
      return list.map((s: string) => {
        try { return JSON.parse(s); } catch { return { role: 'unknown', content: s }; }
      });
    } catch (e) {
      console.warn('Redis readMemory error', e?.message || e);
    }
  }
  return IN_MEMORY_MEMORY.get(userId) || [];
}

// ------------------------------
// Context cache (LRU) to avoid repeated heavy ops
// ------------------------------
let contextCache: any;
if (LRU) {
  contextCache = new LRU({ max: 2000, ttl: 1000 * 60 * 10 }); // 10 min
} else {
  const map = new Map<string, { value: any; expiresAt: number }>();
  contextCache = {
    get: (k: string) => {
      const e = map.get(k);
      if (!e) return undefined;
      if (Date.now() > e.expiresAt) { map.delete(k); return undefined; }
      return e.value;
    },
    set: (k: string, v: any, ttl = 1000 * 60 * 10) => {
      map.set(k, { value: v, expiresAt: Date.now() + ttl });
    },
  };
}

// ------------------------------
// Violation detection (intent only: message + url)
// ------------------------------
const FORBIDDEN_PATTERNS: { re: RegExp; reason: string }[] = [
  { re: /\b(do my homework|write my essay|take my test|do my assignment|complete my assignment)\b/i, reason: 'academic dishonesty' },
  { re: /\b(cheat|cheating|exam answers|test answers|answer key)\b/i, reason: 'cheating' },
  { re: /\b(bomb|explosive|detonate|make a bomb)\b/i, reason: 'weapons/explosives' },
  { re: /\b(how to hack|hack into|break into|steal password)\b/i, reason: 'illegal hacking' },
  { re: /\b(play (games|minecraft|roblox)|start a game for me)\b/i, reason: 'playing games / disruption' },
];

function checkViolations(text: string) {
  const hits: string[] = [];
  for (const p of FORBIDDEN_PATTERNS) {
    if (p.re.test(text)) hits.push(p.reason);
  }
  return hits;
}

// ------------------------------
// Heuristics: when to search & when to use reasoning
// ------------------------------
function needsSearchHeuristic(body: any) {
  if (body.useBraveSearch === true) return true;
  if (typeof body.useBraveSearch === 'string' && ['1', 'true', 'yes'].includes(body.useBraveSearch)) return true;
  const q = String(body.message || body.essay || '');
  if (/\b(search for|find|sources|verify|ground|cite|references?)\b/i.test(q)) return true;
  if (body.url) return true;
  return false;
}
function needsReasoningHeuristic(body: any) {
  if (body.useReasoning === true) return true;
  if (typeof body.useReasoning === 'string' && ['1', 'true', 'yes'].includes(body.useReasoning)) return true;
  if (body.action === 'analyze_essay') return true;
  const content = String(body.message || body.essay || '');
  if (content.length > 800) return true;
  if (/\b(analyze|critique|evaluate|grade|feedback|rubric|thesis)\b/i.test(content)) return true;
  return false;
}

// ------------------------------
// Brave Search helper (server-side web search)
// ------------------------------
async function braveWebSearch(query: string, size = 6) {
  if (!BRAVE_TOKEN) throw new Error('BRAVE_SUBSCRIPTION_TOKEN not configured');
  const url = `https://api.search.brave.com/res/v1/web/search?q=${encodeURIComponent(query)}&size=${size}`;
  const r = await fetch(url, { headers: { 'Accept': 'application/json', 'X-Subscription-Token': BRAVE_TOKEN } });
  if (!r.ok) {
    const body = await r.text();
    throw new Error(`Brave Search error ${r.status}: ${body}`);
  }
  const j = await r.json();
  return j;
}

// ------------------------------
// Optional: Puppeteer/Brave browsing (best-effort)
// ------------------------------
async function fetchPageWithPuppeteer(url: string, cookieHeader?: string) {
  // Use puppeteer-core + BRAVE_PATH if available. If not, fallback to fetch().
  if (puppeteer && BRAVE_PATH) {
    try {
      const browser = await puppeteer.launch({
        executablePath: BRAVE_PATH,
        headless: true,
        args: ['--no-sandbox', '--disable-setuid-sandbox'],
      });
      const page = await browser.newPage();
      if (cookieHeader) {
        await page.setExtraHTTPHeaders({ cookie: cookieHeader });
      }
      await page.goto(url, { waitUntil: 'networkidle2', timeout: 30_000 });
      const html = await page.content();
      const cookies = await page.cookies();
      await browser.close();
      return { html, cookies };
    } catch (e) {
      // If puppeteer fails, fallback to fetch below
      console.warn('Puppeteer browse failed:', e?.message || e);
    }
  }
  // fallback to server-side fetch (won't run JS)
  const headers: any = {};
  if (cookieHeader) headers['cookie'] = cookieHeader;
  const r = await fetch(url, { headers });
  const html = await r.text();
  const setCookie = r.headers.get('set-cookie') || '';
  return { html, cookies: setCookie };
}

// ------------------------------
// Gemini wrapper (generic)
 // ------------------------------
async function callGemini(prompt: string, opts: { temperature?: number; maxTokens?: number } = {}) {
  if (!GEMINI_API_KEY) return 'No GEMINI_API_KEY configured.';
  try {
    const res = await fetch(DEFAULT_GEMINI_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${GEMINI_API_KEY}`,
      },
      body: JSON.stringify({
        prompt,
        temperature: opts.temperature ?? 0.2,
        max_tokens: opts.maxTokens ?? 1024,
      }),
    });
    const j = await res.json();
    // handle a few different provider shapes
    if (j.output?.text) return String(j.output.text);
    if (j.choices && j.choices[0] && j.choices[0].text) return String(j.choices[0].text);
    if (j.text) return String(j.text);
    return JSON.stringify(j).slice(0, 3000);
  } catch (e: any) {
    console.warn('callGemini error:', e?.message || e);
    return `LLM call error: ${String(e?.message || e)}`;
  }
}

// ------------------------------
// Grade rubric helper
// ------------------------------
function normalizeGrade(g: any) {
  if (!g) return null;
  const s = String(g).trim().toLowerCase();
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
    return 'Grade not specified: default to general K-12 expectations (prefer simpler language and scaffolded suggestions).';
  }
  if (grade === 0 || grade <= 5) {
    return 'Elementary (K-5) rubric: focus on topic sentences, basic paragraph structure, short sentences, spelling/grammar, and a clear main idea. Give concrete step-by-step suggestions.';
  }
  if (grade >= 6 && grade <= 8) {
    return 'Middle school (6-8): expect clearer thesis statements, paragraphs with evidence and explanation, basic citation/attribution, and logical flow. Provide guidance for paragraph development and evidence explanation.';
  }
  if (grade >= 9 && grade <= 12) {
    return 'High school (9-12): expect a clear thesis, structured paragraphs with claims and textual evidence, deeper analysis, varied sentence structure, and appropriate tone. Recommend revisions for argument strength, evidence integration, and organization.';
  }
  return 'Advanced: college-prep style — focus on argument coherence, evidence quality, and polished style.';
}

// ------------------------------
// System prompt builder (includes role, rules, grade guidance)
// ------------------------------
function buildSystemPrompt(gradeRaw: any) {
  const rulesText = RULES.map((r, i) => `${i + 1}. ${r}`).join('\n');
  const gradeSummary = gradeRubricSummary(gradeRaw);
  return `You are an educational assistant for a K-12 school platform.
Your role:
- Provide scaffolded, grade-appropriate feedback for students and teachers.
- Focus only on the educational task; avoid unrelated subjects unless explicitly requested and age-appropriate.
- Enforce the rules below; when refusing, explain briefly and offer helpful alternatives (hints, scaffolding, teacher questions).

Rules:
${rulesText}

Grade guidance: ${gradeSummary}

Do NOT reveal internal chain-of-thought. Use reasoning internally to improve answer quality, but output only the final, user-facing guidance. If a user requests forbidden content (against rules), refuse and provide safe alternatives.`;
}

async function buildConversationContext(userId: string, gradeRaw: any, extra = '') {
  const mem = await readMemory(userId);
  const recent = mem.slice(-40).map(m => `${m.role === 'user' ? 'User' : 'Assistant'}: ${m.content}`).join('\n');
  return `${buildSystemPrompt(gradeRaw)}\n\nRecent conversation (most recent last):\n${recent}\n\n${extra}`;
}

// ------------------------------
// Small helpers
// ------------------------------
function shortHash(s: string) {
  const h = crypto.createHash('sha1').update(s).digest('hex');
  return h.slice(0, 10);
}

// ------------------------------
// Main handler (single POST endpoint)
// ------------------------------
app.post('/ai', async (req, res) => {
  try {
    const body: any = req.body || {};
    const userId = String(body.userId || body.user || 'anon').slice(0, 200);
    const action: string = body.action || 'chat'; // chat | analyze_essay | search_and_analyze
    const message: string = String(body.message || '');
    const essay: string = String(body.essay || '');
    const gradeRaw = body.grade || body.gradeLevel || null;
    const url: string | undefined = body.url;
    const cookieHeader: string | undefined = body.cookies;

    // 1) Check intent-only violations (message + url only)
    const checkText = `${message || ''}\n${url || ''}`.trim();
    const violations = checkViolations(checkText);
    if (violations.length > 0) {
      const reply = `I can't help with that because it conflicts with platform rules (${violations.join(', ')}). I can help with hints, scaffolding, or step-by-step explanations instead.`;
      await appendMemory(userId, 'assistant', reply);
      return res.status(200).json({ refused: true, reason: violations, reply });
    }

    // 2) Soft essay scan (do not refuse; only log / warn)
    let essayWarning: string | null = null;
    if (essay && essay.length > 0) {
      const essayViol = checkViolations(essay);
      if (essayViol.length > 0) {
        essayWarning = `Essay contains potentially concerning phrases (${essayViol.join(', ')}). Proceeding with analysis; teacher review advised.`;
        // store a note in memory for teacher review
        await appendMemory(userId, 'assistant', essayWarning);
      }
    }

    // 3) Heuristics: do we need Brave Search and reasoning?
    const doSearch = needsSearchHeuristic(body);
    const doReasoning = needsReasoningHeuristic(body);

    // 4) Optionally perform Brave search (only when needed and token present)
    let searchResults: any = null;
    if (doSearch && BRAVE_TOKEN) {
      try {
        const q = body.searchQuery || (essay ? essay.slice(0, 300) : message.slice(0, 300));
        const cacheKey = `brave:${shortHash(q)}`;
        const cached = contextCache.get(cacheKey);
        if (cached) searchResults = cached;
        else {
          const sr = await braveWebSearch(q, 3);
          contextCache.set(cacheKey, sr, 1000 * 60 * 5);
          searchResults = sr;
        }
      } catch (e: any) {
        console.warn('Brave search error:', e?.message || e);
        searchResults = { error: String(e?.message || e) };
      }
    }

    // 5) If ask to browse a URL (and user provided cookies), fetch page (optional)
    let pageHtmlSnippet: string | undefined = undefined;
    if (body.action === 'browse_and_analyze' && body.url) {
      try {
        const cacheKey = `page:${shortHash(body.url + (cookieHeader || ''))}`;
        const cached = contextCache.get(cacheKey);
        if (cached) {
          pageHtmlSnippet = cached;
        } else {
          const { html, cookies } = await fetchPageWithPuppeteer(body.url, cookieHeader);
          const snippet = html.slice(0, 60_000);
          contextCache.set(cacheKey, snippet, 1000 * 60 * 10);
          pageHtmlSnippet = snippet;
          if (SAVE_COOKIES && cookies) {
            await appendMemory(userId, 'assistant', `Saved cookies for ${body.url}: ${JSON.stringify(cookies).slice(0, 400)}`);
          }
        }
      } catch (e: any) {
        console.warn('Page fetch failed:', e?.message || e);
      }
    }

    // 6) Build prompt context (include grade rubric and search grounding if present)
    const gradeGuidance = gradeRubricSummary(gradeRaw);
    const extraContextParts: string[] = [];
    if (searchResults && searchResults.json) {
      try {
        const hits = searchResults.json.web?.results?.slice(0, 3) || [];
        extraContextParts.push(`Search grounding (top results): ${JSON.stringify(hits).slice(0, 8000)}`);
      } catch { /* ignore */ }
    }
    if (pageHtmlSnippet) extraContextParts.push(`Page snippet: ${pageHtmlSnippet.slice(0, 20000)}`);
    extraContextParts.push(`Grade guidance: ${gradeGuidance}`);
    const extraContext = extraContextParts.join('\n\n');

    const context = await buildConversationContext(userId, gradeRaw, extraContext);

    // 7) Compose task-specific prompt
    let prompt = '';
    if (action === 'analyze_essay') {
      const essayText = essay || message;
      const cacheKey = `essay:${userId}:${shortHash(essayText)}`;
      const cached = contextCache.get(cacheKey);
      if (cached) {
        await appendMemory(userId, 'assistant', '(Returned cached essay analysis)');
        return res.status(200).json({ fromCache: true, analysis: cached, essayWarning });
      }

      prompt = `${buildSystemPrompt(gradeRaw)}\n\nConversation context:\n${context}\n\nTask: Analyze the student's essay below and produce:
1) 2-3 sentence summary
2) Strengths
3) Weaknesses (structure, thesis, evidence, style, grammar)
4) Concrete numbered revision steps tailored to the student's grade level (${String(gradeRaw || 'unspecified')})
5) Teacher prompts/questions to guide revision (3-6)
6) Do NOT rewrite the essay or complete the student's assignment. Provide scaffolding, examples, and sample sentence frames only when helpful and explicitly requested.

Essay:
${essayText.slice(0, 30000)}`;
      // Use reasoning for essay analyses by default
    } else if (action === 'search_and_analyze') {
      prompt = `${buildSystemPrompt(gradeRaw)}\n\nContext:\n${context}\n\nTask: Using the search grounding, produce a clear, grade-appropriate summary and student-facing explanation, plus teacher notes and suggested next steps.`;
    } else if (action === 'browse_and_analyze') {
      prompt = `${buildSystemPrompt(gradeRaw)}\n\nContext:\n${context}\n\nTask: Analyze the page content (above) and produce a student-facing summary, teacher notes, and scaffolded revision steps.`;
    } else {
      // general chat
      const internalNote = doReasoning ? '/* Use internal reasoning to produce a more careful, high-quality response. Do not reveal chain-of-thought. */' : '';
      prompt = `${buildSystemPrompt(gradeRaw)}\n\n${internalNote}\nConversation context:\n${context}\n\nUser: ${message}\nAssistant:`;
    }

    // 8) Determine token budget & temp
    const tokenBudget = doReasoning ? 1200 : 600;
    const temperature = doReasoning ? 0.15 : 0.2;

    // Record user message to memory
    await appendMemory(userId, 'user', message || essay || (body.searchQuery || ''));

    // 9) Call Gemini LLM
    const aiOutput = await callGemini(prompt, { temperature, maxTokens: tokenBudget });

    // 10) Cache essay analysis if appropriate
    if (action === 'analyze_essay' && essay && essay.length > 10) {
      const cacheKey = `essay:${userId}:${shortHash(essay)}`;
      contextCache.set(cacheKey, aiOutput, 1000 * 60 * 10);
    }

    // 11) Save AI response into memory
    await appendMemory(userId, 'assistant', aiOutput);

    // 12) Response
    return res.status(200).json({
      reply: aiOutput,
      usedSearch: !!doSearch && !!BRAVE_TOKEN,
      usedReasoning: !!doReasoning,
      searchResults: doSearch ? (searchResults?.json || searchResults) : undefined,
      essayWarning
    });
  } catch (err: any) {
    console.error('Handler error:', err?.message || err);
    return res.status(500).json({ error: String(err?.message || err) });
  }
});

// Health endpoint
app.get('/health', (_req, res) => {
  res.json({
    ok: true,
    env: {
      GEMINI: !!GEMINI_API_KEY,
      BRAVE: !!BRAVE_TOKEN,
      REDIS: !!redis,
      PUPPETEER: !!puppeteer && !!BRAVE_PATH
    }
  });
});

app.listen(PORT, () => {
  console.log(`School AI server listening on port ${PORT}`);
});

