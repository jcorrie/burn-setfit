// Runs the browser smoke test headlessly and reports what the page measured.
//
// Serves `public/` over http (a `file://` page cannot instantiate wasm or fetch
// the checkpoint), drives it in Chromium, and exits non-zero if anything the
// page asserts came back wrong.

import { chromium } from 'playwright';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(fileURLToPath(new URL('.', import.meta.url)), 'public');
const CHROME = process.env.CHROME_PATH ?? '/opt/pw-browsers/chromium-1194/chrome-linux/chrome';
const TIMEOUT_MS = Number(process.env.TIMEOUT_MS ?? 180_000);

const MIME = {
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.json': 'application/json',
  // Without this, `WebAssembly.instantiateStreaming` refuses the response.
  '.wasm': 'application/wasm',
  '.safetensors': 'application/octet-stream',
};

const server = createServer(async (req, res) => {
  try {
    const path = normalize(decodeURIComponent(new URL(req.url, 'http://x').pathname));
    const file = join(ROOT, path === '/' ? 'index.html' : path);
    if (!file.startsWith(ROOT)) throw new Error('outside root');
    const body = await readFile(file);
    res.writeHead(200, { 'content-type': MIME[extname(file)] ?? 'application/octet-stream' });
    res.end(body);
  } catch {
    res.writeHead(404).end('not found');
  }
});

await new Promise((r) => server.listen(0, '127.0.0.1', r));
const url = `http://127.0.0.1:${server.address().port}/`;
console.log(`serving ${ROOT} at ${url}`);

const browser = await chromium.launch({ executablePath: CHROME });
const page = await browser.newPage();

const consoleErrors = [];
// Chromium requests /favicon.ico unprompted; the resulting 404 is logged as a
// console error whose text does not name the URL, so filter on the location.
const isFavicon = (m) => (m.location()?.url ?? '').endsWith('/favicon.ico');
page.on('console', (m) => {
  if (m.type() !== 'error') console.log(`  [page] ${m.text()}`);
  else if (!isFavicon(m)) consoleErrors.push(`${m.text()} (${m.location()?.url ?? 'no url'})`);
});
page.on('pageerror', (e) => consoleErrors.push(String(e)));

let result;
try {
  await page.goto(url);
  await page.waitForFunction(() => window.__RESULT__?.done === true, null, { timeout: TIMEOUT_MS });
  result = await page.evaluate(() => window.__RESULT__);
} finally {
  await browser.close();
  server.close();
}

console.log('\n--- result ---');
console.log(JSON.stringify({ ...result, steps: `${result.steps.length} steps` }, null, 2));

const failures = [];
if (!result.ok) failures.push('page reported not ok');
for (const e of result.errors) failures.push(`page error: ${e}`);
for (const e of consoleErrors) failures.push(`console error: ${e}`);
if (result.short_correct !== result.short?.length) {
  failures.push(`held-in examples: ${result.short_correct}/${result.short?.length} correct`);
}
if (!(result.long?.chunks_seen > 1)) failures.push('long document did not chunk');
// The whole reason training is a step-wise state machine: no single step may
// wedge the main thread. 250 ms is the threshold past which a tab feels stuck.
if (!(result.responsiveness?.worst_frame_gap_ms < 250)) {
  failures.push(`worst frame gap ${result.responsiveness?.worst_frame_gap_ms} ms`);
}

if (failures.length) {
  console.error('\nFAIL');
  for (const f of failures) console.error(`  - ${f}`);
  process.exit(1);
}
console.log('\nPASS');
