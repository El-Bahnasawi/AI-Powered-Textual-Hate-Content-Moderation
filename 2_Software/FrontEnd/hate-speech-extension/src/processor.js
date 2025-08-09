// processor.js  (drop-in replacement)
import { getCleanText, shouldProcess, log, DEBUG_MODE, error } from "./utils.js";

// const API_ENDPOINT   = "https://medoxz543-hate-endpoint.hf.space/log-results";
const API_ENDPOINT   = "https://medoxz543-hate-endpoint.hf.space/check-text";

const MAX_BATCH_SIZE = 10;    // tune as you like
const CALL_GAP_MS    = 200;   // 0.2 s – prevents self-DoS on Render

const elementMap = new Map(); // text → [DOM nodes]
const seenTexts  = new Map(); // text → { blur, score }

export async function processText(elements) {
    const newTexts = [];

    /* 1️⃣  Gather fresh texts */
    elements.forEach(el => {
        const text = getCleanText(el);
        if (!text || !shouldProcess(el)) return;

        // remember which DOM elements map to this text
        if (!elementMap.has(text)) elementMap.set(text, []);
        elementMap.get(text).push(el);

        // if we’ve already classified this text, reuse the decision
        if (seenTexts.has(text)) {
            const { blur, score } = seenTexts.get(text);
            if (blur) blurElement(el, score);
            return;                 // skip server round-trip
        }
        newTexts.push(text);
    });

    if (!newTexts.length) return;   // nothing to classify

    /* 2️⃣  Send in batches */
    for (let i = 0; i < newTexts.length; i += MAX_BATCH_SIZE) {
        const batch   = newTexts.slice(i, i + MAX_BATCH_SIZE);
        const startMs = Date.now();

        log("%c📤 Sending batch to server:", "color: green; font-weight: bold;", batch);

        try {
            const res = await fetch(API_ENDPOINT, {
                method:  "POST",
                mode:    "cors",
                headers: { "Content-Type": "application/json" },
                body:    JSON.stringify({ texts: batch })
            });

            if (!res.ok) {
                error(`❌ Server error ${res.status} – ${res.statusText}`);
                error(await res.text());
                continue;           // move on to next batch
            }

            const { results = [], db_logged } = await res.json();
            log(`📡 API response (${batch.length}) ${Date.now() - startMs} ms`);
            log("🧾 Results:", results);
            
            db_logged !== undefined && log("📄 DB logging status:", db_logged);

            /* 3️⃣  Apply decisions */
            results.forEach(({ blur, score }, idx) => {
                const text = batch[idx];
                seenTexts.set(text, { blur, score });
                if (blur) {
                    (elementMap.get(text) || []).forEach(el => blurElement(el, score));
                }
            });

        } catch (err) {
            error("🔥 Fetch failed:", err);
        }

        // small gap before the next batch (never blocks UI)
        await delay(CALL_GAP_MS);
    }
}

/* Helpers */
function blurElement(el, score) {
    el.classList.add("ai-blur");
    el.setAttribute("title", `Blurred by AI 🛡️ (Score: ${score.toFixed(2)})`);
    el.dataset.blurred = "true";
    el.dataset.score   = score;
}

function delay(ms) {
    return new Promise(r => setTimeout(r, ms));
}
