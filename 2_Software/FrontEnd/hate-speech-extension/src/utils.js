export const DEBUG_MODE = true;

export const log = (...args) => DEBUG_MODE && console.log(...args);
export const error = (...args) => DEBUG_MODE && console.error(...args);


export const isTextElement = el => ["P", "SPAN", "A", "H1", "H2", "H3", "H4", "H5", "H6", "DIV", "YT-ATTRIBUTED-STRING"].includes(el.tagName);
export const getCleanText = el => el?.innerText?.trim();
export const hasEnoughWords = text => text?.split(/\s+/).length >= 2;
export const isEnglish = el => el.getAttribute("lang") === "en";

export const shouldProcess = el => {
    const tag = el?.tagName;
    const text = getCleanText(el);

    const isTwitter = location.hostname.includes("twitter") || location.hostname.includes("x.com");

    const valid = text
        && isTextElement(el)
        && (!isTwitter || isEnglish(el))  // only apply language filter on Twitter
        && hasEnoughWords(text)
        && !el.hasAttribute("data-score");  // ✅ Skip already processed elements

    log(valid ? "✅ Valid:" : "❌ Skipped:", `[${tag}]`, text);
    return valid;
};
