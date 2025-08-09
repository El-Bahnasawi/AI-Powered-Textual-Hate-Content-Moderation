import { log, getCleanText, hasEnoughWords } from "./utils.js";
import { processText } from "./processor.js";

export function scanTwitter() {
    log("🐦 Twitter scan activated");

    const scan = () => {
        const elements = [];
        document.querySelectorAll('[data-testid="cellInnerDiv"]').forEach(container => {
            container.querySelectorAll('[data-testid="tweetText"]').forEach(el => {
                if (isEnglish(el) && hasEnoughWords(getCleanText(el))) {
                    elements.push(el);
                }
            });
        });
        processText(elements);
    };

    const observer = new MutationObserver(mutations => {
        const elements = [];
        mutations.forEach(m => {
            m.addedNodes.forEach(node => {
                if (node.nodeType === 1) {
                    if (node.matches?.('[data-testid="cellInnerDiv"]')) {
                        elements.push(...node.querySelectorAll('[data-testid="tweetText"], div[lang="en"]'));
                    } else {
                        node.querySelectorAll?.('[data-testid="cellInnerDiv"]')?.forEach(tweet => {
                            elements.push(...tweet.querySelectorAll('[data-testid="tweetText"], div[lang="en"]'));
                        });
                    }
                }
            });
        });
        
        if (elements.length > 0) {
            log("👀 New dynamic tweets:", elements.length);
            processText(elements);
        }
    });

    scan();
    observer.observe(document.body, { childList: true, subtree: true });
}
