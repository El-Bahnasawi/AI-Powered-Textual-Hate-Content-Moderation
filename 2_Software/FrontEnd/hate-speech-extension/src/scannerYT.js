import { log, getCleanText, hasEnoughWords } from "./utils.js";
import { processText } from "./processor.js";

export function scanYT() {
    log("📺 YouTube scan activated");

    // Extract comment text element from a container
    const getCommentTextElement = (container) => {
        const content = container.querySelector("#content-text");
        return content?.querySelector("yt-attributed-string") || content; // fallback
    };

    // Process a single comment thread
    const processCommentThread = (thread, elements) => {
        const textEl = getCommentTextElement(thread);
        if (textEl && hasEnoughWords(getCleanText(textEl))) {
            elements.push(textEl);
        }
    };

    // Initial scan of all existing comments
    const scan = () => {
        const elements = [];
        document.querySelectorAll("ytd-comment-thread-renderer").forEach(thread => {
            processCommentThread(thread, elements);
        });
        processText(elements);
    };

    // Handle mutations (new dynamically loaded comments)
    const handleMutations = (mutations) => {
        const elements = [];
        
        mutations.forEach(mutation => {
            mutation.addedNodes.forEach(node => {
                if (node.nodeType !== Node.ELEMENT_NODE) return;
                
                if (node.matches?.("ytd-comment-thread-renderer")) {
                    processCommentThread(node, elements);
                } else {
                    node.querySelectorAll?.("ytd-comment-thread-renderer")?.forEach(thread => {
                        processCommentThread(thread, elements);
                    });
                }
            });
        });

        if (elements.length > 0) {
            log("👀 New dynamic YT comments:", elements.length);
            processText(elements);
        }
    };

    const observer = new MutationObserver(handleMutations);
    scan();
    observer.observe(document.body, { childList: true, subtree: true });
}