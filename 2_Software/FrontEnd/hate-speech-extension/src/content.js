import { scanYT } from "./scannerYT.js";
import { scanTwitter } from "./scannerTwitter.js";
import { log } from "./utils.js";

const scanners = {
    "x.com": scanTwitter,
    "twitter.com": scanTwitter,
    "youtube.com": scanYT
};

const currentHost = location.hostname.replace(/^www\./, "");
const handler = scanners[currentHost];

// Default to true unless explicitly stored as "false"
const stored = localStorage.getItem("blurEnabled");
window.blurEnabled = stored === null ? true : stored !== "false";

if (handler) {
    createBlurToggleButton(); // Inject the toggle after scanner activation
    handler();
} else {
    log("🌐 No scanner defined for this domain:", currentHost);
}

function createBlurToggleButton() {
    const tryInject = () => {
        if (!document.body) {
            requestAnimationFrame(tryInject);
            return;
        }

        console.log("🟢 Attempting to inject toggle button...");

        const btn = document.createElement("button");
        btn.id = "blur-toggle-btn";
        btn.textContent = window.blurEnabled ? "🛡️ Blur: ON" : "🛡️ Blur: OFF";

        btn.addEventListener("click", () => {
            window.blurEnabled = !window.blurEnabled;
            localStorage.setItem("blurEnabled", window.blurEnabled);
            btn.textContent = window.blurEnabled ? "🛡️ Blur: ON" : "🛡️ Blur: OFF";

            document.querySelectorAll("[data-blurred='true']").forEach(el => {
                if (window.blurEnabled) {
                    el.classList.add("ai-blur");
                } else {
                    el.classList.remove("ai-blur");
                }
            });
        });

        document.body.appendChild(btn);
        console.log("✅ Toggle button injected.");
    };

    tryInject();
}

function showPrivacyConsent() {
    if (localStorage.getItem("ai_blur_consent") === "true") return;

    const banner = document.createElement("div");
    banner.style.cssText = `
        position: fixed; bottom: 0; left: 0; right: 0;
        background: #111; color: #fff;
        padding: 10px 20px;
        font-size: 14px;
        z-index: 9999;
        display: flex; justify-content: space-between; align-items: center;
        box-shadow: 0 -2px 5px rgba(0,0,0,0.2);
    `;

    banner.innerHTML = `
        <span>
            This extension sends visible text to an AI model to detect potential hate speech.
        </span>
        <button id="acceptPrivacy" style="
            background: #0f62fe; border: none; padding: 5px 12px;
            color: white; cursor: pointer; border-radius: 4px;
        ">
            I Understand
        </button>
    `;

    document.body.appendChild(banner);

    document.getElementById("acceptPrivacy").addEventListener("click", () => {
        localStorage.setItem("ai_blur_consent", "true");
        banner.remove();
    });
}

