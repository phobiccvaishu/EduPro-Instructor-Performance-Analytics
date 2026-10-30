import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..');
const outDir = path.join(repoRoot, 'docs', 'screenshots');

const CDP_PORT = parseInt(process.env.CDP_PORT || '9222', 10);
const ST_PORT = parseInt(process.env.ST_PORT || '8533', 10);

async function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function cdpCall(ws, method, params = {}) {
    const id = Math.floor(Math.random() * 1000000);
    return new Promise((resolve, reject) => {
        function onMessage(event) {
            const data = JSON.parse(event.data);
            if (data.id === id) {
                ws.removeEventListener('message', onMessage);
                if (data.error) reject(data.error);
                else resolve(data.result);
            }
        }
        ws.addEventListener('message', onMessage);
        ws.send(JSON.stringify({ id, method, params }));
    });
}

async function captureTarget(browserUrl, targetUrl, width, height, outPath) {
    console.log(`Creating target for ${targetUrl} (${width}x${height})...`);
    // Create new tab
    const newTabRes = await fetch(`http://127.0.0.1:${CDP_PORT}/json/new?${encodeURIComponent(targetUrl)}`, { method: 'PUT' });
    const tabInfo = await newTabRes.json();
    const wsUrl = tabInfo.webSocketDebuggerUrl;

    const ws = new WebSocket(wsUrl);
    await new Promise((resolve, reject) => {
        ws.onopen = resolve;
        ws.onerror = reject;
    });

    try {
        await cdpCall(ws, 'Page.enable');
        await cdpCall(ws, 'Emulation.setDeviceMetricsOverride', {
            width,
            height,
            deviceScaleFactor: 1,
            mobile: false,
        });

        // Wait for streamlit to finish rendering (metrics and charts appear, stStatusWidget disappears)
        console.log(`Waiting for Streamlit app to render...`);
        let rendered = false;
        for (let i = 0; i < 40; i++) {
            await sleep(500);
            const evalRes = await cdpCall(ws, 'Runtime.evaluate', {
                expression: `(() => {
                    const metrics = document.querySelectorAll('[data-testid="stMetric"]');
                    const running = document.querySelector('[data-testid="stStatusWidget"]');
                    const plotly = document.querySelectorAll('.js-plotly-plot');
                    return {
                        metricsCount: metrics.length,
                        isRunning: !!running,
                        plotlyCount: plotly.length
                    };
                })()`,
                returnByValue: true
            });
            const val = evalRes.result.value;
            if (val.metricsCount >= 5 && !val.isRunning) {
                console.log(`Render complete: ${val.metricsCount} metrics, ${val.plotlyCount} plots.`);
                rendered = true;
                break;
            }
        }

        // Additional pause to let layout and Plotly settle
        await sleep(2500);

        console.log(`Capturing screenshot to ${outPath}...`);
        const screenshotRes = await cdpCall(ws, 'Page.captureScreenshot', {
            format: 'png',
            fromSurface: true,
        });

        const buffer = Buffer.from(screenshotRes.data, 'base64');
        fs.writeFileSync(outPath, buffer);
        console.log(`Saved ${outPath} (${buffer.length} bytes).`);
    } finally {
        ws.close();
        await fetch(`http://127.0.0.1:${CDP_PORT}/json/close/${tabInfo.id}`).catch(() => {});
    }
}

async function run() {
    const base = `http://localhost:${ST_PORT}`;
    const targets = [
        { file: 'desktop_1440px_default.png', w: 1440, h: 900, url: base },
        { file: 'tablet_1024px_default.png', w: 1024, h: 768, url: base },
        { file: 'desktop_1440px_filtered.png', w: 1440, h: 900, url: `${base}/?preset=narrow` },
        { file: 'tablet_1024px_filtered.png', w: 1024, h: 768, url: `${base}/?preset=narrow` },
    ];

    for (const t of targets) {
        const outPath = path.join(outDir, t.file);
        await captureTarget(`http://localhost:8501`, t.url, t.w, t.h, outPath);
    }
    console.log('All screenshots captured successfully.');
}

run().catch(err => {
    console.error('Error during capture:', err);
    process.exit(1);
});
