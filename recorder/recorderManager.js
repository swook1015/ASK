import { Recorder } from "./Recorder.js";

export class RecorderManager {
    constructor(camerasConfig) {
        // camerasConfig: { cam01: { device, width, height, fps }, ... }
        this.camerasConfig = camerasConfig;
        this.recorders = new Map();
    }

    async startAll() {
        const entries = Object.entries(this.camerasConfig);
        for (const [camId, cfg] of entries) {
            await this.start(camId, cfg);
        }
    }

    async start(camId, cfg) {
        if (this.recorders.has(camId)) return;

        const r = new Recorder({
            camId,
            devicePath: cfg.device,
            width: cfg.width ?? 1280,
            height: cfg.height ?? 720,
            fps: cfg.fps ?? 30,
        });

        await r.start();
        this.recorders.set(camId, r);
    }

    async stop(camId) {
        const r = this.recorders.get(camId);
        if (!r) return;
        await r.stop();
        this.recorders.delete(camId);
    }

    async handleFall(camId, eventMs = Date.now()) {
        const r = this.recorders.get(camId);
        if (!r) {
            console.warn(`[RecorderManager] no recorder for camId=${camId}`);
            return;
        }
        await r.handleFall(eventMs);
    }
}