import { Recorder } from "./Recorder.js";
import path from "path";
import conf from "../conf.js";

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

    // recorderManager.js
    async handleFall(camId, eventMs = Date.now()) {
        const r = this.recorders.get(camId);
        if (!r) return;

        const result = await r.handleFall(eventMs);
        if (!result) return;

        const fileName = path.basename(result.outMp4);

        const clipUrl =
            `http://${conf.clip.host}:${conf.clip.port}/clips/${camId}/${eventMs}/${fileName}`;

        global.pushEvent?.("fall", {
            camId,
            eventMs: result.eventMs,
            clipUrl,
        });

        const cnt = global.conf?.cnt?.find(c => c.name === "fall_clip");
        if (global.onem2m_client && cnt) {
            const parent = cnt.parent + "/" + cnt.name;
            const payload = {
                camId,
                eventMs: result.eventMs,
                from: result.from,
                to: result.to,
                clipUrl,
            };
            global.onem2m_client.create_cin(parent, 1, JSON.stringify(payload), this, () => {});
        }
    }
}