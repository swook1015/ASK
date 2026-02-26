import { spawn } from "child_process";

export function spawnFfmpeg(args, { name = "ffmpeg" } = {}) {
    const p = spawn("ffmpeg", args, { stdio: ["ignore", "inherit", "inherit"] });
    p.on("exit", (code, sig) => {
        console.log(`[${name}] exited code=${code} sig=${sig}`);
    });
    return p;
}

export function runFfmpeg(args, { name = "ffmpeg-run" } = {}) {
    return new Promise((resolve, reject) => {
        const p = spawn("ffmpeg", args, { stdio: ["ignore", "inherit", "inherit"] });
        p.on("close", (code) => {
            if (code === 0) resolve();
            else reject(new Error(`[${name}] ffmpeg failed code=${code}`));
        });
    });
}