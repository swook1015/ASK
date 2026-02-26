import fs from "fs/promises";
import path from "path";
import chokidar from "chokidar";
import { spawnFfmpeg, runFfmpeg } from "./ffmpeg.js";
import { RING_DIR, ARCHIVE_DIR, SEG_MS, KEEP_RING_MS, FALL_PRE_MS, FALL_POST_MS } from "./paths.js";

function pad2(n) { return String(n).padStart(2, "0"); }

function dateParts(ms) {
    const d = new Date(ms);
    const yyyy = d.getFullYear();
    const mm = pad2(d.getMonth() + 1);
    const dd = pad2(d.getDate());
    const HH = pad2(d.getHours());
    return { ymd: `${yyyy}${mm}${dd}`, HH };
}

// 파일명 끝에 epoch sec(%s)를 넣는 형태: ..._1708950012.mp4
function parseEpochSecFromName(filename) {
    const m = filename.match(/_(\d+)\.mp4$/);
    if (!m) return null;
    return Number(m[1]);
}

export class Recorder {
    constructor({ camId, devicePath, width = 1280, height = 720, fps = 30 }) {
        this.camId = camId;
        this.devicePath = devicePath;
        this.width = width;
        this.height = height;
        this.fps = fps;

        this.ffmpegProc = null;
        this.watcher = null;

        // 최근 2시간 세그먼트 인덱스
        // { startMs, file }
        this.index = [];

        // 중복 낙상 이벤트 폭발 방지(필요 없으면 0)
        this.cooldownMs = 20_000;
        this.lastFallMs = 0;
    }

    ringBase() {
        return path.join(RING_DIR, this.camId);
    }
    archiveBase() {
        return path.join(ARCHIVE_DIR, this.camId);
    }

    async start() {
        await fs.mkdir(this.ringBase(), { recursive: true });
        await fs.mkdir(this.archiveBase(), { recursive: true });

        // 1) 폴더 watch로 세그먼트 인덱스 채우기
        this.watcher = chokidar.watch(this.ringBase(), {
            ignoreInitial: false,
            depth: 3,
        });

        this.watcher.on("add", (file) => this.onSegmentAdded(file));

        // 2) FFmpeg로 2초 세그먼트 생성 시작
        this.ffmpegProc = this.spawnSegmenter();

        // 3) ring cleanup 주기 실행
        this.cleanupTimer = setInterval(() => {
            this.cleanupOldSegments().catch(console.error);
        }, 60 * 1000);

        console.log(`[Recorder:${this.camId}] started device=${this.devicePath}`);
    }

    async stop() {
        if (this.cleanupTimer) clearInterval(this.cleanupTimer);

        if (this.watcher) await this.watcher.close().catch(() => {});
        this.watcher = null;

        if (this.ffmpegProc) {
            this.ffmpegProc.kill("SIGTERM");
            this.ffmpegProc = null;
        }
        console.log(`[Recorder:${this.camId}] stopped`);
    }

    spawnSegmenter() {
        const size = `${this.width}x${this.height}`;

        const outPattern =
            process.platform === "win32"
                ? path.join(this.ringBase(), "%Y%m%d_%H%M%S.mp4")
                : path.join(this.ringBase(), "%Y%m%d", "%H", "%Y%m%d_%H%M%S_%s.mp4");

        const inputArgs =
            process.platform === "win32"
                ? ["-f", "dshow", "-video_size", size, "-framerate", String(this.fps), "-i", this.devicePath]
                : ["-f", "v4l2", "-framerate", String(this.fps), "-video_size", size, "-i", this.devicePath];

        const args = [
            "-hide_banner",
            "-loglevel", "warning",
            ...inputArgs,

            "-c:v", "libx264",
            "-preset", "veryfast",
            "-tune", "zerolatency",
            "-g", String(this.fps * 2),
            "-keyint_min", String(this.fps * 2),
            "-sc_threshold", "0",
            "-pix_fmt", "yuv420p",

            "-f", "segment",
            "-segment_time", "2",
            "-reset_timestamps", "1",
            "-strftime", "1",

            outPattern,
        ];

        return spawnFfmpeg(args, { name: `ffmpeg-seg-${this.camId}` });
    }

    async onSegmentAdded(file) {
        if (!file.endsWith(".mp4")) return;

        const name = path.basename(file);
        const epochSec = parseEpochSecFromName(name);

        let startMs;
        if (epochSec) {
            startMs = epochSec * 1000;
        } else {
            const st = await fs.stat(file).catch(() => null);
            if (!st) return;
            startMs = st.mtimeMs;
        }

        this.index.push({ startMs, file });

        const cutoff = Date.now() - (KEEP_RING_MS + 60_000);
        while (this.index.length && this.index[0].startMs < cutoff) this.index.shift();
    }

    pickSegmentsBetween(startMs, endMs) {
        return this.index.filter(s => (s.startMs + SEG_MS) > startMs && s.startMs < endMs);
    }

    async waitUntilRecorded(endMs, timeoutMs = 25_000) {
        const start = Date.now();
        while (Date.now() - start < timeoutMs) {
            const latest = this.index.length ? this.index[this.index.length - 1].startMs : 0;
            if (latest >= endMs - SEG_MS) return;
            await new Promise(r => setTimeout(r, 200));
        }
        // timeout이어도 일단 진행(가능한 만큼만 저장)
    }

    async handleFall(eventMs = Date.now()) {
        // 쿨다운(연속 이벤트 방지)
        if (this.cooldownMs > 0 && eventMs - this.lastFallMs < this.cooldownMs) {
            console.log(`[Recorder:${this.camId}] fall ignored (cooldown)`);
            return;
        }
        this.lastFallMs = eventMs;

        const from = eventMs - FALL_PRE_MS;
        const to = eventMs + FALL_POST_MS;

        // 미래 15초분 세그먼트가 생성될 때까지 대기
        await this.waitUntilRecorded(to);

        const segs = this.pickSegmentsBetween(from, to);
        if (segs.length === 0) {
            console.log(`[Recorder:${this.camId}] no segments found for fall`);
            return;
        }

        const outDir = path.join(this.archiveBase(), String(eventMs));
        await fs.mkdir(outDir, { recursive: true });

        // 안전을 위해 ring 세그먼트를 복사(나중에 cleanup으로 지워질 수 있음)
        const copied = [];
        for (const s of segs) {
            const dst = path.join(outDir, path.basename(s.file));
            await fs.copyFile(s.file, dst).catch(() => {});
            copied.push(dst);
        }

        // concat 리스트 작성
        copied.sort(); // 파일명에 시간이 있으면 정렬로 순서 맞음
        const listPath = path.join(outDir, "list.txt");
        const listBody = copied.map(f => `file '${f.replace(/'/g, "'\\''")}'`).join("\n");
        await fs.writeFile(listPath, listBody, "utf8");

        const outMp4 = path.join(outDir, `fall_${this.camId}_${eventMs}.mp4`);

        // concat (재인코딩 없이)
        await runFfmpeg([
            "-hide_banner",
            "-loglevel", "warning",
            "-f", "concat",
            "-safe", "0",
            "-i", listPath,
            "-c", "copy",
            outMp4
        ], { name: `ffmpeg-concat-${this.camId}` });

        // 메타 저장(선택)
        await fs.writeFile(
            path.join(outDir, "meta.json"),
            JSON.stringify({ camId: this.camId, eventMs, from, to, segmentCount: copied.length }, null, 2),
            "utf8"
        );

        console.log(`[Recorder:${this.camId}] saved fall clip -> ${outMp4}`);
    }

    async cleanupOldSegments() {
        // 2시간보다 오래된 파일 삭제(파일 기준)
        const base = this.ringBase();
        const now = Date.now();

        // ring/camId 아래는 날짜/시간 폴더라서 재귀 삭제 필요
        // 간단하게: 3단 깊이까지 훑고 오래된 mp4 삭제
        const walk = async (dir, depth = 0) => {
            if (depth > 4) return;
            let entries;
            try { entries = await fs.readdir(dir, { withFileTypes: true }); }
            catch { return; }

            for (const e of entries) {
                const full = path.join(dir, e.name);
                if (e.isDirectory()) {
                    await walk(full, depth + 1);
                } else if (e.isFile() && e.name.endsWith(".mp4")) {
                    try {
                        const st = await fs.stat(full);
                        if (now - st.mtimeMs > KEEP_RING_MS) {
                            await fs.unlink(full).catch(() => {});
                        }
                    } catch {}
                }
            }
        };

        await walk(base);

        // 인덱스도 같이 정리
        const cutoff = Date.now() - KEEP_RING_MS;
        while (this.index.length && this.index[0].startMs < cutoff) this.index.shift();
    }
}