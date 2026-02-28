import path from "path";

export const BASE_DIR =
    process.platform === "win32"
        ? path.join(process.cwd(), "recordings")
        : "/var/recordings";

export const RING_DIR = path.join(BASE_DIR, "ring");
export const ARCHIVE_DIR = path.join(BASE_DIR, "archive");

export const SEG_MS = 2000;
export const KEEP_RING_MS = 10 * 60 * 1000;
export const FALL_PRE_MS = 16_000;
export const FALL_POST_MS = 16_000;