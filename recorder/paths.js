import path from "path";

export const BASE_DIR = "/var/recordings"; // 라즈베리파이 저장 위치(원하면 바꿔)
export const RING_DIR = path.join(BASE_DIR, "ring");
export const ARCHIVE_DIR = path.join(BASE_DIR, "archive");

export const SEG_MS = 2000;
export const KEEP_RING_MS = 2 * 60 * 60 * 1000; // 2 hours
export const FALL_PRE_MS = 15_000;
export const FALL_POST_MS = 15_000;