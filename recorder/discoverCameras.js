const fs = require("fs");
const path = require("path");
const { execFileSync } = require("child_process");
const {spawnSync} = require("child_process");
function discoverCamerasLinux() {
    const dir = "/dev/v4l/by-id";
    if (!fs.existsSync(dir)) return {};

    const entries = fs
        .readdirSync(dir)
        .filter((name) => name.includes("video-index0"))
        .sort();

    const cams = {};
    entries.forEach((name, i) => {
        const camId = `cam${String(i + 1).padStart(2, "0")}`;
        const full = path.join(dir, name);
        cams[camId] = { device: full, width: 1280, height: 720, fps: 30 };
    });
    return cams;
}

function discoverCamerasWindows() {
  const r = require("child_process").spawnSync(
    "ffmpeg",
    ["-list_devices", "true", "-f", "dshow", "-i", "dummy"],
    { encoding: "utf8" }
  );

  const out = `${r.stdout || ""}\n${r.stderr || ""}`;
  const lines = out.split(/\r?\n/);

  const cams = {};
  const names = [];

  for (const line of lines) {
    // ✅ 너 출력 포맷: [dshow @ ...] "NC20  " (video)
    if (!line.includes("(video)")) continue;

    const m = line.match(/"(.+?)"/);
    if (m && m[1]) names.push(m[1]);
  }

  names.forEach((name, i) => {
    const camId = `cam${String(i + 1).padStart(2, "0")}`;
    cams[camId] = { device: `video=${name}`, width: 1920, height: 1080, fps: 30 };
  });

  if (Object.keys(cams).length === 0) {
    console.warn("[discoverCamerasWindows] no devices parsed.\n", out);
  }

  return cams;
}

function discoverCameras() {
    if (process.platform === "win32") return discoverCamerasWindows();
    return discoverCamerasLinux();
}

module.exports = { discoverCameras };