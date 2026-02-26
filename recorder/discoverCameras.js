const fs = require("fs");
const path = require("path");
const { execFileSync } = require("child_process");

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
    // ffmpeg -list_devices true -f dshow -i dummy 출력 파싱
    let out = "";
    try {
        out = execFileSync("ffmpeg", ["-list_devices", "true", "-f", "dshow", "-i", "dummy"], {
            encoding: "utf8",
            stdio: ["ignore", "pipe", "pipe"],
        });
    } catch (e) {
        // ffmpeg는 stderr로 내보내는 경우가 많음
        out = (e.stderr && e.stderr.toString()) || "";
    }

    const lines = out.split(/\r?\n/);

    // "DirectShow video devices" 섹션 이후의 "  \"NAME\"" 형태를 수집
    const cams = {};
    let inVideoSection = false;
    const names = [];

    for (const line of lines) {
        if (line.includes("DirectShow video devices")) {
            inVideoSection = true;
            continue;
        }
        if (line.includes("DirectShow audio devices")) {
            inVideoSection = false;
            continue;
        }
        if (!inVideoSection) continue;

        const m = line.match(/"(.+?)"/);
        if (m && m[1] && !m[1].includes("Alternative name")) {
            names.push(m[1]);
        }
    }

    names.forEach((name, i) => {
        const camId = `cam${String(i + 1).padStart(2, "0")}`;
        cams[camId] = {
            device: `video=${name}`, // dshow 입력 형식
            width: 1280,
            height: 720,
            fps: 30,
        };
    });

    return cams;
}

function discoverCameras() {
    if (process.platform === "win32") return discoverCamerasWindows();
    return discoverCamerasLinux();
}

module.exports = { discoverCameras };