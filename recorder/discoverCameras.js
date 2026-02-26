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
    let stdout = "";
    let stderr = "";
    try {
        stdout = execFileSync(
            "ffmpeg",
            ["-list_devices", "true", "-f", "dshow", "-i", "dummy"],
            { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] }
        );
    } catch (e) {
        // ffmpeg는 종종 exit code 1을 내면서도 목록을 stderr로 출력함
        stdout = (e.stdout && e.stdout.toString()) || "";
        stderr = (e.stderr && e.stderr.toString()) || "";
    }

    // ✅ 성공/실패 상관없이 stderr에 나오는 경우가 많아서 합쳐서 파싱
    const out = `${stdout}\n${stderr}`;
    const lines = out.split(/\r?\n/);

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
        if (m && m[1] && !line.includes("Alternative name")) {
            names.push(m[1]);
        }
    }

    names.forEach((name, i) => {
        const camId = `cam${String(i + 1).padStart(2, "0")}`;
        cams[camId] = { device: `video=${name}`, width: 1280, height: 720, fps: 30 };
    });

    return cams;
}

function discoverCameras() {
    if (process.platform === "win32") return discoverCamerasWindows();
    return discoverCamerasLinux();
}

module.exports = { discoverCameras };