const fs = require("fs");
const path = require("path");

function discoverCameras() {
    const dir = "/dev/v4l/by-id";
    if (!fs.existsSync(dir)) return {};

    const entries = fs.readdirSync(dir)
        .filter(name => name.includes("video-index0"))
        .sort();

    const cams = {};
    entries.forEach((name, i) => {
        const camId = `cam${String(i + 1).padStart(2, "0")}`;
        const full = path.join(dir, name); // symlink
        cams[camId] = { device: full, width: 1280, height: 720, fps: 30 };
    });

    return cams;
}

module.exports = { discoverCameras };