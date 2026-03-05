/**
 * [글로벌 설정 및 상수]
 * Choi Inha 님의 Project ASK 관제 시스템용 스크립트입니다.
 */
const CLIP_BASE_URL = "http://192.168.0.2:8081"; // 로그 및 SSE 서버
const CORAL_AI_URL = "http://192.168.0.140:5050"; // 코랄보드 AI 영상 서버
let currentCamId = ""; 
let camStates = {}; 
let pc = null; // WebRTC PeerConnection 객체

let currentPage = 1;
const rowsPerPage = 10;
let selectedFilterDate = "";

/**
 * 1. 초기 로드 및 이벤트 리스너
 */
document.addEventListener('DOMContentLoaded', async () => {
    initSidebar();

    // 1) 서버에서 과거 기록 동기화
    await fetchHistoryFromServer();

    // 2) 백엔드 데이터를 기반으로 카메라 탭 UI 생성
    await loadCameraUI();

    // 3) 실시간 수신 시작 (SSE)
    initSSE();

    const datePicker = document.getElementById('date-filter');
    if (datePicker) {
        datePicker.addEventListener('change', (e) => {
            selectedFilterDate = e.target.value;
            renderHistory(1);
        });
    }

    window.onclick = (event) => {
        const historyModal = document.getElementById('video-modal');
        const liveModal = document.getElementById('live-expand-modal');

        if (event.target === historyModal) closeModal();
        if (event.target === liveModal) closeLiveModal();
    };

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            closeModal();
            closeLiveModal();
        }
    });
});

/**
 * 2. WebRTC 핵심 로직 (코랄보드 연동 수정)
 */

function stopWebRTC() {
    if (pc) {
        pc.close();
        pc = null;
    }
    const webcamElement = document.getElementById('webcam');
    if (webcamElement) {
        webcamElement.srcObject = null;
        // 만약 MJPEG 방식을 혼용한다면 src도 초기화
        webcamElement.src = "";
    }
}

async function initWebcam() {
    const webcamElement = document.getElementById('webcam');
    if (!webcamElement || !currentCamId) return;

    stopWebRTC();

    /**
     * [참고] 만약 코랄보드가 WebRTC 시그널링을 지원하지 않고 
     * 단순 MJPEG 스트림(/video_feed)만 제공한다면 아래 주석을 해제하고 사용하세요.
     * * webcamElement.src = `${CORAL_AI_URL}/video_feed`; // MJPEG 방식일 때
     * return;
     */

    // WebRTC 방식 (코랄보드와 SDP 교환)
    pc = new RTCPeerConnection({
        iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
    });

    pc.ontrack = (event) => {
        if (webcamElement.srcObject !== event.streams[0]) {
            webcamElement.srcObject = event.streams[0];
            console.log(`[${currentCamId}] 코랄보드 AI 스트림 수신 시작`);
        }
    };

    pc.addTransceiver('video', { direction: 'recvonly' });

    try {
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);

        // 코랄보드 AI 서버의 WebRTC 엔드포인트로 Offer 전송
        // 엔드포인트 경로는 코랄보드 백엔드 설정에 따라 /offer 등으로 바뀔 수 있습니다.
        const response = await fetch(`${CORAL_AI_URL}/offer`, {
            method: 'POST',
            body: JSON.stringify({
                sdp: pc.localDescription.sdp,
                type: pc.localDescription.type,
                cam_id: currentCamId // 카메라 ID 전달
            }),
            headers: { 'Content-Type': 'application/json' }
        });

        if (!response.ok) throw new Error("코랄보드 응답 오류");

        const answer = await response.json();
        await pc.setRemoteDescription(new RTCSessionDescription(answer));
    } catch (err) {
        console.error(`[${currentCamId}] 코랄보드 WebRTC 연결 실패:`, err);
        // 실패 시 폴백(Fallback)으로 일반 MJPEG 연결 시도 시나리오
        webcamElement.src = `${CORAL_AI_URL}/video_feed`;
    }
}

/**
 * 3. 카메라 UI 및 전환 로직 (기능 유지)
 */
async function fetchAvailableCameras() {
    try {
        const logs = JSON.parse(localStorage.getItem('risk_logs')) || [];
        const uniqueCams = [...new Set(logs.map(log => log.camId))].filter(Boolean);
        return uniqueCams.length > 0 ? uniqueCams : ['cam01'];
    } catch (err) {
        return ['cam01'];
    }
}

async function loadCameraUI() {
    const camList = await fetchAvailableCameras();
    const container = document.getElementById('camera-selector-list');
    if (!container) return;

    container.innerHTML = '';
    camList.forEach((camId, index) => {
        ensureCameraExistsInUI(camId);
        if (index === 0 && currentCamId === "") currentCamId = camId;
    });

    if (currentCamId) switchCamera(currentCamId);
}

function ensureCameraExistsInUI(camId) {
    const container = document.getElementById('camera-selector-list');
    if (!container || document.getElementById(`btn-${camId}`)) return;

    if (!camStates[camId]) camStates[camId] = { status: 'normal', timer: null };

    const btn = document.createElement('button');
    btn.className = 'cam-btn';
    btn.id = `btn-${camId}`;
    btn.innerText = camId.toUpperCase();
    btn.onclick = () => switchCamera(camId);
    container.appendChild(btn);
}

function switchCamera(targetCamId) {
    currentCamId = targetCamId;
    document.querySelectorAll('.cam-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.id === `btn-${targetCamId}`) {
            btn.classList.add('active');
            btn.classList.remove('has-alert');
        }
    });

    initWebcam();
    refreshDashboardView();
}

/**
 * 4. SSE 및 실시간 이벤트 (기능 유지)
 */
function initSSE() {
    const es = new EventSource(`${CLIP_BASE_URL}/events`);
    es.addEventListener("hello", (e) => console.log("SSE Connected:", e.data));
    es.addEventListener("fall", (e) => {
        try {
            const data = JSON.parse(e.data);
            ensureCameraExistsInUI(data.camId);
            updateDashboardUI(3, data.camId, data.eventMs, data.clipUrl);
        } catch (err) {
            console.error("SSE 파싱 에러:", err);
        }
    });
}

function updateDashboardUI(level, camId, eventMs, clipUrl) {
    if (!camStates[camId]) camStates[camId] = { status: 'normal', timer: null };
    camStates[camId].status = 'danger';
    if (camStates[camId].timer) clearTimeout(camStates[camId].timer);

    camStates[camId].timer = setTimeout(() => {
        camStates[camId].status = 'normal';
        camStates[camId].timer = null;
        if (currentCamId === camId) refreshDashboardView();
    }, 30000);

    const logData = transformDataToLog({ camId, eventMs, clipUrl });
    saveLogToStorage(logData);

    if (currentCamId === camId) {
        refreshDashboardView();
    } else {
        const camBtn = document.getElementById(`btn-${camId}`);
        if (camBtn) camBtn.classList.add('has-alert');
    }
    if (document.getElementById('history-body')) renderHistory(1);
}

/**
 * 5. 화면 렌더링 및 대시보드 관리 (기능 유지)
 */
function refreshDashboardView() {
    const statusCard = document.getElementById('status-card');
    const statusIcon = document.getElementById('status-icon');
    const statusText = document.getElementById('status-text');
    const emergencyBox = document.getElementById('emergency-actions');
    const lastUpdate = document.getElementById('last-update');

    if (!statusCard) return;

    const state = camStates[currentCamId] || { status: 'normal' };
    lastUpdate.innerText = getFormattedTime(new Date());

    if (state.status === 'danger') {
        statusCard.className = "status-card status-danger";
        if (statusIcon) statusIcon.innerHTML = `<i class="fas fa-exclamation-triangle"></i>`;
        statusText.innerHTML = `[${currentCamId}] 낙상 감지!`;
        if (emergencyBox) emergencyBox.className = "emergency-box active";
    } else {
        statusCard.className = "status-card status-normal";
        if (statusIcon) statusIcon.innerHTML = `<i class="fas fa-check-circle"></i>`;
        statusText.innerHTML = `[${currentCamId}] 감지 중`;
        if (emergencyBox) emergencyBox.className = "emergency-box disabled";
    }
    renderDashboardMiniLogs();
}

function renderDashboardMiniLogs() {
    const container = document.getElementById('log-container');
    if (!container) return;
    const allLogs = JSON.parse(localStorage.getItem('risk_logs')) || [];
    const targetLogs = allLogs.filter(l => l.camId === currentCamId);

    if (targetLogs.length === 0) {
        container.innerHTML = `<div class="log-placeholder">${currentCamId} 로그가 없습니다.</div>`;
        return;
    }

    container.innerHTML = targetLogs.slice(0, 5).map(l => `
        <div class="log-item ${l.className}" onclick="openVideoModal('${l.videoUrl}', '${l.date} ${l.time}')">
            <span><strong>[Lv.${l.level}]</strong> ${l.message}</span>
            <span style="font-size: 0.75rem; color: #999;">${l.time}</span>
        </div>
    `).join('');
}

/**
 * 6. 데이터 영속성 및 히스토리 (기능 유지)
 */
async function fetchHistoryFromServer() {
    try {
        const response = await fetch(`${CLIP_BASE_URL}/api/falls`);
        const serverData = await response.json();
        const localLogs = JSON.parse(localStorage.getItem('risk_logs')) || [];
        const allLogsMap = new Map();

        serverData.forEach(item => {
            const log = transformDataToLog(item);
            allLogsMap.set(log.id, log);
        });
        localLogs.forEach(log => {
            if (!allLogsMap.has(log.id)) allLogsMap.set(log.id, log);
        });

        const finalLogs = Array.from(allLogsMap.values()).sort((a, b) => b.id - a.id);
        localStorage.setItem('risk_logs', JSON.stringify(finalLogs.slice(0, 500)));
        renderHistory(1);
    } catch (err) {
        console.error("기록 동기화 실패:", err);
    }
}

function transformDataToLog(item) {
    const dateObj = new Date(item.eventMs);
    return {
        id: item.eventMs,
        camId: item.camId,
        level: 3,
        isoDate: dateObj.toISOString().split('T')[0],
        hour: dateObj.getHours(),
        date: dateObj.toLocaleDateString(),
        time: getFormattedTime(dateObj),
        type: "위험",
        message: `[${item.camId}] 낙상 발생`,
        className: "danger",
        videoUrl: item.clipUrl
    };
}

function saveLogToStorage(newLog) {
    const logs = JSON.parse(localStorage.getItem('risk_logs')) || [];
    if (logs.find(l => l.id === newLog.id)) return;
    logs.unshift(newLog);
    localStorage.setItem('risk_logs', JSON.stringify(logs.slice(0, 500)));
}

function renderHistory(page = 1) {
    const tbody = document.getElementById('history-body');
    if (!tbody) return;
    const allLogs = JSON.parse(localStorage.getItem('risk_logs')) || [];
    let filteredLogs = selectedFilterDate ? allLogs.filter(l => l.isoDate === selectedFilterDate) : allLogs;

    const totalPages = Math.ceil(filteredLogs.length / rowsPerPage) || 1;
    currentPage = Math.max(1, Math.min(page, totalPages));
    const pagedLogs = filteredLogs.slice((currentPage - 1) * rowsPerPage, currentPage * rowsPerPage);

    tbody.innerHTML = pagedLogs.length === 0
        ? `<tr><td colspan="3" style="padding:40px;">데이터가 없습니다.</td></tr>`
        : pagedLogs.map(l => `
            <tr onclick="openVideoModal('${l.videoUrl}', '${l.date} ${l.time}')">
                <td>${l.date}<br>${l.time}</td>
                <td><span class="tag ${l.className}">${l.type}</span></td>
                <td style="text-align: center;">${l.message}<br><span style="color:#007aff; font-size:0.8rem;">▶ 영상보기</span></td>
            </tr>
        `).join('');

    updatePaginationUI(totalPages);
    renderRiskChart(allLogs);
}

/**
 * 7. 모달 및 헬퍼 함수
 */
function openVideoModal(url, timeInfo) {
    const modal = document.getElementById('video-modal');
    const player = document.getElementById('history-video-player');
    const infoText = document.getElementById('modal-info');
    if (modal && player) {
        modal.style.display = 'block';
        infoText.innerText = `감지 시간: ${timeInfo}`;
        player.src = url;
        player.play().catch(() => {});
    }
}

function closeModal() {
    const modal = document.getElementById('video-modal');
    const player = document.getElementById('history-video-player');
    if (modal) {
        modal.style.display = 'none';
        player.pause();
        player.src = "";
    }
}

function openLiveModal() {
    const modal = document.getElementById('live-expand-modal');
    const mainWebcam = document.getElementById('webcam');
    const expandedVideo = document.getElementById('live-expanded-stream');
    const title = document.getElementById('live-modal-title');

    if (!modal || !mainWebcam || !expandedVideo) return;

    if (mainWebcam.srcObject) {
        expandedVideo.srcObject = mainWebcam.srcObject;
    } else {
        expandedVideo.src = mainWebcam.src; // MJPEG 대응
    }
    
    title.innerHTML = `<i class="fas fa-video"></i> [실시간 관제] ${currentCamId.toUpperCase()}`;
    modal.style.display = 'block';
    document.body.style.overflow = 'hidden';
}

function closeLiveModal() {
    const modal = document.getElementById('live-expand-modal');
    const expandedVideo = document.getElementById('live-expanded-stream');
    if (modal) {
        modal.style.display = 'none';
        expandedVideo.srcObject = null;
        expandedVideo.src = "";
        document.body.style.overflow = '';
    }
}

function getFormattedTime(date) {
    return `${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}:${String(date.getSeconds()).padStart(2, '0')}`;
}

function initSidebar() {
    const menuBtn = document.getElementById('menu-btn'), closeBtn = document.getElementById('close-btn'), 
          sideNav = document.getElementById('side-nav'), overlay = document.getElementById('overlay');
    if (menuBtn && sideNav && overlay) {
        const toggleNav = (isOpen) => {
            sideNav.classList.toggle('active', isOpen);
            overlay.classList.toggle('active', isOpen);
        };
        menuBtn.addEventListener('click', () => toggleNav(true));
        if (closeBtn) closeBtn.addEventListener('click', () => toggleNav(false));
        overlay.addEventListener('click', () => toggleNav(false));
    }
}

function updatePaginationUI(totalPages) {
    const container = document.getElementById('pagination');
    if (!container) return;
    container.innerHTML = `
        <button class="pagination-btn ${currentPage === 1 ? 'disabled' : ''}" onclick="renderHistory(${currentPage - 1})">이전</button>
        <span style="margin: 0 15px;">${currentPage} / ${totalPages}</span>
        <button class="pagination-btn ${currentPage === totalPages ? 'disabled' : ''}" onclick="renderHistory(${currentPage + 1})">다음</button>
    `;
}

function renderRiskChart(allLogs) {
    const canvas = document.getElementById('riskChart');
    if (!canvas || typeof Chart === 'undefined') return;
    let labels = [], dataCounts = [];
    if (selectedFilterDate) {
        labels = ["00", "04", "08", "12", "16", "20"];
        dataCounts = new Array(6).fill(0);
        allLogs.filter(l => l.isoDate === selectedFilterDate).forEach(l => dataCounts[Math.floor(l.hour / 4)]++);
    } else {
        for (let i = 6; i >= 0; i--) {
            const d = new Date(); d.setDate(d.getDate() - i);
            labels.push((d.getMonth() + 1) + "/" + d.getDate());
            dataCounts.push(allLogs.filter(l => l.isoDate === d.toISOString().split('T')[0]).length);
        }
    }
    const existingChart = Chart.getChart("riskChart");
    if (existingChart) existingChart.destroy();
    new Chart(canvas.getContext('2d'), {
        type: 'line',
        data: { labels, datasets: [{ label: '감지 횟수', data: dataCounts, borderColor: '#1d1d1f', borderWidth: 2, fill: true, tension: 0.1 }] },
        options: { responsive: true, maintainAspectRatio: false }
    });
}