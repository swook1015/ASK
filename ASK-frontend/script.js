/**
 * [글로벌 설정 및 상수]
 */
const CLIP_BASE_URL = "http://192.168.0.2:8081"; // 명세서 기반 Base URL
let currentPage = 1;
const rowsPerPage = 10;
let selectedFilterDate = "";
let emergencyActiveTimer = null;

/**
 * 1. 초기 로드 및 이벤트 리스너
 */
document.addEventListener('DOMContentLoaded', () => {
    initSidebar();

    // 대시보드 및 히스토리 요소 존재 확인 후 초기화
    const isDashboard = document.getElementById('status-card');
    const isHistory = document.getElementById('history-body');

    // 1) 서버로부터 기존 낙상 기록 불러오기 (새로고침 대비)
    fetchHistoryFromServer();

    // 2) 실시간 이벤트 수신 시작 (SSE 연동)
    initSSE();

    // 날짜 필터 이벤트 리스너
    const datePicker = document.getElementById('date-filter');
    if (datePicker) {
        datePicker.addEventListener('change', (e) => {
            selectedFilterDate = e.target.value;
            renderHistory(1);
        });
    }

    // 모달 닫기 이벤트 (외부 클릭 시)
    window.onclick = (event) => {
        const modal = document.getElementById('video-modal');
        if (event.target === modal) closeModal();
    };
});

/**
 * 2. 실시간 이벤트 수신 (SSE: Server-Sent Events)
 */
function initSSE() {
    const es = new EventSource(`${CLIP_BASE_URL}/events`);

    // 연결 확인용
    es.addEventListener("hello", (e) => console.log("SSE Connected:", e.data));

    // 낙상(fall) 이벤트 수신 시
    es.addEventListener("fall", (e) => {
        try {
            const data = JSON.parse(e.data);
            console.log("실시간 낙상 감지:", data);

            // UI 업데이트 및 로그 저장 (위험 레벨 3 고정)
            updateDashboardUI(3, data.camId, data.eventMs, data.clipUrl);
        } catch (err) {
            console.error("SSE 데이터 파싱 에러:", err);
        }
    });

    es.onerror = (err) => {
        console.error("SSE 연결 오류. 재연결을 시도합니다.");
    };
}

/**
 * 3. 과거 기록 조회 (API 연동)
 */
async function fetchHistoryFromServer() {
    try {
        const response = await fetch(`${CLIP_BASE_URL}/api/falls`);
        if (!response.ok) throw new Error("서버 응답 오류");

        const data = await response.json(); // [{camId, eventMs, clipUrl}, ...]

        // 서버 데이터를 기존 로그 포맷으로 변환하여 저장
        const formattedLogs = data.map(item => transformDataToLog(item));

        // LocalStorage 동기화 (최신순 정렬)
        localStorage.setItem('risk_logs', JSON.stringify(formattedLogs));

        // 화면 렌더링
        renderHistory(1);
        renderDashboardMiniLogs();
    } catch (err) {
        console.error("기록 데이터를 가져오는데 실패했습니다:", err);
        renderHistory(1); // 실패 시 기존 로컬 데이터라도 노출
    }
}

/**
 * 4. 데이터 변환 및 저장 유틸리티
 */
function transformDataToLog(item) {
    const dateObj = new Date(item.eventMs);
    return {
        id: item.eventMs, // 타임스탬프를 고유 ID로 사용
        level: 3,
        isoDate: dateObj.toISOString().split('T')[0],
        hour: dateObj.getHours(),
        date: dateObj.toLocaleDateString(),
        time: getFormattedTime(dateObj),
        type: "위험",
        message: `[카메라 ${item.camId}] 낙상 발생`,
        className: "danger",
        videoUrl: item.clipUrl // 영상 재생 경로
    };
}

function saveLogToStorage(newLog) {
    const logs = JSON.parse(localStorage.getItem('risk_logs')) || [];
    // 중복 체크
    if (logs.find(l => l.id === newLog.id)) return;

    logs.unshift(newLog);
    localStorage.setItem('risk_logs', JSON.stringify(logs.slice(0, 500)));
}

/**
 * 5. UI 업데이트 (대시보드)
 */
function updateDashboardUI(level, camId, eventMs, clipUrl) {
    const statusCard = document.getElementById('status-card');
    const statusText = document.getElementById('status-text');
    const emergencyBox = document.getElementById('emergency-actions');
    const lastUpdate = document.getElementById('last-update');

    if (!statusCard) return;

    const eventDate = new Date(eventMs);
    lastUpdate.innerText = getFormattedTime(eventDate);

    // 위험 상태(Level 3) UI 적용
    statusCard.className = "status-card status-danger";
    statusText.innerText = `[${camId}] 낙상 발생!`;

    if (emergencyBox) {
        emergencyBox.className = "emergency-box active";
        if (emergencyActiveTimer) clearTimeout(emergencyActiveTimer);

        // 30초 후 알림 UI 자동 초기화
        emergencyActiveTimer = setTimeout(() => {
            emergencyBox.className = "emergency-box disabled";
            emergencyActiveTimer = null;
        }, 30000);
    }

    // 로그 객체 생성 및 저장
    const logData = transformDataToLog({ camId, eventMs, clipUrl });
    saveLogToStorage(logData);

    // 화면 갱신
    renderDashboardMiniLogs();
    if (document.getElementById('history-body')) renderHistory(1);
}

/**
 * 6. 히스토리 표 및 영상 재생 모달
 */
function renderHistory(page = 1) {
    const tbody = document.getElementById('history-body');
    if (!tbody) return;

    const allLogs = JSON.parse(localStorage.getItem('risk_logs')) || [];
    let filteredLogs = selectedFilterDate
        ? allLogs.filter(l => l.isoDate === selectedFilterDate)
        : allLogs;

    const totalPages = Math.ceil(filteredLogs.length / rowsPerPage) || 1;
    currentPage = Math.max(1, Math.min(page, totalPages));

    const startIndex = (currentPage - 1) * rowsPerPage;
    const pagedLogs = filteredLogs.slice(startIndex, startIndex + rowsPerPage);

    tbody.innerHTML = pagedLogs.length === 0
        ? `<tr><td colspan="3" style="padding:40px;">데이터가 없습니다.</td></tr>`
        : pagedLogs.map(l => `
            <tr onclick="openVideoModal('${l.videoUrl}', '${l.date} ${l.time}')">
                <td>${l.date}<br>${l.time}</td>
                <td><span class="tag ${l.className}">${l.type}</span></td>
                <td style="text-align: center;">
                    ${l.message}<br>
                    <span style="color:#007aff; font-size:0.8rem;">▶ 영상보기</span>
                </td>
            </tr>
        `).join('');

    updatePaginationUI(totalPages);
    renderRiskChart(allLogs);
}

// 영상 재생 모달 열기
function openVideoModal(url, timeInfo) {
    if (!url) return alert("영상 경로를 찾을 수 없습니다.");

    const modal = document.getElementById('video-modal');
    const player = document.getElementById('history-video-player');
    const infoText = document.getElementById('modal-info');

    if (modal && player) {
        modal.style.display = 'block';
        infoText.innerText = `감지 시간: ${timeInfo}`;
        player.src = url;
        player.load();
        player.play().catch(e => console.warn("자동 재생 차단됨:", e));
    }
}

// 모달 닫기
function closeModal() {
    const modal = document.getElementById('video-modal');
    const player = document.getElementById('history-video-player');
    if (modal) {
        modal.style.display = 'none';
        player.pause();
        player.src = "";
    }
}

/**
 * 7. 기타 UI 헬퍼 함수 (차트, 사이드바, 페이징)
 */
function renderDashboardMiniLogs() {
    const container = document.getElementById('log-container');
    if (!container) return;

    const logs = JSON.parse(localStorage.getItem('risk_logs')) || [];

    // 로그가 없을 때
    if (logs.length === 0) {
        container.innerHTML = '<div class="log-placeholder">현재 감지된 위험이 없습니다.</div>';
        return;
    }

    // 최근 5개 로그 렌더링 (클릭 이벤트 추가)
    container.innerHTML = logs.slice(0, 5).map(l => `
        <div class="log-item ${l.className}" 
             onclick="openVideoModal('${l.videoUrl}', '${l.date} ${l.time}')" 
             style="cursor: pointer;">
            <span><strong>[Lv.${l.level}]</strong> ${l.message}</span>
            <span style="font-size: 0.75rem; color: #999;">${l.time}</span>
        </div>
    `).join('');
}

function getFormattedTime(date) {
    const h = String(date.getHours()).padStart(2, '0');
    const m = String(date.getMinutes()).padStart(2, '0');
    const s = String(date.getSeconds()).padStart(2, '0');
    return `${h}:${m}:${s}`;
}

function initSidebar() {
    const menuBtn = document.getElementById('menu-btn');
    const closeBtn = document.getElementById('close-btn');
    const sideNav = document.getElementById('side-nav');
    const overlay = document.getElementById('overlay');

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
    container.innerHTML = '';

    const prevBtn = document.createElement('button');
    prevBtn.className = `pagination-btn ${currentPage === 1 ? 'disabled' : ''}`;
    prevBtn.innerText = '이전';
    prevBtn.onclick = () => renderHistory(currentPage - 1);

    const info = document.createElement('span');
    info.style.margin = "0 15px";
    info.innerText = `${currentPage} / ${totalPages}`;

    const nextBtn = document.createElement('button');
    nextBtn.className = `pagination-btn ${currentPage === totalPages ? 'disabled' : ''}`;
    nextBtn.innerText = '다음';
    nextBtn.onclick = () => renderHistory(currentPage + 1);

    container.appendChild(prevBtn);
    container.appendChild(info);
    container.appendChild(nextBtn);
}

function renderRiskChart(allLogs) {
    const canvas = document.getElementById('riskChart');
    if (!canvas || typeof Chart === 'undefined') return;

    let labels = [];
    let dataCounts = [];
    let chartTitle = selectedFilterDate ? `${selectedFilterDate} 시간대별 추이` : "최근 7일간 추이";

    if (selectedFilterDate) {
        labels = ["00", "02", "04", "06", "08", "10", "12", "14", "16", "18", "20", "22"];
        dataCounts = new Array(12).fill(0);
        allLogs.filter(l => l.isoDate === selectedFilterDate).forEach(l => {
            const idx = Math.floor(l.hour / 2);
            dataCounts[idx]++;
        });
    } else {
        for (let i = 6; i >= 0; i--) {
            const d = new Date();
            d.setDate(d.getDate() - i);
            const iso = d.toISOString().split('T')[0];
            labels.push((d.getMonth() + 1) + "/" + d.getDate());
            dataCounts.push(allLogs.filter(l => l.isoDate === iso).length);
        }
    }

    const existingChart = Chart.getChart("riskChart");
    if (existingChart) existingChart.destroy();

    new Chart(canvas.getContext('2d'), {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: '감지 횟수',
                data: dataCounts,
                borderColor: '#1d1d1f',
                backgroundColor: 'rgba(29, 29, 31, 0.05)',
                borderWidth: 2,
                fill: true,
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false }, title: { display: true, text: chartTitle } },
            scales: { y: { beginAtZero: true, ticks: { stepSize: 1 } } }
        }
    });
}