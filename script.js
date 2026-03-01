import mobiusClient from './api.js';

let currentPage = 1;
const rowsPerPage = 10;

/* 2. 초기 로드 및 이벤트 리스너 */
document.addEventListener('DOMContentLoaded', () => {
    initSidebar();
    initWebcam();

    // 대시보드 페이지 초기화
    if (document.getElementById('status-card')) {
        renderDashboardMiniLogs();
        startDashboardSimulation();
    }

    // 히스토리 페이지 초기화
    if (document.getElementById('history-body')) {
        renderHistory(1);

        // 이미지 속 '초기화' 버튼 이벤트 바인딩
        const resetBtn = document.querySelector('.history-section button, #reset-logs');
        if (resetBtn) resetBtn.addEventListener('click', clearLogs);
    }
});

/* 3. 사이드바 제어 */
function initSidebar() {
    const menuBtn = document.getElementById('menu-btn');
    const closeBtn = document.getElementById('close-btn');
    const sideNav = document.getElementById('side-nav');
    const overlay = document.getElementById('overlay');

    if (menuBtn && sideNav && overlay) {
        const toggleNav = (isOpen) => {
            sideNav.classList.toggle('active', isOpen);
            overlay.classList.toggle('active', isOpen);
            document.body.style.overflow = isOpen ? 'hidden' : '';
        };
        menuBtn.addEventListener('click', () => toggleNav(true));
        if (closeBtn) closeBtn.addEventListener('click', () => toggleNav(false));
        overlay.addEventListener('click', () => toggleNav(false));
    }
}

/* 4. 웹캠 연동 */
function initWebcam() {
    const video = document.getElementById('webcam');
    if (!video) return;

    video.setAttribute('playsinline', '');
    video.setAttribute('muted', '');

    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false })
            .then(stream => { video.srcObject = stream; video.play(); })
            .catch(err => console.error("Webcam Error: ", err));
    }
}

/* 5. 실시간 대시보드 로직 (등급 데이터 주입) */
function updateDashboardUI(level) {
    const statusCard = document.getElementById('status-card');
    const statusText = document.getElementById('status-text');
    const emergencyBox = document.getElementById('emergency-actions');
    const lastUpdate = document.getElementById('last-update');

    if (!statusCard) return;

    lastUpdate.innerText = new Date().toLocaleTimeString();

    // 등급별 데이터 매핑
    const config = {
        1: { type: "정상", cls: "status-normal", tag: "info" },
        2: { type: "주의", cls: "status-warning", tag: "warning" },
        3: { type: "위험", cls: "status-danger", tag: "danger" }
    };

    const current = config[level];

    if (level === 1) {
        statusCard.className = current.cls;
        statusText.innerText = "감지 중";
        if (emergencyBox) emergencyBox.className = "emergency-box disabled";
    } else {
        statusCard.className = current.cls;
        statusText.innerText = current.type + " 상태";
        if (level === 3 && emergencyBox) emergencyBox.className = "emergency-box active";

        // 중요: level 숫자를 명시적으로 전달
        saveLogToStorage(level, current.type, `${current.type} 상황이 시스템에 의해 감지됨`, current.tag);
        renderDashboardMiniLogs();
    }
}

function renderDashboardMiniLogs() {
    const container = document.getElementById('log-container');
    if (!container) return;
    const logs = JSON.parse(localStorage.getItem('risk_logs')) || [];

    container.innerHTML = logs.slice(0, 5).map(l => `
        <div class="log-item ${l.className || 'info'}">
            <span><strong>[Lv.${l.level || '-'}]</strong> ${l.message}</span>
            <span style="font-size: 0.75rem; color: #999;">${l.time}</span>
        </div>
    `).join('') || '<div style="padding:20px; text-align:center;">로그 없음</div>';
}

/* 6. 데이터 저장 (데이터 구조 보강) */
function saveLogToStorage(level, type, message, className) {
    try {
        const logs = JSON.parse(localStorage.getItem('risk_logs')) || [];
        const newLog = {
            id: Date.now(),
            level: level, // 등급 숫자 저장
            date: new Date().toLocaleDateString(),
            time: new Date().toLocaleTimeString(),
            type: type,
            message: message,
            className: className
        };
        logs.unshift(newLog);
        localStorage.setItem('risk_logs', JSON.stringify(logs.slice(0, 100)));
    } catch (e) {
        console.error("Storage Error: ", e);
    }
}

/* 7. 히스토리 관리 (이미지의 빈 컬럼 해결) */
function renderHistory(page = 1) {
    const tbody = document.getElementById('history-body');
    if (!tbody) return;

    const logs = JSON.parse(localStorage.getItem('risk_logs')) || [];
    const totalPages = Math.ceil(logs.length / rowsPerPage) || 1;

    if (page < 1) page = 1;
    if (page > totalPages) page = totalPages;
    currentPage = page;

    const startIndex = (currentPage - 1) * rowsPerPage;
    const pagedLogs = logs.slice(startIndex, startIndex + rowsPerPage);

    tbody.innerHTML = pagedLogs.length === 0
        ? '<tr><td colspan="3" style="padding:40px;">저장된 기록이 없습니다.</td></tr>'
        : pagedLogs.map(l => `
            <tr>
                <td style="font-size: 0.8rem;">${l.date}<br>${l.time}</td>
                <td>
                    <span class="tag ${l.className || 'info'}">
                        Lv.${l.level || '2'} ${l.type || '주의'}
                    </span>
                </td>
                <td style="text-align: left; padding-left: 10px; font-size: 0.85rem;">${l.message}</td>
            </tr>
        `).join('');

    updatePaginationUI(totalPages);
    renderRiskChart(logs);
}

/* 8. 데이터 초기화 기능 (이미지의 '초기화' 버튼 대응) */
function clearLogs() {
    if (confirm("모든 히스토리 데이터를 삭제하시겠습니까?")) {
        localStorage.removeItem('risk_logs');
        renderHistory(1);
        if (document.getElementById('log-container')) renderDashboardMiniLogs();
        alert("데이터가 초기화되었습니다.");
    }
}

function updatePaginationUI(totalPages) {
    const container = document.getElementById('pagination');
    if (!container) return;

    container.innerHTML = '';
    const prevBtn = document.createElement('button');
    prevBtn.className = `pagination-btn ${currentPage === 1 ? 'disabled' : ''}`;
    prevBtn.innerText = '이전';
    prevBtn.onclick = () => { if (currentPage > 1) renderHistory(currentPage - 1); };

    const info = document.createElement('span');
    info.style.margin = "0 15px";
    info.innerText = `${currentPage} / ${totalPages}`;

    const nextBtn = document.createElement('button');
    nextBtn.className = `pagination-btn ${currentPage === totalPages ? 'disabled' : ''}`;
    nextBtn.innerText = '다음';
    nextBtn.onclick = () => { if (currentPage < totalPages) renderHistory(currentPage + 1); };

    container.appendChild(prevBtn);
    container.appendChild(info);
    container.appendChild(nextBtn);
}

function renderRiskChart(logs) {
    const canvas = document.getElementById('riskChart');
    if (!canvas) return;
    const stats = {};
    logs.forEach(l => { stats[l.date] = (stats[l.date] || 0) + 1; });
    const existingChart = Chart.getChart("riskChart");
    if (existingChart) existingChart.destroy();
    new Chart(canvas.getContext('2d'), {
        type: 'bar',
        data: {
            labels: Object.keys(stats).reverse(),
            datasets: [{ label: '위험 감지', data: Object.values(stats).reverse(), backgroundColor: '#000' }]
        },
        options: { responsive: true, maintainAspectRatio: false }
    });
}

function startDashboardSimulation() {
    setInterval(() => {
        const rand = Math.random();
        const level = rand > 0.85 ? (rand > 0.96 ? 3 : 2) : 1;
        updateDashboardUI(level);
    }, 5000);
}