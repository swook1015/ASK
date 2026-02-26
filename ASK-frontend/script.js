/**
 * RISK-WATCH AI: 통합 제어 스크립트 
 * 1. 시간 형식: 00:00:00 (엄격한 24시간제)
 * 2. 상태 구성: Level 1(정상), Level 3(위험)
 * 3. 긴급 신고 활성화: 위험 감지 시 30초간 유지
 */

/* 1. 글로벌 변수 설정 */
let currentPage = 1;
const rowsPerPage = 10;
let selectedFilterDate = ""; 
let emergencyActiveTimer = null; // 긴급 버튼 타이머 관리 변수

/* 2. 초기 로드 및 이벤트 리스너 */
document.addEventListener('DOMContentLoaded', () => {
    initSidebar();
    initWebcam();
    
    if (document.getElementById('status-card')) {
        renderDashboardMiniLogs();
        startDashboardSimulation();
    }

    if (document.getElementById('history-body')) {
        const datePicker = document.getElementById('date-filter');
        if (datePicker) {
            datePicker.addEventListener('change', (e) => {
                selectedFilterDate = e.target.value;
                renderHistory(1);
            });
        }
        renderHistory(1);
    }
});

/** * [공통] 시간 포맷 함수 (HH:mm:ss) */
function getFormattedTime(date) {
    const h = String(date.getHours()).padStart(2, '0');
    const m = String(date.getMinutes()).padStart(2, '0');
    const s = String(date.getSeconds()).padStart(2, '0');
    return `${h}:${m}:${s}`;
}

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

/* 5. 실시간 대시보드 로직 (30초 타이머 로직 포함) */
function updateDashboardUI(level) {
    const statusCard = document.getElementById('status-card');
    const statusText = document.getElementById('status-text');
    const emergencyBox = document.getElementById('emergency-actions');
    const lastUpdate = document.getElementById('last-update');

    if (!statusCard) return;

    lastUpdate.innerText = getFormattedTime(new Date());

    const config = {
        1: { type: "정상", cls: "status-normal", tag: "info" },
        3: { type: "위험", cls: "status-danger", tag: "danger" }
    };
    
    const current = config[level];

    if (level === 1) {
        // [정상 상태]
        statusCard.className = current.cls;
        statusText.innerText = "감지 중";
        
        // 현재 활성화된 30초 타이머가 없을 때만 버튼을 비활성화 함
        if (emergencyBox && !emergencyActiveTimer) {
            emergencyBox.className = "emergency-box disabled";
        }
    } else {
        // [위험 상태 (Level 3)]
        statusCard.className = current.cls;
        statusText.innerText = current.type + " 상태";
        
        if (emergencyBox) {
            emergencyBox.className = "emergency-box active";
            
            // 기존에 돌아가던 타이머가 있다면 취소하고 새로 30초 시작 (중첩 방지)
            if (emergencyActiveTimer) {
                clearTimeout(emergencyActiveTimer);
            }
            
            // 30초(30000ms) 후에 버튼을 다시 비활성화 상태로 돌림
            emergencyActiveTimer = setTimeout(() => {
                emergencyBox.className = "emergency-box disabled";
                emergencyActiveTimer = null; // 타이머 종료 알림
                console.log("30초 경과: 긴급 신고 버튼이 비활성화되었습니다.");
            }, 30000);
        }
        
        saveLogToStorage(level, current.type, `${current.type} 상황이 감지됨`, current.tag);
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

/* 6. 데이터 저장 */
function saveLogToStorage(level, type, message, className) {
    try {
        const logs = JSON.parse(localStorage.getItem('risk_logs')) || [];
        const now = new Date();
        const newLog = {
            id: Date.now(),
            level: level,
            isoDate: now.toISOString().split('T')[0], 
            hour: now.getHours(),
            date: now.toLocaleDateString(),
            time: getFormattedTime(now),
            type, message, className
        };
        logs.unshift(newLog);
        localStorage.setItem('risk_logs', JSON.stringify(logs.slice(0, 500)));
    } catch (e) { console.error(e); }
}

/* 7. 히스토리 렌더링 */
function renderHistory(page = 1) {
    const tbody = document.getElementById('history-body');
    if (!tbody) return;

    const allLogs = JSON.parse(localStorage.getItem('risk_logs')) || [];
    let filteredLogs = allLogs;

    if (selectedFilterDate) {
        filteredLogs = allLogs.filter(l => l.isoDate === selectedFilterDate);
    }

    const totalPages = Math.ceil(filteredLogs.length / rowsPerPage) || 1;
    if (page < 1) page = 1;
    if (page > totalPages) page = totalPages;
    currentPage = page;

    const startIndex = (currentPage - 1) * rowsPerPage;
    const pagedLogs = filteredLogs.slice(startIndex, startIndex + rowsPerPage);

    tbody.innerHTML = pagedLogs.length === 0 
        ? `<tr><td colspan="3" style="padding:40px;">데이터가 없습니다.</td></tr>` 
        : pagedLogs.map(l => `
            <tr>
                <td>${l.date}<br>${l.time}</td>
                <td><span class="tag ${l.className}">${l.type}</span></td>
                <td style="text-align: center;">${l.message}</td>
            </tr>
        `).join('');

    updatePaginationUI(totalPages);
    renderRiskChart(allLogs); 
}

/* 8. 그래프 렌더링 (직선형) */
function renderRiskChart(allLogs) {
    const canvas = document.getElementById('riskChart');
    if (!canvas) return;

    let labels = [];
    let dataCounts = [];
    let chartTitle = "";

    if (selectedFilterDate) {
        chartTitle = `${selectedFilterDate} 시간대별 추이 (24H)`;
        labels = ["00-02", "02-04", "04-06", "06-08", "08-10", "10-12", "12-14", "14-16", "16-18", "18-20", "20-22", "22-24"];
        dataCounts = new Array(12).fill(0);
        
        const targetDayLogs = allLogs.filter(l => l.isoDate === selectedFilterDate);
        targetDayLogs.forEach(l => {
            if (l.hour !== undefined) {
                const index = Math.floor(l.hour / 2);
                dataCounts[index]++;
            }
        });
    } else {
        chartTitle = "최근 7일간 위험 감지 추이";
        for (let i = 6; i >= 0; i--) {
            const d = new Date();
            d.setDate(d.getDate() - i);
            const iso = d.toISOString().split('T')[0];
            const displayDate = (d.getMonth() + 1) + "/" + d.getDate();
            
            labels.push(displayDate);
            const count = allLogs.filter(l => l.isoDate === iso).length;
            dataCounts.push(count);
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
                tension: 0,
                pointBackgroundColor: '#1d1d1f',
                pointRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { beginAtZero: true, ticks: { stepSize: 1 } },
                x: { grid: { display: false } }
            },
            plugins: {
                legend: { display: false },
                title: { display: true, text: chartTitle, font: { size: 14 } }
            }
        }
    });
}

function updatePaginationUI(totalPages) {
    const container = document.getElementById('pagination');
    if (!container) return;
    container.innerHTML = '';
    const prevBtn = document.createElement('button');
    prevBtn.className = `pagination-btn ${currentPage === 1 ? 'disabled' : ''}`;
    prevBtn.innerText = '이전';
    prevBtn.onclick = () => { if(currentPage > 1) renderHistory(currentPage - 1); };
    const info = document.createElement('span');
    info.style.margin = "0 15px";
    info.innerText = `${currentPage} / ${totalPages}`;
    const nextBtn = document.createElement('button');
    nextBtn.className = `pagination-btn ${currentPage === totalPages ? 'disabled' : ''}`;
    nextBtn.innerText = '다음';
    nextBtn.onclick = () => { if(currentPage < totalPages) renderHistory(currentPage + 1); };
    container.appendChild(prevBtn);
    container.appendChild(info);
    container.appendChild(nextBtn);
}

function startDashboardSimulation() {
    setInterval(() => {
        const rand = Math.random();
        const level = rand > 0.90 ? 3 : 1; 
        updateDashboardUI(level);
    }, 5000);
}