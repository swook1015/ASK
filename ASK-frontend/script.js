/**
 * RISK-WATCH AI: 통합 제어 스크립트 
 * 1. 서버 연동: 192.168.0.2:8080 (Mobius)
 * 2. 상태 구성: Level 1(정상), Level 3(위험)
 * 3. 기능: 서버 데이터(camId, v, t) 기반 UI 갱신 및 로그 저장
 */

/* 1. 글로벌 변수 설정 */
let currentPage = 1;
const rowsPerPage = 10;
let selectedFilterDate = ""; 
let emergencyActiveTimer = null; // 긴급 버튼 타이머 관리 변수
let lastEventTime = null;       // 중복 데이터 처리 방지 변수 (이벤트 시간 t 기준)

/* 2. 초기 로드 및 이벤트 리스너 */
document.addEventListener('DOMContentLoaded', () => {
    initSidebar();
    initWebcam();
    
    // 대시보드 요소가 있을 경우에만 폴링 및 로그 렌더링 시작
    if (document.getElementById('status-card')) {
        renderDashboardMiniLogs();
        // 실제 Mobius 서버 데이터 폴링 시작
        startMobiusPolling();
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

/* 4. 서버 연동 액션캠 스트리밍 */
function initWebcam() {
    const webcamElement = document.getElementById('webcam');
    if (!webcamElement) return;

    // 실제 스트리밍 엔드포인트 주소
    const actionCamStreamUrl = "http://192.168.0.2:8080/video_feed"; 

    if (webcamElement.tagName.toLowerCase() === 'video') {
        webcamElement.src = actionCamStreamUrl;
        webcamElement.setAttribute('playsinline', '');
        webcamElement.setAttribute('muted', '');
        webcamElement.play().catch(err => {
            console.error("액션캠 영상 재생 실패:", err);
        });
    } else if (webcamElement.tagName.toLowerCase() === 'img') {
        webcamElement.src = actionCamStreamUrl;
    }
}

/* 5. 실시간 대시보드 로직 (서버 데이터 기반 반영) */
function updateDashboardUI(level, camId, eventTimeStr) {
    const statusCard = document.getElementById('status-card');
    const statusText = document.getElementById('status-text');
    const emergencyBox = document.getElementById('emergency-actions');
    const lastUpdate = document.getElementById('last-update');

    if (!statusCard) return;

    // 서버에서 보내준 이벤트 발생 시간(t)을 Date 객체로 변환
    const serverDateObj = new Date(eventTimeStr);
    lastUpdate.innerText = getFormattedTime(serverDateObj);

    const config = {
        1: { type: "정상", cls: "status-normal", tag: "info" },
        3: { type: "위험", cls: "status-danger", tag: "danger" }
    };
    
    const current = config[level] || config[1];

    if (level === 1) {
        statusCard.className = `status-card ${current.cls}`;
        statusText.innerText = `[CAM ${camId}] 감지 중`; 
        
        if (emergencyBox && !emergencyActiveTimer) {
            emergencyBox.className = "emergency-box disabled";
        }
    } else {
        statusCard.className = `status-card ${current.cls}`;
        statusText.innerText = `[CAM ${camId}] ${current.type} 상태`; 
        
        if (emergencyBox) {
            emergencyBox.className = "emergency-box active";
            
            // 위험 상황 시 30초간 알림 활성화 후 자동 비활성화
            if (emergencyActiveTimer) {
                clearTimeout(emergencyActiveTimer);
            }
            
            emergencyActiveTimer = setTimeout(() => {
                emergencyBox.className = "emergency-box disabled";
                emergencyActiveTimer = null; 
            }, 30000);
        }
    }
    
    const logMessage = `[카메라 ${camId}] ${current.type} 상황 감지`;
    saveLogToStorage(level, current.type, logMessage, current.tag, serverDateObj);
    renderDashboardMiniLogs();
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
function saveLogToStorage(level, type, message, className, serverDateObj) {
    try {
        const logs = JSON.parse(localStorage.getItem('risk_logs')) || [];
        
        const newLog = {
            id: Date.now(),
            level: level,
            isoDate: serverDateObj.toISOString().split('T')[0], 
            hour: serverDateObj.getHours(),
            date: serverDateObj.toLocaleDateString(),
            time: getFormattedTime(serverDateObj),
            type, 
            message,
            className
        };
        logs.unshift(newLog);
        // 최대 500개까지만 로그 저장
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

/* 9. [Mobius 서버 연동] 192.168.0.2:8080 데이터 수신 로직 */
async function fetchMobiusData() {
    try {
        const raspIp = "192.168.0.2"; 
        const port = "7579";
        const url = `http://${raspIp}:${port}/Mobius/KETI3_DEMO/fall_state/la`;
        
        const headers = {
            // 매번 새로운 요청 ID를 생성하여 "RI is none" 에러 방지
            'X-M2M-RI': 'req-' + Date.now(), 
            // Mobius AE 설정과 일치해야 함
            'X-M2M-Origin': 'SKETI3_DEMO', 
            'Accept': 'application/json'
        };

        const response = await fetch(url, { 
            method: 'GET', 
            headers: headers,
            mode: 'cors' 
        });
        
        if (!response.ok) {
            throw new Error(`Mobius 서버 응답 에러: ${response.status}`);
        }
        
        const data = await response.json();
        
        // 데이터가 없는 경우 예외 처리
        if (!data['m2m:cin']) {
            console.warn("표시할 최신 데이터(cin)가 없습니다.");
            return;
        }

        let content = data['m2m:cin'].con;
        
        // con 데이터가 문자열 형태일 경우 JSON 파싱
        if (typeof content === 'string') {
            try {
                content = JSON.parse(content);
            } catch(e) {
                console.error("JSON 파싱 에러:", e);
            }
        }
        
        // 서버 데이터 구조 매핑 (camId, t, v)
        const camId = content.camId || "0";
        const eventTime = content.t; 
        const statusValue = content.v; 
        
        // v 값이 'fall'이면 Level 3, 아니면 Level 1
        let level = (statusValue === 'fall') ? 3 : 1;

        // 동일한 이벤트 시간에 대한 중복 처리 방지
        if (lastEventTime !== eventTime) {
            lastEventTime = eventTime; 
            updateDashboardUI(level, camId, eventTime);
            
            if (level === 3) {
                console.log(`🚨 [위험 감지] 카메라: ${camId} | 시간: ${eventTime}`);
            }
        }

    } catch (error) {
        console.error("데이터 수신 실패:", error);
    }
}

/** 2초 주기로 Mobius 서버 데이터를 확인하는 폴링 함수 */
function startMobiusPolling() {
    // 즉시 한 번 호출 후 인터벌 시작
    fetchMobiusData(); 
    setInterval(fetchMobiusData, 2000); 
}