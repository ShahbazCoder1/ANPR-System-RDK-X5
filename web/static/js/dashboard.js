document.addEventListener('DOMContentLoaded', () => {
    let lastSeenId = 0;

    // Fetch stats from API
    async function fetchStats() {
        try {
            const res = await fetch('/api/stats');
            if (!res.ok) return;
            const data = await res.json();

            document.getElementById('stat-today-vehicles').textContent = data.today_vehicles.toLocaleString();
            document.getElementById('stat-today-toll').textContent = `₹${data.today_toll.toLocaleString()}`;
            document.getElementById('stat-fps').textContent = data.fps.toFixed(1);
            document.getElementById('stat-uptime').textContent = data.uptime || '0h 0m';
            
            document.getElementById('stat-total-vehicles').textContent = data.total_vehicles.toLocaleString();
            document.getElementById('stat-total-revenue').textContent = `₹${data.total_toll.toLocaleString()}`;
        } catch (err) {
            console.warn('Error fetching stats:', err);
        }
    }

    // Fetch recent vehicle logs
    async function fetchRecentRecords() {
        try {
            const res = await fetch('/api/recent');
            if (!res.ok) return;
            const records = await res.json();

            const tbody = document.getElementById('records-tbody');
            document.getElementById('record-count').textContent = `${records.length} records`;

            if (records.length === 0) {
                tbody.innerHTML = '<tr><td colspan="7" class="loading-row">No vehicles recorded yet. Waiting for detections...</td></tr>';
                return;
            }

            let html = '';
            records.forEach((rec) => {
                const cropImg = rec.crop_filename 
                    ? `<img src="/api/crops/${rec.crop_filename}" class="plate-thumbnail" alt="Plate Crop" onerror="this.style.display='none';">`
                    : '<span style="color:#64748b;">N/A</span>';

                const ocrConf = (rec.ocr_conf * 100).toFixed(0);
                const yoloConf = (rec.yolo_conf * 100).toFixed(0);

                html += `
                    <tr>
                        <td>${cropImg}</td>
                        <td>
                            <div class="plate-badge">
                                <span class="ind-tag">IND</span>${rec.plate_number}
                            </div>
                        </td>
                        <td style="font-family: var(--font-mono); color: #cbd5e1;">${rec.time_str}</td>
                        <td class="toll-fee">₹${rec.toll_amount}</td>
                        <td style="font-family: var(--font-mono); font-size: 12px; color: #94a3b8;">
                            YOLO: ${yoloConf}% | OCR: ${ocrConf}%
                        </td>
                        <td style="color: #94a3b8; font-size: 12px;">${rec.source || 'mipi'}</td>
                        <td><span class="status-tag">✅ Toll Recorded</span></td>
                    </tr>
                `;
            });

            tbody.innerHTML = html;
        } catch (err) {
            console.warn('Error fetching recent records:', err);
        }
    }

    // Initial load
    fetchStats();
    fetchRecentRecords();

    // Auto-refresh timers (pause when tab is hidden to save board CPU/RAM)
    let statsTimer = setInterval(fetchStats, 1500);
    let recordsTimer = setInterval(fetchRecentRecords, 2000);

    document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
            clearInterval(statsTimer);
            clearInterval(recordsTimer);
        } else {
            fetchStats();
            fetchRecentRecords();
            statsTimer = setInterval(fetchStats, 1500);
            recordsTimer = setInterval(fetchRecentRecords, 2000);
        }
    });

    // Stream error auto-recovery
    const videoStream = document.getElementById('video-stream');
    if (videoStream) {
        videoStream.onerror = () => {
            setTimeout(() => {
                videoStream.src = '/api/feed?' + Date.now();
            }, 2000);
        };
    }
});
