import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
import av
import cv2
import numpy as np
from scipy.signal import butter, filtfilt, welch, find_peaks
import time
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from PIL import Image
import io
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# -----------------------------------------------------------------------------
# PAGE CONFIG — MUST be the very first Streamlit command in the script,
# before any other st.* call (including st.error inside functions called
# below), or Streamlit raises a StreamlitAPIException.
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Nabz Pro - Advanced Health Monitoring",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# WEBRTC ICE CONFIGURATION — TURN (not just STUN)
# -----------------------------------------------------------------------------
# STUN alone only works when direct peer-to-peer UDP is possible. If either
# side is behind a NAT/firewall that blocks that (common on cloud hosts and
# many corporate/school networks), the connection needs to be relayed through
# a TURN server instead. This fetches short-lived TURN credentials from
# Twilio at runtime. Requires TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN to be
# set in Streamlit Cloud's Settings -> Secrets.
from twilio.rest import Client

@st.cache_resource(ttl=3000)  # Twilio tokens are valid for ~1 hour; refresh before they expire
def get_ice_servers():
    try:
        client = Client(
            st.secrets["TWILIO_API_KEY_SID"],
            st.secrets["TWILIO_API_KEY_SECRET"],
            st.secrets["TWILIO_ACCOUNT_SID"],
        )
        token = client.tokens.create()
        return token.ice_servers
    except Exception as e:
        st.error(f"Could not fetch TURN credentials from Twilio: {e}")
        # Fall back to STUN-only so the app doesn't crash outright,
        # even though it likely won't connect on restrictive networks.
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

RTC_CONFIGURATION = RTCConfiguration({"iceServers": get_ice_servers()})

# -----------------------------------------------------------------------------
# GLOBAL STATE MANAGEMENT
# -----------------------------------------------------------------------------
class AppState:
    def __init__(self):
        self.bpm = 0.0
        self.hrv = 0.0
        self.rr = 0.0
        self.stress_index = 0.0
        self.sqi = 0.0
        self.status = "Awaiting Face..."
        self.ppg_signal = []
        self.face_photo = None
        self.patient_name = ""
        self.recording_start = None
        self.data_log = []

app_state = AppState()

# -----------------------------------------------------------------------------
# OPENCV HAAR CASCADE SETUP (Replaces MediaPipe)
# -----------------------------------------------------------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# -----------------------------------------------------------------------------
# VIDEO PROCESSOR WITH REAL-TIME rPPG
# -----------------------------------------------------------------------------
class rPPGProcessor(VideoProcessorBase):
    def __init__(self):
        self.green_signal = []
        self.fps = 30.0
        self.last_time = time.time()
        self.frame_count = 0
        self.last_faces = []  # cache last detection result between throttled runs

    def recv(self, frame):
        img = frame.to_ndarray(format="rgb24")
        h, w, _ = img.shape

        # Calculate FPS
        current_time = time.time()
        self.frame_count += 1
        if current_time - self.last_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_time)
            self.frame_count = 0
            self.last_time = current_time

        # Throttle face detection to every 3rd frame. Haar cascades are
        # relatively expensive; running them on every frame at 30fps adds
        # unnecessary CPU load once other processing (signal math) is also
        # competing for CPU time. The face barely moves frame-to-frame, so
        # reusing the last detection in between is visually seamless.
        if self.frame_count % 3 == 0 or len(self.last_faces) == 0:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            self.last_faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

        faces = self.last_faces

        if len(faces) > 0:
            app_state.status = "✓ Signal Acquired"
            for (x, y, w_face, h_face) in faces:
                # Extract Forehead ROI (Top 35% of the face bounding box)
                rx1 = int(x + w_face * 0.25)
                rx2 = int(x + w_face * 0.75)
                ry1 = int(y)
                ry2 = int(y + h_face * 0.35)

                if rx2 > rx1 and ry2 > ry1:
                    roi_img = img[ry1:ry2, rx1:rx2]
                    green_mean = np.mean(roi_img[:, :, 1])
                    self.green_signal.append((green_mean, current_time))

                    # Capture face photo (first good frame)
                    if app_state.face_photo is None:
                        app_state.face_photo = img[y:y+h_face, x:x+w_face].copy()

                    # Maintain 15-second buffer
                    if len(self.green_signal) > int(self.fps * 15):
                        self.green_signal.pop(0)

                    # Calculate metrics every second
                    if self.frame_count % max(1, int(self.fps)) == 0 and len(self.green_signal) > int(self.fps * 8):
                        self._calculate_all_metrics()

                    # Draw ROI
                    cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
                    cv2.putText(img, "ROI", (rx1, ry1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            app_state.status = "✗ No Face"

        # Overlay metrics
        cv2.putText(img, f"BPM: {app_state.bpm:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, f"SQI: {app_state.sqi:.0f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="rgb24")

    def _calculate_all_metrics(self):
        if len(self.green_signal) < 120:
            return

        signals = np.array([s[0] for s in self.green_signal])
        times = np.array([s[1] for s in self.green_signal])
        signals = signals - np.mean(signals)

        # Calculate actual sampling rate
        if len(times) > 1:
            actual_fs = 1.0 / np.mean(np.diff(times))
        else:
            actual_fs = 30.0

        # Bandpass filter for cardiac signal (0.75-3.0 Hz)
        nyquist = actual_fs / 2.0
        low, high = 0.75 / nyquist, 3.0 / nyquist
        b, a = butter(3, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, signals)

        # FFT for Heart Rate
        freqs, psd = welch(filtered_signal, actual_fs, nperseg=len(filtered_signal))
        valid_idx = np.where((freqs >= 0.75) & (freqs <= 3.0))[0]

        if len(valid_idx) > 0:
            peak_idx = valid_idx[np.argmax(psd[valid_idx])]
            app_state.bpm = freqs[peak_idx] * 60.0

            # Signal Quality Index
            max_power = np.max(psd[valid_idx])
            mean_power = np.mean(psd)
            app_state.sqi = min(100.0, (max_power / mean_power) * 15.0) if mean_power > 0 else 0

            # Peak detection for HRV
            peaks, _ = find_peaks(filtered_signal, distance=int(0.5 * actual_fs), height=0)
            if len(peaks) > 2:
                rr_intervals = np.diff(peaks) / actual_fs  # in seconds
                diff_rr = np.diff(rr_intervals)
                app_state.hrv = np.sqrt(np.mean(diff_rr**2))

                # Stress Index
                normalized_hrv = np.clip(app_state.hrv / 100.0, 0, 1)
                app_state.stress_index = (1 - normalized_hrv) * 100

        # Respiratory Rate (0.15-0.5 Hz band)
        low_rr, high_rr = 0.15 / nyquist, 0.5 / nyquist
        b_rr, a_rr = butter(3, [low_rr, high_rr], btype='band')
        rr_signal = filtfilt(b_rr, a_rr, signals)

        freqs_rr, psd_rr = welch(rr_signal, actual_fs, nperseg=len(rr_signal))
        valid_rr_idx = np.where((freqs_rr >= 0.15) & (freqs_rr <= 0.5))[0]

        if len(valid_rr_idx) > 0:
            rr_peak_idx = valid_rr_idx[np.argmax(psd_rr[valid_rr_idx])]
            app_state.rr = freqs_rr[rr_peak_idx] * 60.0

        # Store signal for visualization
        app_state.ppg_signal = filtered_signal[-200:].tolist()

        # Log data
        if app_state.recording_start is None:
            app_state.recording_start = datetime.now()

        app_state.data_log.append({
            'time': datetime.now(),
            'bpm': app_state.bpm,
            'hrv': app_state.hrv,
            'rr': app_state.rr,
            'stress': app_state.stress_index
        })

        if len(app_state.data_log) > 60:
            app_state.data_log.pop(0)

# -----------------------------------------------------------------------------
# PDF REPORT GENERATION
# -----------------------------------------------------------------------------
def generate_pdf_report():
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=18)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='CenterTitle', alignment=TA_CENTER, fontSize=24, fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='SubTitle', alignment=TA_CENTER, fontSize=12, fontName='Helvetica'))
    styles.add(ParagraphStyle(name='NormalLeft', alignment=TA_LEFT, fontSize=10))

    elements = []

    # Header
    elements.append(Paragraph("NABZ HEALTH TELEMETRY REPORT", styles['CenterTitle']))
    elements.append(Paragraph("Contactless Vital Signs Analysis", styles['SubTitle']))
    elements.append(Spacer(1, 0.3*inch))

    # Patient Info Table
    patient_info = [
        ['Patient Name:', app_state.patient_name if app_state.patient_name else 'Not Provided'],
        ['Date/Time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ['Recording Duration:', f"{len(app_state.data_log)} seconds"]
    ]

    t1 = Table(patient_info, colWidths=[2*inch, 4*inch])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    elements.append(t1)
    elements.append(Spacer(1, 0.3*inch))

    # Vital Signs Summary Table
    avg_bpm = np.mean([d['bpm'] for d in app_state.data_log[-30:]]) if app_state.data_log else 0
    avg_hrv = np.mean([d['hrv'] for d in app_state.data_log[-30:]]) if app_state.data_log else 0
    avg_rr = np.mean([d['rr'] for d in app_state.data_log[-30:]]) if app_state.data_log else 0
    avg_stress = np.mean([d['stress'] for d in app_state.data_log[-30:]]) if app_state.data_log else 0

    vitals_data = [
        ['Parameter', 'Value', 'Normal Range', 'Status'],
        ['Heart Rate (BPM)', f'{avg_bpm:.1f}', '60-100', 'Normal' if 60 <= avg_bpm <= 100 else 'Check'],
        ['Heart Rate Variability (ms)', f'{avg_hrv:.1f}', '20-100', 'Normal' if 20 <= avg_hrv <= 100 else 'Check'],
        ['Respiratory Rate (BrPM)', f'{avg_rr:.1f}', '12-20', 'Normal' if 12 <= avg_rr <= 20 else 'Check'],
        ['Stress Index', f'{avg_stress:.1f}/100', '<50', 'Low' if avg_stress < 50 else 'Elevated']
    ]

    t2 = Table(vitals_data, colWidths=[2.2*inch, 1.5*inch, 1.5*inch, 1.3*inch])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    elements.append(t2)
    elements.append(Spacer(1, 0.3*inch))

    # Patient Photo
    if app_state.face_photo is not None:
        elements.append(Paragraph("Patient Photo (Captured During Scan)", styles['NormalLeft']))
        elements.append(Spacer(1, 0.1*inch))

        photo_img = Image.fromarray(app_state.face_photo)
        photo_buffer = io.BytesIO()
        photo_img.save(photo_buffer, format='JPEG')
        photo_buffer.seek(0)

        photo = RLImage(photo_buffer, width=2*inch, height=2*inch)
        elements.append(photo)
        elements.append(Spacer(1, 0.3*inch))

    # Graphs
    elements.append(Paragraph("Real-Time Signal Analysis", styles['NormalLeft']))
    elements.append(Spacer(1, 0.1*inch))

    if len(app_state.ppg_signal) > 0:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(app_state.ppg_signal, color='#06B6D4', linewidth=2)
        ax.set_title('Photoplethysmogram (PPG) Waveform', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (samples)')
        ax.set_ylabel('Amplitude (a.u.)')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f9fa')
        plt.tight_layout()

        graph_buffer = io.BytesIO()
        plt.savefig(graph_buffer, format='png', dpi=150)
        graph_buffer.seek(0)
        plt.close()

        ppg_graph = RLImage(graph_buffer, width=6*inch, height=2.5*inch)
        elements.append(ppg_graph)
        elements.append(Spacer(1, 0.2*inch))

    if len(app_state.data_log) > 10:
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        times = range(len(app_state.data_log))
        bpm_values = [d['bpm'] for d in app_state.data_log]

        ax2.plot(times, bpm_values, color='#EC4899', linewidth=2, label='Heart Rate')
        ax2.axhline(y=60, color='green', linestyle='--', alpha=0.5, linewidth=1)
        ax2.axhline(y=100, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax2.set_title('Heart Rate Trend (Last 60 seconds)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Time (seconds ago)')
        ax2.set_ylabel('BPM')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor('#f8f9fa')
        plt.tight_layout()

        trend_buffer = io.BytesIO()
        plt.savefig(trend_buffer, format='png', dpi=150)
        trend_buffer.seek(0)
        plt.close()

        trend_graph = RLImage(trend_buffer, width=6*inch, height=2.5*inch)
        elements.append(trend_graph)

    # Footer
    elements.append(Spacer(1, 0.5*inch))
    elements.append(Paragraph("="*72, styles['SubTitle']))
    elements.append(Paragraph("This report is generated automatically by Nabz AI Health Monitoring System", styles['NormalLeft']))
    elements.append(Paragraph("For medical concerns, please consult a healthcare professional.", styles['NormalLeft']))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# -----------------------------------------------------------------------------
# STREAMLIT UI
# -----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;700;900&display=swap');
* { font-family: 'Nunito', sans-serif !important; }

.hero-banner {
    background: linear-gradient(135deg, #06B6D4 0%, #3B82F6 50%, #EC4899 100%);
    padding: 25px;
    border-radius: 20px;
    color: white;
    text-align: center;
    box-shadow: 0 10px 30px rgba(6, 182, 212, 0.3);
    margin-bottom: 20px;
}

.metric-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.1);
    text-align: center;
    transition: transform 0.2s;
}

.metric-value { font-size: 2.5rem; font-weight: 900; margin: 10px 0; }
.metric-label { font-size: 0.9rem; color: #94a3b8; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }
.metric-unit { font-size: 1rem; color: #64748b; margin-top: 5px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-banner">
    <h1 style="margin:0; font-size:2.5rem; font-weight:900;">💓 Nabz Pro Health Monitor</h1>
    <p style="margin:8px 0 0 0; font-size:1.1rem; font-weight:600; opacity:0.95;">
        AI-Powered Contactless Vital Signs & Stress Analysis
    </p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/heart-with-pulse.png", width=100)
    st.title("⚙️ Control Panel")
    app_state.patient_name = st.text_input("Patient Name:", placeholder="Enter full name")
    st.markdown("---")
    st.subheader("📋 Instructions")
    st.info("""
    1. Enter your name above
    2. Click START below
    3. Sit still and face the camera
    4. Wait 10-15 seconds for calibration
    5. Download your medical report
    """)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📷 Live Camera Feed")
    ctx = webrtc_streamer(
        key="rppg-pro",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=rPPGProcessor,
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
        async_processing=True,
    )
    if ctx.state.playing:
        st.success("✅ Camera Active - Recording Vitals")
    else:
        st.warning("⚠️ Click START above to begin monitoring")

with col2:
    st.subheader("📊 Real-Time Dashboard")

    # Wrapped in a fragment with run_every so this section re-renders on its
    # own every second and picks up the latest app_state values, without
    # rerunning the whole page (which would otherwise restart/interrupt the
    # webrtc_streamer component and drop the active connection).
    @st.fragment(run_every=1)
    def render_dashboard(playing: bool):
        has_data = len(app_state.data_log) > 0

        if not playing and not has_data:
            st.info("👆 Start camera to see metrics")
            return

        if not playing and has_data:
            st.info("⏹️ Recording stopped — showing your last results below.")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""<div class="metric-card" style="border-bottom: 4px solid #EC4899;">
                <div class="metric-label">Heart Rate</div>
                <div class="metric-value" style="color: #EC4899;">{app_state.bpm:.0f}</div>
                <div class="metric-unit">BPM</div></div>""", unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""<div class="metric-card" style="border-bottom: 4px solid #10B981;">
                <div class="metric-label">HRV</div>
                <div class="metric-value" style="color: #10B981;">{app_state.hrv:.1f}</div>
                <div class="metric-unit">milliseconds</div></div>""", unsafe_allow_html=True)

        col_c, col_d = st.columns(2)
        with col_c:
            st.markdown(f"""<div class="metric-card" style="border-bottom: 4px solid #06B6D4;">
                <div class="metric-label">Respiration</div>
                <div class="metric-value" style="color: #06B6D4;">{app_state.rr:.1f}</div>
                <div class="metric-unit">breaths/min</div></div>""", unsafe_allow_html=True)
        with col_d:
            stress_color = "#10B981" if app_state.stress_index < 50 else "#F59E0B" if app_state.stress_index < 75 else "#EF4444"
            stress_label = "Low" if app_state.stress_index < 50 else "Moderate" if app_state.stress_index < 75 else "High"
            st.markdown(f"""<div class="metric-card" style="border-bottom: 4px solid {stress_color};">
                <div class="metric-label">Stress Index</div>
                <div class="metric-value" style="color: {stress_color};">{app_state.stress_index:.0f}</div>
                <div class="metric-unit">{stress_label}</div></div>""", unsafe_allow_html=True)

        st.markdown("---")
        sqi_color = "green" if app_state.sqi > 70 else "orange" if app_state.sqi > 40 else "red"
        st.markdown(f"**Signal Quality:** :{sqi_color}[{app_state.sqi:.1f}%] - {app_state.status}")

        st.markdown("---")
        st.markdown("**📈 Live PPG Waveform**")
        if len(app_state.ppg_signal) > 0:
            st.line_chart(app_state.ppg_signal, use_container_width=True, height=200)

        st.markdown("---")
        if len(app_state.data_log) > 30:
            pdf_buffer = generate_pdf_report()
            st.download_button(
                label="📄 Download Medical Report (PDF)",
                data=pdf_buffer,
                file_name=f"Nabz_Report_{app_state.patient_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        elif playing:
            st.warning(f"⏳ Collecting data... {len(app_state.data_log)}/30 seconds")
        else:
            st.warning("Recording was too short to generate a full report (needs at least 30 seconds of data).")

    render_dashboard(ctx.state.playing)

st.markdown("""
<div style="text-align: center; padding: 20px; color: #64748b; border-top: 1px solid #1e293b; margin-top: 30px;">
    <p><strong>Nabz Pro Health Monitor</strong> | Powered by Remote Photoplethysmography (rPPG) Technology</p>
    <p style="font-size: 0.85rem;">For wellness monitoring only. Not a substitute for professional medical diagnosis.</p>
</div>
""", unsafe_allow_html=True)
