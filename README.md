# 💓 Nabz Pro — Contactless AI Vitals Telemetry & Stress Monitor

**Nabz Pro** is a contactless AI health monitor that uses a standard webcam to track real-time vital signs—including heart rate, HRV, respiration, and stress levels—without any physical sensors. It leverages advanced Remote Photoplethysmography (rPPG) and digital signal processing to analyze subtle facial skin color changes for accurate, clinical-grade biometric insights. The app instantly generates a professional, downloadable PDF medical report complete with patient photos, live waveform graphs, and personalized health analytics.

---

## ✨ Key Features

*   **Non-Contact Vitals Monitoring:** Measures Heart Rate (BPM), Heart Rate Variability (HRV), Respiratory Rate (BrPM), and Stress Index using just a webcam.
*   **Advanced rPPG Pipeline:** Utilizes MediaPipe for facial ROI tracking, Green-channel extraction, Butterworth bandpass filtering, and Welch’s FFT for high-accuracy signal processing.
*   **Real-Time Dashboard:** Beautiful, responsive UI with live PPG waveform visualization and color-coded metric cards.
*   **Automated PDF Medical Reports:** Generates a professional, downloadable PDF report featuring the patient's captured photo, timestamp, vital statistics, and plotted trend graphs.
*   **Stress & Autonomic Analysis:** Calculates RMSSD for HRV and derives a real-time Stress Index based on sympathetic/parasympathetic nervous system balance.

---

## 🧠 How It Works (The Science)

1.  **Face Detection & ROI:** MediaPipe detects the face and isolates the forehead/cheek regions (Regions of Interest) where capillary beds are most visible.
2.  **Optical Extraction:** The system extracts the **Green color channel** from the ROI. Oxygenated hemoglobin absorbs green light, causing micro-fluctuations in pixel intensity with every heartbeat.
3.  **Signal Preprocessing:** A 3rd-order Butterworth Bandpass filter isolates the cardiac frequency (0.75–3.0 Hz) and respiratory frequency (0.15–0.5 Hz), removing ambient light noise and motion artifacts.
4.  **Spectral Analysis:** Fast Fourier Transform (FFT) via Welch’s method identifies the dominant frequency peaks to calculate BPM and BrPM.
5.  **HRV & Stress:** Time-domain analysis of the peak-to-peak intervals (RR intervals) calculates RMSSD, which is normalized to generate a 0-100 Stress Index.

---

## 🛠️ Tech Stack

*   **Frontend/UI:** Streamlit, Streamlit-WebRTC
*   **Computer Vision:** OpenCV, MediaPipe
*   **Signal Processing:** SciPy, NumPy
*   **Visualization:** Matplotlib
*   **Report Generation:** ReportLab, Pillow

---

## 🚀 Installation & Setup

### Prerequisites
*   Python 3.8 or higher
*   A working webcam (built-in or external)
*   A modern web browser (Chrome, Edge, or Firefox recommended for WebRTC)

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/nabz-pro.git
cd nabz-pro
