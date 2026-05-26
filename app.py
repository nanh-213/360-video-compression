import os
import math
import time
import json
import subprocess
import tempfile
from flask import Flask, render_template, request, jsonify

# Khai báo thư viện vẽ Distortion Map (chạy ngầm)
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static/output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ------------------------------------------------------------
# CÁC HÀM TRÍCH XUẤT THÔNG SỐ 
# ------------------------------------------------------------
def get_resolution(filepath):
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
           '-show_entries', 'stream=width,height', '-of', 'json', filepath]
    out = subprocess.check_output(cmd).decode()
    info = json.loads(out)['streams'][0]
    return int(info['width']), int(info['height'])

def get_duration_seconds(filepath):
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
           '-show_entries', 'format=duration', '-of', 'csv=p=0', filepath]
    return float(subprocess.check_output(cmd).decode().strip())

def calculate_psnr(original_path, decoded_path):
    psnr_log = tempfile.NamedTemporaryFile(suffix='.log', delete=False).name
    cmd = f'ffmpeg -i "{original_path}" -i "{decoded_path}" -lavfi "psnr=stats_file={psnr_log}" -f null - 2>/dev/null'
    subprocess.run(cmd, shell=True)
    
    psnr_avg = 0.0
    if os.path.exists(psnr_log):
        with open(psnr_log, 'r') as f:
            lines = f.readlines()
        for line in reversed(lines):
            if 'psnr_avg' in line:
                psnr_avg = float(line.split('psnr_avg:')[1].split()[0])
                break
        os.remove(psnr_log)
    return psnr_avg

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({"error": "Không tìm thấy file video"}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "Chưa chọn file"}), 400

    # 1. Lưu video gốc người dùng upload
    input_path = os.path.join(UPLOAD_FOLDER, 'input_video.mp4')
    file.save(input_path)

    WIDTH, HEIGHT = get_resolution(input_path)
    duration = get_duration_seconds(input_path)
    original_size_mb = os.path.getsize(input_path) / (1024 * 1024)
    bitrate_original = (original_size_mb * 8) / duration

    high_encoded = os.path.join(OUTPUT_FOLDER, 'video_encoded_high.mp4')
    low_encoded = os.path.join(OUTPUT_FOLDER, 'video_encoded_low.mp4')
    high_decoded = os.path.join(OUTPUT_FOLDER, 'video_decoded_high.mp4')
    low_decoded = os.path.join(OUTPUT_FOLDER, 'video_decoded_low.mp4')

    # 2. Tính toán Distortion Map ROI Chains & Vẽ bản đồ heatmap
    STRIP_H = 64
    num_strips = (HEIGHT + STRIP_H - 1) // STRIP_H
    distortionmap_high = []
    distortionmap_low = []
    
    map_weights = [] # Lưu data để xuất ảnh

    for i in range(num_strips):
        y_start = i * STRIP_H
        y_end = min(y_start + STRIP_H, HEIGHT)
        strip_h = y_end - y_start

        weights = []
        for py in range(y_start, y_end):
            lat = (py / (HEIGHT - 1) - 0.5) * math.pi
            weights.append(math.cos(lat))
        avg_w = sum(weights) / len(weights)

        qoffset_high = -0.5 * avg_w
        qoffset_low = -0.1 * avg_w
        
        map_weights.append(qoffset_high) 

        distortionmap_high.append(f"addroi=x=0:y={y_start}:w={WIDTH}:h={strip_h}:qoffset={qoffset_high:.3f}")
        distortionmap_low.append(f"addroi=x=0:y={y_start}:w={WIDTH}:h={strip_h}:qoffset={qoffset_low:.3f}")

    DISTORTION_FILTER_HIGH = ",".join(distortionmap_high)
    DISTORTION_FILTER_LOW = ",".join(distortionmap_low)

    # Khởi tạo và lưu file ảnh Bản đồ độ méo (Distortion Map)
    plt.figure(figsize=(4, 2.5))
    heatmap_data = np.array(map_weights).reshape(-1, 1)
    plt.imshow(heatmap_data, aspect='auto', cmap='jet', interpolation='bilinear')
    plt.colorbar(label='Q-Offset')
    plt.title('Distortion Map (Xích đạo -> 2 Cực)', fontsize=10)
    plt.xticks([]) 
    plt.tight_layout()
    
    map_path = os.path.join(OUTPUT_FOLDER, 'distortion_map.png')
    plt.savefig(map_path, dpi=120)
    plt.close()

    # 3. Tiến hành Nén & Giải mã
    t0 = time.time()
    subprocess.run(f'ffmpeg -y -i "{input_path}" -vf "{DISTORTION_FILTER_HIGH}" -c:v libx265 -preset ultrafast -crf 20 -tune psnr -x265-params "log-level=error" "{high_encoded}"', shell=True, check=True)
    encode_time_high = time.time() - t0

    t0 = time.time()
    subprocess.run(f'ffmpeg -y -i "{input_path}" -vf "{DISTORTION_FILTER_LOW}" -c:v libx265 -preset ultrafast -crf 32 -tune psnr -x265-params "log-level=error" "{low_encoded}"', shell=True, check=True)
    encode_time_low = time.time() - t0

    subprocess.run(f'ffmpeg -y -i "{high_encoded}" -c:v libx265 -preset ultrafast "{high_decoded}" 2>/dev/null', shell=True, check=True)
    subprocess.run(f'ffmpeg -y -i "{low_encoded}" -c:v libx265 -preset ultrafast "{low_decoded}" 2>/dev/null', shell=True, check=True)

    # 4. Tính toán Metrics
    high_size_mb = os.path.getsize(high_encoded) / (1024 * 1024)
    low_size_mb = os.path.getsize(low_encoded) / (1024 * 1024)
    
    bitrate_high = (high_size_mb * 8) / duration
    bitrate_low = (low_size_mb * 8) / duration

    psnr_high = calculate_psnr(input_path, high_decoded)
    psnr_low = calculate_psnr(input_path, low_decoded)

    metrics = {
        "original_size": f"{original_size_mb:.2f} MB",
        "original_bitrate": f"{bitrate_original:.2f} Mbps",
        "high_size": f"{high_size_mb:.2f} MB",
        "high_bitrate": f"{bitrate_high:.2f} Mbps",
        "high_psnr": f"{psnr_high:.2f} dB",
        "high_time": f"{encode_time_high:.2f}s",
        "low_size": f"{low_size_mb:.2f} MB",
        "low_bitrate": f"{bitrate_low:.2f} Mbps",
        "low_psnr": f"{psnr_low:.2f} dB",
        "low_time": f"{encode_time_low:.2f}s"
    }

    return jsonify({
        "high_url": f"/{high_encoded}",
        "low_url": f"/{low_encoded}",
        "distortion_map_url": f"/{map_path}",
        "metrics": metrics
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)