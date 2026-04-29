import os
import cv2
import numpy as np
import gradio as gr
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
import time
import subprocess
import onnxruntime as ort
import gc
import multiprocessing

ort.set_default_logger_severity(3)
cpu_count = str(multiprocessing.cpu_count())
os.environ["OMP_NUM_THREADS"] = cpu_count
os.environ["MKL_NUM_THREADS"] = cpu_count
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
os.environ["ORT_KEEP_MEMORY"] = "0"

print("Başlatılıyor...")

model_file = "inswapper_128.onnx"

haar_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

haar_profile = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_profileface.xml')

app = FaceAnalysis(
    name='buffalo_sc',
    providers=['CPUExecutionProvider'],
    allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=-1, det_size=(128, 128))

app_source = FaceAnalysis(
    name='buffalo_l',
    providers=['CPUExecutionProvider'],
    allowed_modules=['detection', 'recognition'])
app_source.prepare(ctx_id=-1, det_size=(320, 320))

swapper = get_model(model_file, download=False, download_zip=False)

class SwapCache:
    def __init__(self):
        self.last_swap_frame = None
        self.last_bbox = None
        self.hit = 0
        self.miss = 0

    def can_reuse(self, new_bbox, threshold=20):
        if self.last_swap_frame is None or self.last_bbox is None:
            return False
        old_cx = (self.last_bbox[0] + self.last_bbox[2]) / 2
        old_cy = (self.last_bbox[1] + self.last_bbox[3]) / 2
        new_cx = (new_bbox[0] + new_bbox[2]) / 2
        new_cy = (new_bbox[1] + new_bbox[3]) / 2
        dx = abs(new_cx - old_cx)
        dy = abs(new_cy - old_cy)
        old_w = self.last_bbox[2] - self.last_bbox[0]
        new_w = new_bbox[2] - new_bbox[0]
        size_diff = abs(old_w - new_w) / max(old_w, 1)
        return dx < threshold and dy < threshold and size_diff < 0.15

    def reuse(self, frame, new_bbox):
        old_cx = (self.last_bbox[0] + self.last_bbox[2]) / 2
        old_cy = (self.last_bbox[1] + self.last_bbox[3]) / 2
        new_cx = (new_bbox[0] + new_bbox[2]) / 2
        new_cy = (new_bbox[1] + new_bbox[3]) / 2
        dx = int(new_cx - old_cx)
        dy = int(new_cy - old_cy)
        h, w = frame.shape[:2]
        ob = self.last_bbox.astype(int)
        pad = 10
        ox1 = max(0, ob[0] - pad)
        oy1 = max(0, ob[1] - pad)
        ox2 = min(w, ob[2] + pad)
        oy2 = min(h, ob[3] + pad)
        nx1 = max(0, min(w - 1, ox1 + dx))
        ny1 = max(0, min(h - 1, oy1 + dy))
        nx2 = max(0, min(w, ox2 + dx))
        ny2 = max(0, min(h, oy2 + dy))
        rw = min(ox2 - ox1, nx2 - nx1)
        rh = min(oy2 - oy1, ny2 - ny1)
        
        if rw < 10 or rh < 10:
            return frame
            
        patch = self.last_swap_frame[oy1:oy1+rh, ox1:ox1+rw]
        mask = np.ones((rh, rw), dtype=np.float32)
        border = max(rh, rw) // 6
        if border > 2:
            mask[:border, :] *= np.linspace(0, 1, border)[:, None]
            mask[-border:, :] *= np.linspace(1, 0, border)[:, None]
            mask[:, :border] *= np.linspace(0, 1, border)[None, :]
            mask[:, -border:] *= np.linspace(1, 0, border)[None, :]
        mask_3 = mask[:, :, None]
        region = frame[ny1:ny1+rh, nx1:nx1+rw]
        
        if region.shape[:2] == patch.shape[:2]:
            frame[ny1:ny1+rh, nx1:nx1+rw] = (
                patch * mask_3 + region * (1 - mask_3)
            ).astype(np.uint8)
        self.hit += 1
        return frame

    def store(self, frame, bbox):
        self.last_swap_frame = frame.copy()
        self.last_bbox = bbox.copy()
        self.miss += 1


def haar_detect_all(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frontal = haar_cascade.detectMultiScale(
        gray, scaleFactor=1.15, minNeighbors=4, minSize=(40, 40)
    )
    profile_left = haar_profile.detectMultiScale(
        gray, scaleFactor=1.15, minNeighbors=6, minSize=(40, 40)
    )
    flipped = cv2.flip(gray, 1)
    profile_right_raw = haar_profile.detectMultiScale(
        flipped, scaleFactor=1.15, minNeighbors=6, minSize=(40, 40)
    )
    w = gray.shape[1]
    profile_right = []
    for (x, y, pw, ph) in profile_right_raw:
        profile_right.append((w - x - pw, y, pw, ph))
        
    all_faces = []
    if len(frontal) > 0:
        all_faces.extend(frontal.tolist())
    if len(profile_left) > 0:
        all_faces.extend(profile_left.tolist())
    if len(profile_right) > 0:
        all_faces.extend(profile_right)
        
    if len(all_faces) > 1:
        all_faces = merge_overlapping(all_faces)
        
    return all_faces

def merge_overlapping(faces, iou_threshold=0.3):
    if len(faces) <= 1:
        return faces
    boxes = [(x, y, x+w, y+h, w*h) for (x, y, w, h) in faces]
    boxes.sort(key=lambda b: b[4], reverse=True)
    keep = []
    used = set()
    for i, (x1, y1, x2, y2, area) in enumerate(boxes):
        if i in used:
            continue
        keep.append((x1, y1, x2-x1, y2-y1))
        for j in range(i+1, len(boxes)):
            if j in used:
                continue
            jx1, jy1, jx2, jy2, _ = boxes[j]
            ix1 = max(x1, jx1)
            iy1 = max(y1, jy1)
            ix2 = min(x2, jx2)
            iy2 = min(y2, jy2)
            if ix1 < ix2 and iy1 < iy2:
                inter = (ix2 - ix1) * (iy2 - iy1)
                union = area + boxes[j][4] - inter
                if inter / max(union, 1) > iou_threshold:
                    used.add(j)
    return keep

def haar_find_closest(haar_faces, last_center):
    if len(haar_faces) == 0:
        return None
    if last_center is None:
        return tuple(max(haar_faces, key=lambda f: f[2] * f[3]))
    best = None
    best_dist = float('inf')
    lx, ly = last_center
    for face in haar_faces:
        x, y, w, h = face[0], face[1], face[2], face[3]
        dist = (x + w/2 - lx)**2 + (y + h/2 - ly)**2
        if dist < best_dist:
            best_dist = dist
            best = (x, y, w, h)
    return best

def insight_crop_with_embedding(frame, haar_box, ref_emb_norm, fw, fh):
    hx, hy, hw, hh = haar_box
    pad = int(max(hw, hh) * 0.5)
    x1 = max(0, hx - pad)
    y1 = max(0, hy - pad)
    x2 = min(fw, hx + hw + pad)
    y2 = min(fh, hy + hh + pad)
    crop = frame[y1:y2, x1:x2]
    
    if crop.size == 0:
        return None, -1.0
    faces = app.get(crop)
    if not faces:
        return None, -1.0
        
    best_face = None
    best_sim = -1.0
    for face in faces:
        if face.det_score < 0.5:
            continue
        try:
            en = face.embedding / np.linalg.norm(face.embedding)
            sim = float(np.dot(ref_emb_norm, en))
            if sim > best_sim:
                face.bbox[0] += x1; face.bbox[1] += y1
                face.bbox[2] += x1; face.bbox[3] += y1
                if face.kps is not None:
                    face.kps[:, 0] += x1
                    face.kps[:, 1] += y1
                best_sim = sim
                best_face = face
        except:
            continue
    return best_face, best_sim

def insight_full_frame(frame, ref_emb_norm):
    faces = app.get(frame)
    if not faces:
        return None, -1.0
    best_face = None
    best_sim = -1.0
    for face in faces:
        if face.det_score < 0.5:
            continue
        try:
            en = face.embedding / np.linalg.norm(face.embedding)
            sim = float(np.dot(ref_emb_norm, en))
            if sim > best_sim:
                best_sim = sim
                best_face = face
        except:
            continue
    return best_face, best_sim

def find_best_target(frame, haar_faces, ref_emb_norm, fw, fh):
    best_face = None
    best_sim = -1.0
    for face_rect in haar_faces:
        hx, hy, hw, hh = face_rect[0], face_rect[1], face_rect[2], face_rect[3]
        pad = int(max(hw, hh) * 0.5)
        x1 = max(0, hx - pad)
        y1 = max(0, hy - pad)
        x2 = min(fw, hx + hw + pad)
        y2 = min(fh, hy + hh + pad)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        faces = app.get(crop)
        for face in faces:
            if face.det_score < 0.5:
                continue
            try:
                en = face.embedding / np.linalg.norm(face.embedding)
                sim = float(np.dot(ref_emb_norm, en))
                if sim > best_sim:
                    face.bbox[0] += x1; face.bbox[1] += y1
                    face.bbox[2] += x1; face.bbox[3] += y1
                    if face.kps is not None:
                        face.kps[:, 0] += x1
                        face.kps[:, 1] += y1
                    best_sim = sim
                    best_face = face
            except:
                continue
    return best_face, best_sim


def swap_faces_in_video(source_img_path, target_img_path, video_path, resolution="360p", progress=gr.Progress()):
    if not all([source_img_path, target_img_path, video_path]):
        return None, "❌ Tüm dosyaları yükleyin."
        
    source_img = cv2.imread(source_img_path)
    target_ref = cv2.imread(target_img_path)
    source_faces = app_source.get(source_img)
    
    if not source_faces:
        return None, "❌ Kaynak yüz bulunamadı."
    source_face = max(source_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    
    target_faces = app.get(target_ref)
    if not target_faces:
        return None, "❌ Hedef yüz bulunamadı."
    ref_emb_norm = target_faces[0].embedding / np.linalg.norm(target_faces[0].embedding)
    
    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 24
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    FRAME_SKIP = 3
    output_fps = orig_fps / FRAME_SKIP

    if resolution == "360p":
        scale = min(1.0, 360 / max(orig_h, 1))
    else:
        scale = 1.0

    new_w = int(orig_w * scale) // 2 * 2
    new_h = int(orig_h * scale) // 2 * 2
  
    audio_path = "temp_audio.aac"
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "copy", audio_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    
    temp_out = "temp_output.mp4"
    writer = cv2.VideoWriter(temp_out, cv2.VideoWriter_fourcc(*'mp4v'), output_fps, (new_w, new_h))
    
    ANCHOR_EVERY = 10
    SIM_THRESHOLD_HIGH = 0.38
    SIM_THRESHOLD_TRACK = 0.32
    CACHE_REUSE_PX = 20
    MAX_LOST_FRAMES = 3
    FULL_INSIGHT_EVERY = 5
    
    last_center = None
    tracking_active = False
    lost_count = 0
    cache = SwapCache()
    written = 0
    swap_count = 0
    reuse_count = 0
    skip_wrong = 0
    full_insight_count = 0
    start_time = time.time()
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % FRAME_SKIP != 0:
            continue
            
        if scale < 1.0:
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
        process_idx = written
        force_anchor = (
            (process_idx % ANCHOR_EVERY == 0) or
            (not tracking_active) or
            (lost_count >= MAX_LOST_FRAMES)
        )
        
        haar_faces = haar_detect_all(frame)
        swapped = False
        face = None
        sim = -1.0
        
        if force_anchor:
            if len(haar_faces) > 0:
                face, sim = find_best_target(frame, haar_faces, ref_emb_norm, new_w, new_h)
            if face is None or sim <= SIM_THRESHOLD_HIGH:
                face, sim = insight_full_frame(frame, ref_emb_norm)
                if face is not None and sim > SIM_THRESHOLD_HIGH:
                    full_insight_count += 1
            if face is not None and sim > SIM_THRESHOLD_HIGH:
                try:
                    frame = swapper.get(frame, face, source_face, paste_back=True)
                    swap_count += 1
                    swapped = True
                    cache.store(frame, face.bbox)
                    last_center = ((face.bbox[0]+face.bbox[2])/2, (face.bbox[1]+face.bbox[3])/2)
                    tracking_active = True
                    lost_count = 0
                except:
                    lost_count += 1
            else:
                lost_count += 1
                if lost_count > MAX_LOST_FRAMES * 3:
                    tracking_active = False
                    last_center = None
        elif tracking_active:
            if len(haar_faces) > 0:
                closest_haar = haar_find_closest(haar_faces, last_center)
            else:
                closest_haar = None
                
            if closest_haar is not None:
                temp_bbox = np.array([
                    closest_haar[0], closest_haar[1],
                    closest_haar[0] + closest_haar[2],
                    closest_haar[1] + closest_haar[3]
                ], dtype=np.float32)
                
                if cache.can_reuse(temp_bbox, CACHE_REUSE_PX):
                    frame = cache.reuse(frame, temp_bbox)
                    reuse_count += 1
                    swapped = True
                    last_center = (closest_haar[0] + closest_haar[2]/2, closest_haar[1] + closest_haar[3]/2)
                    lost_count = 0
                else:
                    face, sim = insight_crop_with_embedding(frame, closest_haar, ref_emb_norm, new_w, new_h)
                    if face is not None and sim > SIM_THRESHOLD_TRACK:
                        try:
                            frame = swapper.get(frame, face, source_face, paste_back=True)
                            swap_count += 1
                            swapped = True
                            cache.store(frame, face.bbox)
                            last_center = ((face.bbox[0]+face.bbox[2])/2, (face.bbox[1]+face.bbox[3])/2)
                            lost_count = 0
                        except:
                            lost_count += 1
                    elif face is not None and sim <= SIM_THRESHOLD_TRACK:
                        skip_wrong += 1
                        lost_count += 1
                    else:
                        lost_count += 1
            else:
                if lost_count % FULL_INSIGHT_EVERY == 0:
                    face, sim = insight_full_frame(frame, ref_emb_norm)
                    if face is not None and sim > SIM_THRESHOLD_TRACK:
                        try:
                            frame = swapper.get(frame, face, source_face, paste_back=True)
                            swap_count += 1
                            swapped = True
                            cache.store(frame, face.bbox)
                            last_center = ((face.bbox[0]+face.bbox[2])/2, (face.bbox[1]+face.bbox[3])/2)
                            lost_count = 0
                            full_insight_count += 1
                        except:
                            lost_count += 1
                    else:
                        lost_count += 1
                else:
                    lost_count += 1
                    
            if lost_count > MAX_LOST_FRAMES * 3:
                tracking_active = False
                
        writer.write(frame)
        written += 1
        
        if written % 100 == 0:
            gc.collect()
        if written % 10 == 0:
            pct = min(written / max(total_frames // FRAME_SKIP, 1), 0.99)
            elapsed = time.time() - start_time
            fps = written / max(elapsed, 0.001)
            progress(pct, desc=f"{written} kare | {fps:.1f} FPS")
            
    cap.release()
    writer.release()
    final_output = "final_result.mp4"
    
    if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
        subprocess.run([
            "ffmpeg", "-y", "-i", temp_out, "-i", audio_path,
            "-c:v", "copy", "-c:a", "aac", "-shortest", final_output
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        os.rename(temp_out, final_output)
        
    for f in [audio_path, temp_out]:
        if os.path.exists(f):
            os.remove(f)
            
    elapsed = int(time.time() - start_time)
    fps_avg = written / max(elapsed, 1)
    
    return final_output, f"✅ Toplam {written} kare işlendi | ⏱️ {elapsed} saniye sürdü"
css = """
body, .gradio-container { background: #0b0f19 !important; color: #e2e8f0 !important; }
footer { display: none !important; }
"""

with gr.Blocks(css=css, title="Face Swap") as demo:
    gr.Markdown("# Video Yüz Değiştirici")
    
    with gr.Row():
        source_img = gr.Image(type="filepath", label="📸 Kaynak Yüz")
        target_img = gr.Image(type="filepath", label="🎯 Hedef Kişi")
        
    with gr.Row():
        video_input = gr.Video(label="🎬 Video")
        
    with gr.Row():
        resolution = gr.Radio(
            choices=["360p", "Orijinal Boyut"], 
            value="360p", 
            label="⚙️ Çıktı Çözünürlüğü",
            info="360p seçerseniz video işlem hızını artırmak için küçültülür. Orijinal Boyut seçerseniz mevcut çözünürlük korunur."
        )

    btn = gr.Button("🚀 BAŞLAT", variant="primary")
    status = gr.Textbox(label="📊 Durum", interactive=False)
    video_output = gr.Video(label="🎥 Sonuç")
    
    btn.click(
        fn=swap_faces_in_video,
        inputs=[source_img, target_img, video_input, resolution],
        outputs=[video_output, status]
    )

demo.queue().launch()
