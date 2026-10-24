import os
import cv2
import time
import threading
import numpy as np
from flask import Flask, request, jsonify
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from bioclip.predict import TreeOfLifeClassifier
from bioclip import Rank
from pywaggle import Waggle

app = Flask(__name__)

# --- Model Initialization ---
yolo_model = YOLO("yolov8x.pt")
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth").to(device="cuda")
sam_predictor = SamPredictor(sam)
classifier = TreeOfLifeClassifier()

# Apply the Plausible Taxa filter (using the dictionary from Section 2)
combined_filter = None
for rank, names in PLAUSIBLE_TAXA.items():
    for name in names:
        try:
            taxa_mask = classifier.create_taxa_filter(rank=rank, value=name)
            combined_filter = taxa_mask if combined_filter is None else np.logical_or(combined_filter, taxa_mask)
        except ValueError:
            pass
if combined_filter is not None:
    classifier.apply_filter(combined_filter)

# --- Shared Core Logic for Inference ---
def process_frame(img_bgr, detection_source="webhook"):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = img_bgr.shape
    
    # If polling the lure, crop specifically to the foreground area to save compute
    if detection_source == "lure_polling":
        # Example: Focus on the center-bottom 30% of the screen where the lure hangs
        img_rgb = img_rgb[int(h*0.5):int(h*0.9), int(w*0.3):int(w*0.7)]
    
    yolo_results = yolo_model(img_rgb, verbose=False)
    
    for result in yolo_results:
        for box in result.boxes:
            xyxy = box.xyxy.cpu().numpy().astype(int)[0]
            conf = float(box.conf.cpu().numpy()[0])
            
            # Segment and Isolate via SAM
            sam_predictor.set_image(img_rgb)
            masks, _, _ = sam_predictor.predict(box=xyxy, multimask_output=False)
            
            isolated = img_rgb.copy()
            isolated[~masks[0]] = 0
            x1, y1, x2, y2 = xyxy
            crop = isolated[y1:y2, x1:x2]
            
            temp_path = f"/tmp/{detection_source}_crop.jpg"
            cv2.imwrite(temp_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            
            # Subsetting-limited BioCLIP classification
            predictions = classifier.predict_image(temp_path)
            if predictions:
                taxon = predictions.get("species", predictions.get("family", "Unknown"))
                score = float(predictions.get("score", 0.0))
                
                # Publish to Sage Continuum telemetry stream
                with Waggle() as waggle:
                    waggle.plugin.publish("vineyard.bioclip.detection", taxon, meta={
                        "source": detection_source,
                        "common_name": predictions.get("common_name", "Unknown"),
                        "confidence": str(score)
                    })

# --- Stream A: Native Webhook Listener (Background Macro Wildlife) ---
@app.route("/webhook/smart-detection", methods=["POST"])
def handle_webhook():
    if "image" in request.files:
        file = request.files["image"]
        nparr = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        threading.Thread(target=process_frame, args=(img, "webhook")).start()
    return jsonify({"status": "queued"}), 200

# --- Stream B: Foreground Lure Polking Thread (Macro Insects) ---
def rtsp_lure_poller(rtsp_url):
    while True:
        cap = cv2.VideoCapture(rtsp_url)
        ret, frame = cap.read()
        cap.release()
        if ret:
            try:
                process_frame(frame, detection_source="lure_polling")
            except Exception as e:
                print(f"Error processing lure frame: {e}")
        time.sleep(3) # Poll the wintergreen trap every 3 seconds

if __name__ == "__main__":
    RTSP_URL = os.getenv("CAMERA_RTSP_URL", "rtsp://192.168.1.X:554/live/stream_ch1")
    threading.Thread(target=rtsp_lure_poller, args=(RTSP_URL,), daemon=True).start()
    app.run(host="0.0.0.0", port=5000)
