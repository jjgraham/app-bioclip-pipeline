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

# --- 1. Target Taxa Filter for Iron Horse Vineyards ---
PLAUSIBLE_TAXA = {
    Rank.FAMILY: [
        "Chrysopidae",      # Green Lacewings (Beneficial predator)
        "Syrphidae",        # Hoverflies / Syrphid Flies (Beneficial predator)
        "Pentatomidae",     # Stink Bugs (Vineyard Pest)
        "Vespidae",         # Yellowjackets and Paper Wasps
        "Miridae",          # Plant bugs
        "Cicadellidae",     # Leafhoppers (Pierce's Disease Vectors)
        "Pseudococcidae"    # Mealybugs (Leafroll Virus Vectors)
    ],
    Rank.ORDER: [
        "Artiodactyla",     # Black-tailed Deer / Feral Pigs (Canopy & ground damage)
        "Rodentia"          # Gophers / Squirrels (Root damage)
    ],
    Rank.CLASS: [
        "Aves"              # European Starlings / Finches (Fruit-destroying birds)
    ]
}

# --- 2. Initialize Models & Apply Constraints ---
print("Initializing YOLOv8 & SAM...")
yolo_model = YOLO("yolov8x.pt")
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth").to(device="cuda")
sam_predictor = SamPredictor(sam)

print("Loading BioCLIP with Tree-of-Life Subsetting...")
classifier = TreeOfLifeClassifier()

combined_filter = None
for rank, names in PLAUSIBLE_TAXA.items():
    for name in names:
        try:
            taxa_mask = classifier.create_taxa_filter(rank=rank, value=name)
            if combined_filter is None:
                combined_filter = taxa_mask
            else:
                combined_filter = np.logical_or(combined_filter, taxa_mask)
        except ValueError:
            pass

if combined_filter is not None:
    classifier.apply_filter(combined_filter)
    print("Taxonomic subsetting successfully locked to local vineyard ecosystem profile.")

# --- 3. Shared Inference Processing Core ---
def process_frame(img_bgr, detection_source="webhook"):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = img_bgr.shape
    
    # Focus computational power on the lower-center region if polling the lure
    if detection_source == "lure_polling":
        img_rgb = img_rgb[int(h*0.5):int(h*0.9), int(w*0.3):int(w*0.7)]
    
    yolo_results = yolo_model(img_rgb, verbose=False)
    
    for result in yolo_results:
        for box in result.boxes:
            xyxy = box.xyxy.cpu().numpy().astype(int)
            conf = float(box.conf.cpu().numpy())
            
            # Isolate via SAM
            sam_predictor.set_image(img_rgb)
            masks, _, _ = sam_predictor.predict(box=xyxy, multimask_output=False)
            
            isolated = img_rgb.copy()
            isolated[~masks] = 0
            x1, y1, x2, y2 = xyxy
            crop = isolated[y1:y2, x1:x2]
            
            temp_path = f"/tmp/{detection_source}_crop.jpg"
            cv2.imwrite(temp_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            
            # Context-constrained inference
            predictions = classifier.predict_image(temp_path)
            if predictions:
                taxon = predictions.get("species", predictions.get("family", "Unknown"))
                common = predictions.get("common_name", "Unknown")
                score = float(predictions.get("score", 0.0))
                
                # Send out to Sage Continuum architecture telemetry stream
                print(f"[{detection_source.upper()}] Identified: {common} ({taxon}) Conf: {score:.2f}")
                with Waggle() as waggle:
                    waggle.plugin.publish("vineyard.bioclip.detection", taxon, meta={
                        "source": detection_source,
                        "common_name": common,
                        "confidence": str(score),
                        "yolo_bbox": str(xyxy.tolist())
                    })

# --- 4. Webhook Target Listener (Macro Wildlife / Background) ---
@app.route("/webhook/smart-detection", methods=["POST"])
def handle_webhook():
    if "image" in request.files:
        file = request.files["image"]
        nparr = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        threading.Thread(target=process_frame, args=(img, "webhook")).start()
    return jsonify({"status": "queued"}), 200

# --- 5. Foreground Lure RTSP Poller (Insects) ---
def rtsp_lure_poller(rtsp_url):
    print(f"Starting foreground insect polling loop against stream: {rtsp_url}")
    while True:
        cap = cv2.VideoCapture(rtsp_url)
        ret, frame = cap.read()
        cap.release()
        if ret:
            try:
                process_frame(frame, detection_source="lure_polling")
            except Exception as e:
                print(f"Inference processing failure: {e}")
        time.sleep(3) # Execution interval cadence checking the lure

if __name__ == "__main__":
    RTSP_URL = os.getenv("CAMERA_RTSP_URL", "rtsp://192.168.1.50:554/s0")
    threading.Thread(target=rtsp_lure_poller, args=(RTSP_URL,), daemon=True).start()
    app.run(host="0.0.0.0", port=5000)
