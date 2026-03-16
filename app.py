from flask import Flask, request, jsonify, Response
import cv2
import psycopg2
import threading
import os
import tempfile
import threading
import base64
import math
import time
from ultralytics import YOLO

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_PATH = os.path.join(BASE_DIR, 'alert_audio.mp3')

# -----------------------------------
# SETTINGS
# -----------------------------------
MODEL_PATH = "yolov8n.pt"
MAX_WASTE_COUNT = 50
WEIGHT_PER_ITEM = 0.02
WASTE_CLASSES = ["bottle", "cup", "wine glass", "bowl"]
BOAT_SPEED = 0.00006

# -----------------------------------
# DATABASE
# -----------------------------------
db_lock = threading.Lock()

def init_database():
    conn = psycopg2.connect(
        host="db.kpcdzgvnlcjmfqcxxkwg.supabase.co",
        database="postgres",
        user="postgres",
        password="Gubendhiran@1000",
        port="5432"
    )

    cursor = conn.cursor()


    cursor.execute("""
    CREATE TABLE IF NOT EXISTS records(
        id SERIAL PRIMARY KEY,
        latitude DOUBLE PRECISION,
        longitude DOUBLE PRECISION,
        radius DOUBLE PRECISION,
        waste_count INTEGER,
        weight DOUBLE PRECISION
    )
    """)

    conn.commit()
    return conn, cursor


conn, cursor = init_database()
db_lock = threading.Lock()

def save_record(latitude, longitude, radius, waste_count, weight):
    with db_lock:
        print("Saving record:", latitude, longitude, waste_count)
        cursor.execute(
            """
            INSERT INTO records
            (latitude, longitude, radius, waste_count, weight)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (latitude, longitude, radius, waste_count, weight)
        )

        conn.commit()



# -----------------------------------
# LOAD MODEL
# -----------------------------------
model = None
try:
    model = YOLO(MODEL_PATH)
    print("YOLO model loaded successfully")
except Exception as e:
    print(f"Model loading error: {e}")

# -----------------------------------
# COVERAGE PATH PLANNER
# -----------------------------------
def generate_coverage_path(center_lat, center_lon, radius_m, lane_spacing_m=15):
    radius_deg = radius_m / 111320.0
    lane_spacing_deg = lane_spacing_m / 111320.0
    waypoints = []
    y = -radius_deg
    going_right = True
    while y <= radius_deg:
        half_chord = math.sqrt(max(0, radius_deg**2 - y**2))
        if half_chord < 0.00001:
            y += lane_spacing_deg
            going_right = not going_right
            continue
        if going_right:
            waypoints.append({"lat": center_lat + y, "lon": center_lon - half_chord})
            waypoints.append({"lat": center_lat + y, "lon": center_lon + half_chord})
        else:
            waypoints.append({"lat": center_lat + y, "lon": center_lon + half_chord})
            waypoints.append({"lat": center_lat + y, "lon": center_lon - half_chord})
        y += lane_spacing_deg
        going_right = not going_right
    return waypoints

# -----------------------------------
# BOAT STATE
# -----------------------------------
boat_state = {
    "active": False,
    "current_lat": 10.7905, "current_lon": 78.7047,
    "home_lat": 10.7905, "home_lon": 78.7047,
    "home_set": False,
    "target_lat": None, "target_lon": None,
    "status": "docked",
    "heading": 0, "trail": [],
    "arrival_notified": False,
    "coverage_path": [], "coverage_index": 0, "coverage_total": 0,
    "survey_center_lat": None, "survey_center_lon": None, "survey_radius": 0,
}
boat_lock = threading.Lock()

# -----------------------------------
# DETECTION STATE
# -----------------------------------
detection_state = {
    "detecting": False, "waste_count": 0,
    "latitude": None, "longitude": None, "radius": None,
    "cap": None, "last_frame": None, "status": "idle",
    "temp_video_path": None, "max_reached": False,
}
state_lock = threading.Lock()
detection_thread = None

# -----------------------------------
# BOAT SIMULATION THREAD
# -----------------------------------
def boat_simulation_loop():
    while True:
        time.sleep(0.08)
        with boat_lock:
            if not boat_state["active"]:
                continue
            status = boat_state["status"]

            if status == "coverage":
                idx = boat_state["coverage_index"]
                path = boat_state["coverage_path"]
                if idx >= len(path):
                    boat_state["status"] = "returning_home"
                    boat_state["target_lat"] = boat_state["home_lat"]
                    boat_state["target_lon"] = boat_state["home_lon"]
                    continue
                wp = path[idx]
                tlat, tlon = wp["lat"], wp["lon"]
                clat, clon = boat_state["current_lat"], boat_state["current_lon"]
                dlat, dlon = tlat - clat, tlon - clon
                dist = math.sqrt(dlat**2 + dlon**2)
                boat_state["heading"] = math.degrees(math.atan2(dlon, dlat)) % 360
                if dist < BOAT_SPEED * 1.5:
                    boat_state["current_lat"] = tlat
                    boat_state["current_lon"] = tlon
                    boat_state["coverage_index"] = idx + 1
                    boat_state["trail"].append({"lat": tlat, "lon": tlon})
                else:
                    nl = clat + (dlat / dist) * BOAT_SPEED
                    nn = clon + (dlon / dist) * BOAT_SPEED
                    boat_state["current_lat"] = nl
                    boat_state["current_lon"] = nn
                    trail = boat_state["trail"]
                    if not trail or math.sqrt((nl-trail[-1]["lat"])**2+(nn-trail[-1]["lon"])**2) > BOAT_SPEED*3:
                        trail.append({"lat": nl, "lon": nn})
                        if len(trail) > 500: trail.pop(0)

            elif status in ("navigating", "returning_home"):
                clat, clon = boat_state["current_lat"], boat_state["current_lon"]
                tlat, tlon = boat_state["target_lat"], boat_state["target_lon"]
                if tlat is None: continue
                dlat, dlon = tlat - clat, tlon - clon
                dist = math.sqrt(dlat**2 + dlon**2)
                boat_state["heading"] = math.degrees(math.atan2(dlon, dlat)) % 360
                if dist < BOAT_SPEED * 1.5:
                    boat_state["current_lat"] = tlat
                    boat_state["current_lon"] = tlon
                    boat_state["trail"].append({"lat": tlat, "lon": tlon})
                    if status == "navigating":
                        boat_state["status"] = "arrived"
                        boat_state["arrival_notified"] = False
                    else:
                        boat_state["status"] = "docked"
                        boat_state["active"] = False
                        boat_state["trail"] = []
                        boat_state["coverage_path"] = []
                        boat_state["coverage_index"] = 0
                else:
                    nl = clat + (dlat / dist) * BOAT_SPEED
                    nn = clon + (dlon / dist) * BOAT_SPEED
                    boat_state["current_lat"] = nl
                    boat_state["current_lon"] = nn
                    trail = boat_state["trail"]
                    if not trail or math.sqrt((nl-trail[-1]["lat"])**2+(nn-trail[-1]["lon"])**2) > BOAT_SPEED*4:
                        trail.append({"lat": nl, "lon": nn})
                        if len(trail) > 500: trail.pop(0)

boat_sim_thread = threading.Thread(target=boat_simulation_loop, daemon=True)
boat_sim_thread.start()

# -----------------------------------
# DETECTION THREAD
# -----------------------------------
def detection_loop():
    cap = detection_state["cap"]
    if cap is None or not cap.isOpened():
        with state_lock:
            detection_state["status"] = "error"
            detection_state["detecting"] = False
        return

    while True:
        with state_lock:
            if not detection_state["detecting"]: break

        ret, frame = cap.read()
        if not ret:
            with state_lock:
                detection_state["status"] = "ended"
                detection_state["detecting"] = False
            break

        detected_items = 0
        if model:
            results = model(frame, verbose=False)
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    cls_id = int(box.cls)
                    if model.names[cls_id].lower() in [x.lower() for x in WASTE_CLASSES]:
                        detected_items += 1
            annotated = results[0].plot()
        else:
            annotated = frame.copy()
            cv2.putText(annotated, "MODEL NOT LOADED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')

        with state_lock:
            detection_state["waste_count"] += detected_items
            detection_state["last_frame"] = frame_b64
            wc = detection_state["waste_count"]
            lat = detection_state["latitude"]
            lon = detection_state["longitude"]
            rad = detection_state["radius"]

        if wc >= MAX_WASTE_COUNT:
            weight = round(wc * WEIGHT_PER_ITEM, 2)
            if lat is not None:
                save_record(lat, lon, rad, wc, weight)
            
            with state_lock:
                detection_state["status"] = "max_reached"
                detection_state["detecting"] = False
                detection_state["max_reached"] = True
            with boat_lock:
                boat_state["target_lat"] = boat_state["home_lat"]
                boat_state["target_lon"] = boat_state["home_lon"]
                boat_state["status"] = "returning_home"
                boat_state["active"] = True
                boat_state["coverage_path"] = []
            break
        else:
            with state_lock:
                detection_state["status"] = "collecting"

    if cap: cap.release()
    with state_lock:
        detection_state["cap"] = None

# -----------------------------------
# ROUTES
# -----------------------------------
@app.route('/')
def index():
    html_path = os.path.join(BASE_DIR, 'index.html')
    with open(html_path, 'r', encoding='utf-8') as f:
        return Response(f.read(), mimetype='text/html')

@app.route('/api/set_home', methods=['POST'])
def set_home():
    data = request.json or {}
    lat = data.get('latitude')
    lon = data.get('longitude')
    if lat is None or lon is None:
        return jsonify({"error": "lat/lon required"}), 400
    with boat_lock:
        boat_state["home_lat"] = lat
        boat_state["home_lon"] = lon
        boat_state["current_lat"] = lat
        boat_state["current_lon"] = lon
        boat_state["home_set"] = True
    return jsonify({"ok": True})

@app.route('/api/navigate', methods=['POST'])
def navigate():
    data = request.json or {}
    lat = data.get('latitude')
    lon = data.get('longitude')
    radius = data.get('radius', 100)
    if lat is None or lon is None:
        return jsonify({"error": "latitude and longitude required"}), 400
    coverage = generate_coverage_path(lat, lon, radius)
    with boat_lock:
        home_lat = boat_state["home_lat"]
        home_lon = boat_state["home_lon"]
        boat_state["current_lat"] = home_lat
        boat_state["current_lon"] = home_lon
        boat_state["target_lat"] = lat
        boat_state["target_lon"] = lon
        boat_state["status"] = "navigating"
        boat_state["active"] = True
        boat_state["arrival_notified"] = False
        boat_state["trail"] = [{"lat": home_lat, "lon": home_lon}]
        boat_state["coverage_path"] = coverage
        boat_state["coverage_index"] = 0
        boat_state["coverage_total"] = len(coverage)
        boat_state["survey_center_lat"] = lat
        boat_state["survey_center_lon"] = lon
        boat_state["survey_radius"] = radius
    with state_lock:
        detection_state["latitude"] = lat
        detection_state["longitude"] = lon
        detection_state["radius"] = radius
        detection_state["max_reached"] = False
    return jsonify({"ok": True, "home_lat": home_lat, "home_lon": home_lon, "coverage_waypoints": len(coverage)})

@app.route('/api/boat_state')
def get_boat_state():
    with boat_lock:
        idx = boat_state["coverage_index"]
        total = boat_state["coverage_total"]
        pct = round((idx / total * 100) if total > 0 else 0, 1)
        return jsonify({
            "active": boat_state["active"],
            "current_lat": boat_state["current_lat"],
            "current_lon": boat_state["current_lon"],
            "home_lat": boat_state["home_lat"],
            "home_lon": boat_state["home_lon"],
            "home_set": boat_state["home_set"],
            "target_lat": boat_state["target_lat"],
            "target_lon": boat_state["target_lon"],
            "status": boat_state["status"],
            "heading": boat_state["heading"],
            "trail": boat_state["trail"][-100:],
            "arrival_notified": boat_state["arrival_notified"],
            "coverage_index": idx, "coverage_total": total,
            "coverage_pct": pct,
            "survey_center_lat": boat_state["survey_center_lat"],
            "survey_center_lon": boat_state["survey_center_lon"],
            "survey_radius": boat_state["survey_radius"],
            "coverage_path": boat_state["coverage_path"],
        })

@app.route('/api/mark_arrival_notified', methods=['POST'])
def mark_arrival_notified():
    with boat_lock:
        boat_state["arrival_notified"] = True
    return jsonify({"ok": True})

@app.route('/api/start_coverage', methods=['POST'])
def start_coverage():
    with boat_lock:
        if boat_state["status"] not in ("arrived", "collecting"):
            return jsonify({"error": "Boat must be at destination first"}), 400
        boat_state["status"] = "coverage"
        boat_state["active"] = True
        boat_state["coverage_index"] = 0
    return jsonify({"ok": True})

@app.route('/api/return_home', methods=['POST'])
def return_home():
    with boat_lock:
        boat_state["target_lat"] = boat_state["home_lat"]
        boat_state["target_lon"] = boat_state["home_lon"]
        boat_state["status"] = "returning_home"
        boat_state["active"] = True
    return jsonify({"ok": True, "latitude": boat_state["home_lat"], "longitude": boat_state["home_lon"]})

@app.route('/api/start_detection', methods=['POST'])
def start_detection():
    global detection_thread
    data = request.json or {}
    camera_option = data.get('camera_option', 'webcam')
    camera_url = data.get('camera_url', '')
    with state_lock:
        if detection_state["detecting"]:
            return jsonify({"error": "Already detecting"}), 400
        detection_state["waste_count"] = 0
        detection_state["status"] = "starting"
        detection_state["last_frame"] = None
        detection_state["max_reached"] = False
        if camera_option == "webcam":
            cap = cv2.VideoCapture(0)
        elif camera_option == "ip":
            cap = cv2.VideoCapture(camera_url)
        else:
            return jsonify({"error": "Use /api/upload_video for video files"}), 400
        if not cap.isOpened():
            return jsonify({"error": "Could not open camera."}), 500
        detection_state["cap"] = cap
        detection_state["detecting"] = True
    with boat_lock:
        if boat_state["status"] not in ("coverage",):
            boat_state["status"] = "collecting"
    detection_thread = threading.Thread(target=detection_loop, daemon=True)
    detection_thread.start()
    return jsonify({"ok": True})

@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    global detection_thread
    if 'video' not in request.files:
        return jsonify({"error": "No video file"}), 400
    video_file = request.files['video']
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    video_file.save(tfile.name)
    with state_lock:
        detection_state["waste_count"] = 0
        detection_state["status"] = "starting"
        detection_state["last_frame"] = None
        detection_state["max_reached"] = False
        detection_state["temp_video_path"] = tfile.name
        cap = cv2.VideoCapture(tfile.name)
        if not cap.isOpened():
            return jsonify({"error": "Could not open video"}), 500
        detection_state["cap"] = cap
        detection_state["detecting"] = True
    with boat_lock:
        boat_state["status"] = "collecting"
    detection_thread = threading.Thread(target=detection_loop, daemon=True)
    detection_thread.start()
    return jsonify({"ok": True})

@app.route('/api/stop_detection', methods=['POST'])
def stop_detection():
    with state_lock:
        detection_state["detecting"] = False
        detection_state["status"] = "stopped"
    with boat_lock:
        if boat_state["status"] in ("collecting", "coverage"):
            boat_state["status"] = "arrived"
    return jsonify({"ok": True})

@app.route('/api/set_location', methods=['POST'])
def set_location():
    data = request.json or {}
    with state_lock:
        detection_state["latitude"] = data.get("latitude")
        detection_state["longitude"] = data.get("longitude")
        detection_state["radius"] = data.get("radius", 0)
    return jsonify({"ok": True})

@app.route('/api/state')
def get_state():
    with state_lock:
        return jsonify({
            "detecting": detection_state["detecting"],
            "waste_count": detection_state["waste_count"],
            "weight": round(detection_state["waste_count"] * WEIGHT_PER_ITEM, 2),
            "status": detection_state["status"],
            "latitude": detection_state["latitude"],
            "longitude": detection_state["longitude"],
            "radius": detection_state["radius"],
            "frame": detection_state["last_frame"],
            "max_waste": MAX_WASTE_COUNT,
            "max_reached": detection_state["max_reached"],
        })

@app.route('/api/records')
def get_records():
    with db_lock:
        cursor.execute("SELECT * FROM records ORDER BY id DESC")
        rows = cursor.fetchall()
    return jsonify([{"id": r[0], "latitude": r[1], "longitude": r[2],
                     "radius": r[3], "waste_count": r[4], "weight": r[5]} for r in rows])

# -----------------------------------
# AUDIO SETTINGS
# -----------------------------------
@app.route('/api/upload_audio', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file"}), 400
    audio_file = request.files['audio']
    ext = os.path.splitext(audio_file.filename)[1].lower()
    if ext not in ('.mp3', '.wav', '.ogg', '.m4a'):
        return jsonify({"error": "Unsupported format. Use MP3, WAV, OGG, M4A"}), 400
    save_path = os.path.join(BASE_DIR, 'alert_audio' + ext)
    # Remove old files
    for old_ext in ('.mp3', '.wav', '.ogg', '.m4a'):
        old_path = os.path.join(BASE_DIR, 'alert_audio' + old_ext)
        if os.path.exists(old_path):
            os.remove(old_path)
    audio_file.save(save_path)
    return jsonify({"ok": True, "filename": audio_file.filename, "path": '/api/audio'})

@app.route('/api/audio')
def serve_audio():
    for ext in ('.mp3', '.wav', '.ogg', '.m4a'):
        path = os.path.join(BASE_DIR, 'alert_audio' + ext)
        if os.path.exists(path):
            mime = {'mp3': 'audio/mpeg', 'wav': 'audio/wav', 'ogg': 'audio/ogg', 'm4a': 'audio/mp4'}
            with open(path, 'rb') as f:
                return Response(f.read(), mimetype=mime.get(ext[1:], 'audio/mpeg'))
    return jsonify({"error": "No audio file uploaded"}), 404

@app.route('/api/audio_status')
def audio_status():
    for ext in ('.mp3', '.wav', '.ogg', '.m4a'):
        path = os.path.join(BASE_DIR, 'alert_audio' + ext)
        if os.path.exists(path):
            return jsonify({"has_audio": True, "filename": 'alert_audio' + ext})
    return jsonify({"has_audio": False})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, threaded=True)

