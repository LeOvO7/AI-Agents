### The code originates from ipynb notebooks; this breakdown is for review purposes only.

import cv2
import ollama
import base64
import time

def extract_frames(video_path, seconds_interval=2):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * seconds_interval)
    
    frames_data = []
    frame_num = 0
    
    print(f"Extracting frames every {seconds_interval} seconds...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % interval == 0:
            timestamp = frame_num / fps
            # Convert frame to Base64 for Ollama
            _, buffer = cv2.imencode('.jpg', frame)
            img_b64 = base64.b64encode(buffer).decode('utf-8')
            frames_data.append({"time": timestamp, "image": img_b64})
        frame_num += 1
    
    cap.release()
    return frames_data

def run_analysis(frames):
    print("Starting LLaVA analysis...")
    results = []
    
    for entry in frames:
        ts = entry["time"]
        time_str = time.strftime('%M:%S', time.gmtime(ts))
        
        prompt = "Is there a person in this scene? Answer only YES or NO."
        
        try:
            response = ollama.chat(
                model='llava',
                messages=[{'role': 'user', 'content': prompt, 'images': [entry["image"]]}]
            )
            answer = response['message']['content'].strip().upper()
            is_present = "YES" in answer
            results.append({"time": ts, "time_str": time_str, "person": is_present})
            print(f"[{time_str}] Detection: {'Person present' if is_present else 'Empty'}")
        except Exception as e:
            print(f"Error at {time_str}: {e}")
            
    return results

def report_events(results):
    print("\n--- Surveillance Report ---")
    person_detected = False
    
    for res in results:
        # Simple state machine for enter/exit events
        if res["person"] and not person_detected:
            print(f"Event: Person ENTERED at {res['time_str']}")
            person_detected = True
        elif not res["person"] and person_detected:
            print(f"Event: Person EXITED at {res['time_str']}")
            person_detected = False
            
    if person_detected:
        print("Note: Person was still present at end of video.")

# Main Execution
video_file = "1.mp4"
extracted_frames = extract_frames(video_file)
analysis_results = run_analysis(extracted_frames)
report_events(analysis_results)