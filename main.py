import os
import cv2
import numpy as np
import aiofiles
from fastapi import FastAPI, UploadFile, File
import uvicorn
import threading
from pyngrok import ngrok
from ultralytics import YOLO
from app.dtbase import db, upload_to_imgbb, upload_to_streamable # Thay đổi sang ImgBB
from collections import deque  # 🔥 Lưu lịch sử số lượng swimmer
from datetime import datetime
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request

templates = Jinja2Templates(directory="templates")


# ✅ Khởi tạo FastAPI
app = FastAPI()

# ✅ Cấu hình Ngrok
NGROK_AUTH_TOKEN = "2tcouva4KHG2fccLtZPW7PDXMvZ_4YCgrCFDUKea2cJUhYj8t"
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# ✅ Load mô hình YOLOv8
MODEL_PATH = r"D:\API\best (3).pt"  # Sử dụng raw string để tránh lỗi ký tự đặc biệt


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Không tìm thấy file {MODEL_PATH}")

print(f"🔄 Đang tải mô hình YOLOv8 từ {MODEL_PATH}...")
model = YOLO(MODEL_PATH)
print("✅ Mô hình YOLOv8 đã sẵn sàng!")

@app.post("/predict-image/")
async def predict_image(file: UploadFile = File(...)):
    """Nhận ảnh, chạy YOLOv8, lưu MongoDB & ImgBB"""
    try:
        # ✅ Đọc ảnh từ tệp tải lên
        image_bytes = await file.read()
        image_np = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        
        if image is None:
            return {"error": "Không thể giải mã hình ảnh."}
        
        print("Image Type:", type(image))
        print("Image Shape:", image.shape)
        
        # ✅ Chạy YOLO trên ảnh
        results = model(image)
        predictions = []
        person_count = 0
        
        for result in results:
            for i in range(len(result.boxes)):
                x1, y1, x2, y2 = map(int, result.boxes.xyxy[i].tolist())
                score = round(result.boxes.conf[i].item(), 2)
                label = model.names[int(result.boxes.cls[i].item())]
                print("Detected Label:", label)  # Debug
                
                if label.lower() == "swimmer":  # Chắc chắn là lowercase
                    person_count += 1
                
                predictions.append({
                    "label": label,
                    "confidence": score,
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                })
        
        # ✅ Lưu ảnh có bounding box
        image_with_boxes = results[0].plot()
        output_path = "output.jpg"
        cv2.imwrite(output_path, image_with_boxes)
        
        # ✅ Upload ảnh lên ImgBB
        imgbb_url = upload_to_imgbb(output_path)

        # ✅ Xóa file ảnh đã lưu tạm
        if os.path.exists(output_path):
            os.remove(output_path)
        
        # ✅ Lưu vào MongoDB
        db.predictions.insert_one({
            "image_url": imgbb_url,
            "predictions": predictions,
            "person_count": person_count
        })
        
        return {"person_count": person_count, "image_url": imgbb_url}
    
    except Exception as e:
        print("Lỗi xử lý ảnh:", e)
        return {"error": str(e)}
    

# ✅ XỬ LÝ VIDEO & LƯU VÀO MONGODB + STREAMABLE
@app.post("/predict-video/")
async def predict_video(file: UploadFile = File(...)):
    """Nhận video, xử lý bằng YOLOv8, lưu MongoDB & Streamable"""
    input_video_path = "temp_input.mp4"
    async with aiofiles.open(input_video_path, "wb") as f:
        await f.write(await file.read())

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        return {"error": "Không thể mở video!"}

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video_path = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    total_swimmer_count = 0
    prev_swimmer_count = 0  # Biến theo dõi số swimmer nhóm trước
    prev_counts = []  # Lưu số swimmer của nhóm 3 frame gần nhất

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Hết video

        frame_count += 1

        # ✅ Chỉ xử lý mỗi 5 frame để tối ưu
        if frame_count % 5 == 0:
            results = model(frame)

            if results and len(results) > 0:
                frame = results[0].plot()  # Vẽ bounding box lên frame

                # ✅ Đếm số swimmer trong frame hiện tại
                current_swimmer_count = sum(
                    1 for box in results[0].boxes if model.names[int(box.cls.item())] == "swimmer"
                )
                prev_counts.append(current_swimmer_count)

                # ✅ Khi đủ 3 frame, cập nhật số swimmer nếu có thay đổi
                if len(prev_counts) == 3:
                    avg_swimmer_count = round(sum(prev_counts) / 3)  # Lấy trung bình nhóm 3 frame

                    if avg_swimmer_count != prev_swimmer_count:
                        total_swimmer_count = avg_swimmer_count  # Cập nhật số swimmer
                        prev_swimmer_count = avg_swimmer_count  # Lưu lại để so sánh với nhóm tiếp theo

                    prev_counts = []  # ✅ Reset nhóm 3 frame để tiếp tục theo dõi

        out.write(frame)  # ✅ Ghi frame vào video đầu ra

    # ✅ Giải phóng tài nguyên
    cap.release()
    out.release()

    # ✅ Upload video lên Streamable
    streamable_url = upload_to_streamable(output_video_path)

    # ✅ Xóa các file tạm
    for file_path in [input_video_path, output_video_path]:
        if os.path.exists(file_path):
            os.remove(file_path)

    # ✅ Lưu vào MongoDB
    db.predictions.insert_one({
        "video_url": streamable_url,
        "total_swimmer_count": total_swimmer_count  # 🔥 Tổng số swimmer sau khi xử lý
    })

    return {
        "total_swimmer_count": total_swimmer_count,
        "video_url": streamable_url
    }
recording = False
camera_thread = None

@app.get("/start-camera/")
def start_camera():
    global recording, camera_thread
    if recording:
        return {"message": "Camera đang chạy!"}

    recording = True

    def camera_worker():
        global recording
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Không thể mở webcam.")
            return

        # 🔄 Tạo tên file output có timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"camera_output_{timestamp}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_filename, fourcc, 20.0, (640, 480))
        total_swimmer_count = 0

        while recording:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            frame = results[0].plot()

            swimmer_count = sum(
                1 for box in results[0].boxes
                if model.names[int(box.cls.item())].lower() == "swimmer"
            )
            total_swimmer_count = swimmer_count

            out.write(frame)
            cv2.imshow("🟢 Realtime Detection (Nhấn 'q' để dừng)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                recording = False
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # ✅ Upload video lên Streamable
        streamable_url = upload_to_streamable(output_filename)

        # ✅ Xóa file camera output
        if os.path.exists("camera_output.mp4"):
            os.remove("camera_output.mp4")

        # ✅ Ghi vào MongoDB
        db.predictions.insert_one({
            "video_url": streamable_url,
            "total_swimmer_count": total_swimmer_count,
            "source": "realtime",
            "timestamp": timestamp
        })

        print(f"✅ Video {output_filename} đã được upload!")

    camera_thread = threading.Thread(target=camera_worker)
    camera_thread.start()
    return {"message": "✅ Camera đang ghi hình và phân tích realtime."}

@app.get("/stop-camera/")
def stop_camera():
    global recording
    recording = False
    return {"message": "⏹ Camera đã được tắt và video đã được xử lý."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
