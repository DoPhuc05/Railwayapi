import requests
import os
from pymongo import MongoClient

# ✅ Kết nối MongoDB
MONGO_URI = "mongodb+srv://phuc201005:123@cluster0.ujsoo.mongodb.net/"
client = None
try:
    client = MongoClient(MONGO_URI)
    db = client["DACN2"]

    if "predictions" not in db.list_collection_names():
        db.create_collection("predictions")
        print("✅ Collection `predictions` đã được tạo!")
    else:
        print("✅ Đã kết nối MongoDB và có collection `predictions`")

except Exception as e:
    print(f"❌ Lỗi kết nối MongoDB: {e}")

# ✅ Upload ảnh lên ImgBB
IMGBB_API_KEY = "16df777ae50504dd7633bdf6e2474e6e"

def upload_to_imgbb(file_path):
    """Tải ảnh lên ImgBB và trả về URL"""
    try:
        with open(file_path, "rb") as file:
            response = requests.post(
                f"https://api.imgbb.com/1/upload?key={IMGBB_API_KEY}",
                files={"image": file}
            )
        if response.status_code == 200:
            return response.json()["data"]["url"]
        else:
            print(f"❌ Lỗi upload ImgBB: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Lỗi khi upload ImgBB: {e}")
        return None

# ✅ Upload video lên Streamable
STREAMABLE_USERNAME = "phuc201005@gmail.com"
STREAMABLE_PASSWORD = "dphc052010"

def upload_to_streamable(file_path):
    """Upload video lên Streamable và trả về URL"""
    try:
        with open(file_path, "rb") as video_file:
            response = requests.post(
                "https://api.streamable.com/upload",
                auth=(STREAMABLE_USERNAME, STREAMABLE_PASSWORD),
                files={"file": video_file}
            )
        if response.status_code == 200:
            shortcode = response.json().get("shortcode", "")
            return f"https://streamable.com/{shortcode}" if shortcode else None
        else:
            print(f"❌ Lỗi upload Streamable: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Lỗi khi upload Streamable: {e}")
        return None
