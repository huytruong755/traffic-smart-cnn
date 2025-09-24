# gui.py
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from io import BytesIO
from keras.models import load_model
import time
import os
import random

MODEL_PATH = r"D:\SmartTrafficByCNN\my_model.h5"  
DEMO_FOLDER = r"D:\SmartTrafficByCNN\Meta"  

@st.cache_resource(show_spinner=False)
def load_trained_model(path):
    try:
        m = load_model(path)
        return m
    except Exception as e:
        st.error(f"Không thể load model: {e}")
        return None

model = load_trained_model(MODEL_PATH)

classes = {
    1: 'Giới hạn tốc độ (20 km/h)',
    2: 'Giới hạn tốc độ (30 km/h)',
    3: 'Giới hạn tốc độ (50 km/h)',
    4: 'Giới hạn tốc độ (60 km/h)',
    5: 'Giới hạn tốc độ (70 km/h)',
    6: 'Giới hạn tốc độ (80 km/h)',
    7: 'Hết giới hạn tốc độ (80 km/h)',
    8: 'Giới hạn tốc độ (100 km/h)',
    9: 'Giới hạn tốc độ (120 km/h)',
    10: 'Cấm vượt',
    11: 'Cấm xe trên 3.5 tấn vượt',
    12: 'Được ưu tiên tại ngã tư',
    13: 'Đường ưu tiên',
    14: 'Nhường đường',
    15: 'Dừng lại (STOP)',
    16: 'Cấm mọi phương tiện',
    17: 'Cấm xe trên 3.5 tấn',
    18: 'Không xác định',
    19: 'Chú ý nguy hiểm',
    20: 'Đường cong nguy hiểm bên trái',
    21: 'Đường cong nguy hiểm bên phải',
    22: 'Đường cong liên tiếp',
    23: 'Đường xóc',
    24: 'Đường trơn trượt',
    25: 'Đường hẹp bên phải',
    26: 'Đang thi công',
    27: 'Có đèn giao thông',
    28: 'Có người đi bộ',
    29: 'Trẻ em băng qua đường',
    30: 'Xe đạp băng qua đường',
    31: 'Cẩn thận băng/tuyết',
    32: 'Động vật hoang dã băng qua',
    33: 'Hết giới hạn tốc độ và cấm vượt',
    34: 'Rẽ phải phía trước',
    35: 'Rẽ trái phía trước',
    36: 'Chỉ được đi thẳng',
    37: 'Đi thẳng hoặc rẽ phải',
    38: 'Đi thẳng hoặc rẽ trái',
    39: 'Đi về bên phải',
    40: 'Đi về bên trái',
    41: 'Bắt buộc đi theo vòng xuyến',
    42: 'Hết cấm vượt',
    43: 'Hết cấm vượt xe trên 3.5 tấn'
}

descriptions = {
    1: 'Khi gặp biển báo này phải di chuyển với vận tốc không vượt quá 20 km/h.',
    2: 'Khi gặp biển báo này phải di chuyển với vận tốc không vượt quá 30 km/h.', 
    3: 'Khi gặp biển báo này phải di chuyển với vận tốc không vượt quá 50 km/h.', 
    4: 'Khi gặp biển báo này phải di chuyển với vận tốc không vượt quá 60 km/h.', 
    5: 'Khi gặp biển báo này phải di chuyển với vận tốc không vượt quá 70 km/h.', 
    6: 'Khi gặp biển báo này phải di chuyển với vận tốc không vượt quá 80 km/h.', 
    7: 'Khi gặp biển báo này hết hiệu lực giới hạn tốc độ 80 km/h.', 
    8: 'Khi gặp biển báo này phải di chuyển với vận tốc không vượt quá 100 km/h.', 
    9: 'Khi gặp biển báo này phải di chuyển với vận tốc không vượt quá 120 km/h.', 
    10: 'Khi gặp biển báo này không được phép vượt.', 
    11: 'Khi gặp biển báo này xe trọng tải trên 3.5 tấn cấm vượt vượt.', 
    12: 'Biển báo cho biết bạn có quyền ưu tiên đi trước các xe khác tại ngã tư hoặc nơi giao nhau.', 
    13: 'Biển báo chỉ dẫn đoạn đường bạn đang đi là đường ưu tiên so với các đường khác cắt ngang.', 
    14: 'Biển yêu cầu người lái xe phải nhường đường cho phương tiện khác đang lưu thông.', 
    15: 'Biển báo hiệu lệnh dừng lại hoàn toàn trước khi tiếp tục di chuyển, thường tại ngã tư.', 
    16: 'Biển cấm tất cả các phương tiện (cơ giới và thô sơ) đi vào đoạn đường này.', 
    17: 'Biển cấm các phương tiện có trọng tải lớn hơn 3.5 tấn đi vào khu vực này.', 
    18: 'Biển không xác định được loại – có thể do ảnh không rõ hoặc không đúng biển giao thông.', 
    19: 'Biển cảnh báo khu vực phía trước có tình huống nguy hiểm, người lái xe cần thận trọng.',
    20: 'Biển cảnh báo phía trước có đoạn đường cong nguy hiểm sang bên trái.', 
    21: 'Biển cảnh báo phía trước có đoạn đường cong nguy hiểm sang bên phải.', 
    22: 'Biển cảnh báo sắp tới là đoạn đường có nhiều khúc cua liên tiếp.', 
    23: 'Biển cảnh báo đoạn đường gồ ghề, có ổ gà hoặc mặt đường không bằng phẳng.', 
    24: 'Biển cảnh báo đoạn đường trơn trượt, cần giảm tốc độ và giữ khoảng cách an toàn.', 
    25: 'Biển cảnh báo phần đường phía bên phải bị thu hẹp lại – cần chú ý khi đi qua.', 
    26: 'Biển cảnh báo đoạn đường phía trước đang thi công, có thể gây cản trở giao thông.', 
    27: 'Biển cảnh báo có đèn tín hiệu giao thông phía trước, cần chú ý quan sát.', 
    28: 'Biển cảnh báo khu vực phía trước là đường dành cho người đi bộ sang đường.', 
    29: 'Biển cảnh báo gần khu vực trường học, trẻ em có thể băng qua đường bất ngờ.', 
    30: 'Biển cảnh báo khu vực giao cắt với đường dành cho xe đạp.', 
    31: 'Biển cảnh báo đoạn đường có thể bị đóng băng hoặc có tuyết – dễ trơn trượt.', 
    32: 'Biển cảnh báo đoạn đường có động vật hoang dã băng qua – cần giảm tốc độ.', 
    33: 'Biển báo kết thúc hiệu lực của giới hạn tốc độ và lệnh cấm vượt trước đó.', 
    34: 'Biển hiệu lệnh bắt buộc rẽ phải ở phía trước.', 
    35: 'Biển hiệu lệnh bắt buộc rẽ trái ở phía trước.', 
    36: 'Biển hiệu lệnh bắt buộc chỉ được đi thẳng.', 
    37: 'Biển hiệu lệnh chỉ được đi thẳng hoặc rẽ phải.', 
    38: 'Biển hiệu lệnh chỉ được đi thẳng hoặc rẽ trái.', 
    39: 'Biển hiệu lệnh chỉ được đi về phía bên phải.', 
    40: 'Biển hiệu lệnh chỉ được đi về phía bên trái.', 
    41: 'Biển hiệu lệnh bắt buộc đi theo hướng vòng xuyến phía trước.', 
    42: 'Biển báo hiệu lệnh cấm vượt đã hết hiệu lực – được phép vượt lại.',
    43: 'Biển báo hiệu lệnh cấm xe trên 3.5 tấn vượt đã hết hiệu lực.'
}

def preprocess_pil_image(pil_img, target_size=(30, 30)):
    img = pil_img.convert("RGB").resize(target_size)
    arr = np.array(img).astype("float32")
    # nếu cần chuẩn hóa giống khi train (vd: /255), uncomment:
    # arr = arr / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_image(pil_img, threshold=0.6):
    """
    Trả về: (label, description, confidence)
    Nếu confidence < threshold -> label="Không phải biển báo"
    Nếu có lỗi -> label="Lỗi nhận dạng", description = lỗi
    """
    if model is None:
        return "Lỗi nhận dạng", "Model chưa được load.", 0.0
    try:
        arr = preprocess_pil_image(pil_img)
        preds = model.predict(arr)[0]
        pred_index = int(np.argmax(preds))
        confidence = float(preds[pred_index])
        if confidence < threshold:
            return "Không phải biển báo", "Độ tin cậy thấp (< threshold).", confidence
        label = classes.get(pred_index + 1, "Không xác định")
        desc = descriptions.get(pred_index + 1, "Không có mô tả.")
        return label, desc, confidence
    except Exception as e:
        return "Lỗi nhận dạng", str(e), 0.0

# --------- Streamlit UI ---------
st.set_page_config(page_title="AI Nhận diện biển báo giao thông", page_icon="🚦", layout="wide")
st.title("🚦 AI Nhận diện biển báo giao thông")

# Sidebar: cấu hình
st.sidebar.header("Cấu hình")
threshold = st.sidebar.slider("Ngưỡng confidence (để xem là biển báo)", min_value=0.0, max_value=1.0, value=0.6, step=0.01)
use_demo = st.sidebar.checkbox("Chạy demo với ảnh mẫu", value=False)
input_mode = st.sidebar.radio("Chọn nguồn ảnh / camera:", ("Upload ảnh", "Chụp 1 ảnh (camera trình duyệt)", "Realtime (video liên tục)"))

st.sidebar.markdown("---")
st.sidebar.write("Model path:")
st.sidebar.write(MODEL_PATH)

# Main layout
col1, col2 = st.columns([1, 1])

# Left column: hiển thị input
with col1:
    st.subheader("Ảnh đầu vào")
    input_image = None
    # Demo
    if use_demo:
        if os.path.exists(DEMO_FOLDER) and os.path.isdir(DEMO_FOLDER):
            demo_files = [f for f in os.listdir(DEMO_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            if demo_files:
                chosen_file = random.choice(demo_files)  # chọn ngẫu nhiên 1 file
                chosen_path = os.path.join(DEMO_FOLDER, chosen_file)
                try:
                    input_image = Image.open(chosen_path).convert("RGB")
                    st.image(input_image, caption=f"Ảnh demo: {chosen_file}", use_column_width=True)
                except Exception as e:
                    st.error(f"Không mở được ảnh demo: {e}")
                    input_image = None
            else:
                st.warning(f"Không tìm thấy ảnh (.jpg/.png) trong thư mục {DEMO_FOLDER}.")
        else:
            st.warning(f"Thư mục demo không tồn tại: {DEMO_FOLDER}")
            st.warning(f"Thư mục demo không tồn tại: {DEMO_FOLDER}")
    # Tùy chọn upload
    if input_mode == "Upload ảnh":
        uploaded_file = st.file_uploader("Tải ảnh biển báo lên (jpg, png, jpeg)", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            try:
                input_image = Image.open(uploaded_file).convert("RGB")
                st.image(input_image, caption="Ảnh tải lên", use_column_width=True)
            except Exception as e:
                st.error(f"Lỗi khi mở ảnh: {e}")
                input_image = None

    # Chụp 1 ảnh bằng camera trình duyệt (st.camera_input)
    elif input_mode == "Chụp 1 ảnh (camera trình duyệt)":
        st.info("Dùng webcam trình duyệt. Sau khi chụp, ảnh sẽ được gửi lên server để dự đoán.")
        camera_file = st.camera_input("Chụp ảnh từ webcam")
        if camera_file is not None:
            try:
                input_image = Image.open(camera_file).convert("RGB")
                st.image(input_image, caption="Ảnh chụp từ camera", use_column_width=True)
            except Exception as e:
                st.error(f"Lỗi khi đọc ảnh từ camera: {e}")
                input_image = None

    # Realtime mode note
    elif input_mode == "Realtime (video liên tục)":
        st.info("Realtime sử dụng OpenCV (VideoCapture). Chỉ hoạt động nếu server (nơi chạy Streamlit) có quyền truy cập webcam. Nếu bạn chạy trên máy cục bộ thì OK.")
        st.markdown("Nhấn nút **Bắt đầu** để bật video; **Dừng** để tắt.")
        start_realtime = st.button("Bắt đầu Realtime")
        stop_realtime = st.button("Dừng Realtime")

        # session state quản lý loop
        if "realtime_running" not in st.session_state:
            st.session_state.realtime_running = False

        if start_realtime:
            st.session_state.realtime_running = True
        if stop_realtime:
            st.session_state.realtime_running = False

        # nếu đang chạy realtime, sẽ hiển thị vùng video
        if st.session_state.realtime_running:
            FRAME_WINDOW = st.image([])  # vùng hiển thị khung hình
            try:
                # Mở camera server-side
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("Không thể mở webcam (server). Kiểm tra quyền truy cập hoặc thay chỉ số camera.")
                    st.session_state.realtime_running = False
                else:
                    # Chuẩn bị thay đổi tần suất dự đoán để giảm tải
                    predict_every_n_frames = 10
                    frame_count = 0
                    last_label = None
                    last_desc = None
                    last_conf = 0.0
                    while st.session_state.realtime_running:
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("Không đọc được frame từ webcam.")
                            break
                        # hiển thị frame (BGR -> RGB)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        FRAME_WINDOW.image(frame_rgb)

                        # dự đoán mỗi N frame
                        if frame_count % predict_every_n_frames == 0:
                            pil_img = Image.fromarray(frame_rgb)
                            label, desc, conf = predict_image(pil_img, threshold=threshold)
                            last_label, last_desc, last_conf = label, desc, conf

                        # Hiển thị kết quả nhỏ bên dưới video
                        st.markdown(f"**Kết quả tạm thời:** {last_label} (conf={last_conf:.2f})")
                        if last_label not in ("Không phải biển báo", "Lỗi nhận dạng"):
                            st.caption(last_desc)

                        frame_count += 1
                        # throttle loop để tránh chiếm CPU GPU 100%
                        time.sleep(0.03)

                    cap.release()
            except Exception as e:
                st.error(f"Lỗi realtime: {e}")
                st.session_state.realtime_running = False

# Right column: hiển thị kết quả dự đoán nếu có ảnh đơn
with col2:
    st.subheader("Kết quả nhận diện")
    if input_mode != "Realtime (video liên tục)":
        if input_image is not None:
            label, desc, conf = predict_image(input_image, threshold=threshold)
            if label == "Không phải biển báo":
                st.error("🚫 Không phải biển báo giao thông!")
                st.write(f"Chi tiết: {desc} (conf={conf:.2f})")
            elif label == "Lỗi nhận dạng":
                st.error(f"⚠️ Lỗi: {desc}")
            else:
                st.success(f"✅ Biển báo: {label}")
                st.info(f"📖 Mô tả: {desc}")
                st.progress(min(int(conf * 100), 100))
                st.write(f"🎯 Độ chính xác: **{conf*100:.1f}%**")
        else:
            st.info("Chưa có ảnh để nhận diện. Hãy upload hoặc chụp ảnh, hoặc bật demo.")
    else:
        st.info("Realtime đang chạy (nếu bạn đã nhấn Bắt đầu). Kết quả realtime xuất hiện cạnh khung video.")

st.markdown("---")
st.markdown("<center>© 2025 UTH SmartTasks | Demo nhận diện biển báo giao thông</center>", unsafe_allow_html=True)
