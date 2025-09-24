# AI Nhận diện Biển báo Giao thông

Dự án sử dụng **Convolutional Neural Network (CNN)** để nhận diện biển báo giao thông Việt Nam/Châu Âu.  
Giao diện trực quan được xây dựng bằng **Streamlit**, có thể upload ảnh, chụp từ webcam hoặc chạy realtime qua camera.  

---

## Tính năng chính
- Huấn luyện mô hình CNN để phân loại **43 loại biển báo giao thông**.
- Lưu và tải mô hình dưới định dạng `.h5`.
- Đánh giá mô hình bằng **Confusion Matrix, Precision, Recall, F1-score**.
- Giao diện Streamlit:
  Upload ảnh từ máy tính.
  Chụp ảnh trực tiếp bằng webcam.
  Realtime video qua OpenCV (chưa thể hoạt động vì mô hình này được phát triển bằng CNN)
  Chế độ demo lấy ảnh ngẫu nhiên từ thư mục `Meta/`.

---

## Cấu trúc thư mục
- ui.py # Giao diện Streamlit để nhận diện biển báo
- traffic_sign.py # Huấn luyện mô hình CNN
- test.py # Kiểm thử mô hình với tập test
- my_model.h5 # Mô hình đã train
- test.h5 # (tùy chọn) file model khác/backup
- X_test.npy # Dữ liệu ảnh test
- y_test.npy # Nhãn test
- traffic.png # Ảnh minh họa/demo
- Train.csv # Metadata/training log
- Training/ # Dữ liệu huấn luyện (chia 43 class, mỗi class 1 thư mục con)
- Testing/ # Dữ liệu kiểm thử (nếu có)
- Meta/ # Thư mục chứa ảnh demo (jpg/png)

---

## Yêu cầu 

- Python >= 3.8
- Các thư viện chính:
  `tensorflow` / `keras`
  `numpy`
  `pillow`
  `opencv-python`
  `matplotlib`
  `scikit-learn`
  `seaborn`
  `streamlit`

Cài đặt nhanh qua `requirements.txt` (tạo file này với nội dung trên):
```bash
pip install -r requirements.txt

## Cách huấn luyện

1. Huấn luyện mô hình
- python traffic_sign.py
  Mô hình sau khi train sẽ được lưu thành my_model.h5
  Dữ liệu test lưu ra X_test.npy, y_test.npy
2. Kiểm thử mô hình
- python test.py
  Xuất độ chính xác
  Vẽ Confusion Matrix trực quan bằng Seaborn
3. Chạy giao diện Streamlit
- streamlit run gui.py
  Mở giao diện web tại http://localhost:8501.
  Chọn chế độ: Upload ảnh, Camera, hoặc Realtime.
  Có thể bật chế độ Demo để lấy ảnh ngẫu nhiên trong thư mục Meta/.
