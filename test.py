import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# ======== 1. Load mô hình và dữ liệu ========
model = load_model('my_model.h5')

# Nếu bạn đã lưu dữ liệu test ra file npy trước đó thì dùng:
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# ======== 2. Kiểm tra và xử lý nhãn y_test ========
# Nếu y_test là one-hot (2 chiều), dùng argmax để lấy nhãn thật
if len(y_test.shape) > 1 and y_test.shape[1] > 1:
    y_true = np.argmax(y_test, axis=1)
else:
    y_true = y_test  # y_test đã là dạng nhãn thường

# ======== 3. Dự đoán kết quả từ mô hình ========
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# ======== 4. In báo cáo chi tiết Precision, Recall, F1-score ========
print("===== Báo cáo chi tiết Precision, Recall, F1-score =====")
print(classification_report(y_true, y_pred_classes, digits=4))

# ======== 5. Tính Accuracy toàn bộ tập kiểm tra ========
acc = accuracy_score(y_true, y_pred_classes)
print(f"🎯 Accuracy toàn bộ tập kiểm tra: {acc * 100:.2f}%")

# ======== 6. Vẽ ma trận nhầm lẫn ========
cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, cmap='Blues', fmt='g')
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.title('Ma trận nhầm lẫn (Confusion Matrix)')
plt.show()
