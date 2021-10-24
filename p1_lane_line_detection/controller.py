import cv2
import numpy as np

def find_lane_lines(img):
    """Phát hiện vạch kẻ đường
    Hàm này sẽ nhận vào một hình ảnh màu, ở hệ màu BGR,
    trả ra hình ảnh các vạch kẻ đường đã được lọc
    """

    # Chuyển ảnh đã đọc sang ảnh xám
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Áp dụng bộ lọc Gaussian để loại bỏ bớt nhiễu.
    # Các bạn có thể thử nghiệm các bộ lọc khác tại đây,
    # như bộ lọc Median hoặc Bilateral.
    img_gauss = cv2.GaussianBlur(gray, (11, 11), 0)

    # Áp dụng bộ lọc Canny với 2 ngưỡng.
    # Các bạn có thể điều chỉnh 2 ngưỡng này và xem sự thay đổi
    # trong ảnh kết quả
    thresh_low = 150
    thresh_high = 200
    img_canny = cv2.Canny(img_gauss, thresh_low, thresh_high)

    # Trả về kết quả vạch kẻ đường
    return img_canny

def birdview_transform(img):
    """Áp dụng chuyển đổi birdview
    """
    IMAGE_H = 480
    IMAGE_W = 640
    src = np.float32([[0, IMAGE_H], [640, IMAGE_H], [0, IMAGE_H * 0.4], [IMAGE_W, IMAGE_H * 0.4]])
    dst = np.float32([[240, IMAGE_H], [640 - 240, IMAGE_H], [-160, 0], [IMAGE_W+160, 0]])
    M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
    warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H)) # Image warping
    return warped_img

def calculate_control_signal(img):
    """Tính toán để tìm tốc độ và góc lái cho xe từ ảnh đầu vào
    """

    # Xử lý ảnh để tìm các điểm trái / phải
    img_lines = find_lane_lines(img)
    img_birdview = birdview_transform(img_lines)

    # Hiển thị kết quả xử lý ảnh
    cv2.imshow("Result", img_birdview)
    cv2.waitKey(1)

    # Trả về góc lái và tốc độ
    return 0, 0
