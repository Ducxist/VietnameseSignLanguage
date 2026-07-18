# Vietnamese Sign Language

Đề tài đặt ra bài toán xây dựng một hệ thống phần mềm có khả năng thu nhận hình ảnh từ camera theo thời gian thực, nhận diện các cử chỉ dùng ngôn ngữ kí hiệu tiếng Việt và chuyển đổi chúng thành văn bản, nhằm hỗ trợ giao tiếp giữa người điếc và người bình thường.

## Technologies & Tools

- `python`
- `Google Colab`
- `OpenCV` /  `mediapipe`
- `numpy` / `scipy`/ `pandas` / `json`
- `tensorflow` / `Keras`
- `Matplotlib` / `Seaborn`
  
## Features

- Ghi hình dữ liệu.
- Trích xuất các điểm landmark.
- Nhận điện chính xác trên thời gian thực.
- Ghép kết quả nhận diện thành câu văn bản.
- Hiển thị giao diện trực quan.

## The Process

Hệ thống được tổ chức theo kiến trúc pipeline gồm hai giai đoạn tách biệt: giai đoạn ngoại tuyến (offline) chịu trách nhiệm xây dựng dữ liệu và huấn luyện mô hình, và giai đoạn trực tuyến (online) chịu trách nhiệm nhận diện thời gian thực. Hai giai đoạn chia sẻ chung một bộ hằng số cấu hình (số điểm landmark, độ dài chuỗi, số đặc trưng) để đảm bảo tính nhất quán giữa dữ liệu huấn luyện và dữ liệu suy luận.

## Running the Project

**Quá trình thu thập dữ liệu**
- Run file record_dataset_fixed.py

**Quá trình nhận diện**
- Run file vid1.py

## Preview

https://github.com/user-attachments/assets/1ff35d4b-a4bd-494f-8f68-e42d1af6c4f3

