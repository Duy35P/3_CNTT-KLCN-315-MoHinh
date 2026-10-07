# Đặc tả inference Tầng 3 (Xiaomi) - bàn giao Backend

Phiên bản: `week7_xiaomi_tier3_final`. Hàm vào: `predict_anomaly_week7()` trong `inference_week7.py`.

## 1. Quyết định đã chốt (GVHD)
- Tier 3 (>= 14 ngày hợp lệ) dùng **LSTM Autoencoder + Isolation Forest, kết hợp OR**.
- Giải thích dùng **sai số tái tạo theo từng feature** của LSTM-AE (giống Tầng 2). **Không** dùng SHAP lúc chạy.
- Isolation Forest: **một file** `isolation_forest_v8.joblib`, `contamination = 0.10` (giá trị tiên nghiệm, không dò trên Xiaomi).
  Quyết định thực tế dùng **ngưỡng điểm** `isolation_forest.score_threshold` trong config, KHÔNG dùng `predict()`.

## 2. Hợp đồng gọi hàm
```python
from inference_week7 import predict_anomaly_week7
result = predict_anomaly_week7(user_id, daily_history, age=None)
```
- Hàm không giữ trạng thái. BE phải truyền **toàn bộ lịch sử ngày thô** của user (chưa z-score), kể cả hôm nay.
- Ngày có `date` lớn nhất là ngày được đánh giá.
- `age` mặc định lấy `cold_start.assumed_age_default` (70) nếu thiếu.

### Input: `daily_history` = list[dict], mỗi dict là 1 ngày lịch
| Field | Kiểu | Bắt buộc | Ghi chú |
|---|---|---|---|
| `date` | "YYYY-MM-DD" | có | Theo múi giờ người đeo |
| `TotalSteps`, `TotalMinutesAsleep`, `TotalTimeInBed` | number | có (có thể null) | |
| `heart_rate_mean`, `heart_rate_min`, `heart_rate_max` | number | có (có thể null) | bpm, đã loại nhiễu ngoài [30, 220] |
| `resting_hr_p10` | number | có (có thể null) | Phân vị 10% HR trong ngày |
| `SpO2_min` | number | không | Chỉ tham khảo |
| `valid_day` | bool | có | `true` khi đủ HR (>= 60 mẫu, >= 8 giờ khác nhau), steps > 0 và sleep > 0 |

Giá trị đưa vào là giá trị thô. Không tự điền ngày thiếu (không nội suy); ngày thiếu thì để `valid_day=false` hoặc bỏ hàng.

### Output (dict)
| Field | Ý nghĩa |
|---|---|
| `status` | `ok` / `no_data` / `no_heart_rate_today` |
| `tier` | 1, 2 hoặc 3 (theo số ngày hợp lệ) |
| `alert` | `NONE` / `SOFT` / `CRITICAL` |
| `reason` | Chuỗi tiếng Việt cho UI |
| `action` | Gợi ý hành động cho UI |
| `anomaly_score` | Tier 1/2: z của HR nghỉ. Tier 3: MSE của LSTM-AE (null nếu không có cửa sổ) |
| `n_valid_days`, `valid_day`, `date`, `user_id`, `model_version` | Thông tin kèm theo |
| `rule_hits` | Các luật cứng đã kích hoạt |
| `top_features` | Tier 3: top 3 feature theo sai số tái tạo: `feature`, `name_vi`, `recon_error`, `share`, `z_today` |
| `ml` | Tier 3: `status`, `lstm_mse`, `lstm_threshold`, `lstm_anomaly`, `isof_score`, `isof_threshold`, `isof_anomaly`, `hybrid_anomaly`. `status="no_consecutive_window"` khi chưa có 7 ngày hợp lệ liên tiếp |

## 2b. Cách tính đặc trưng ngày từ dữ liệu thô Xiaomi (BE phải làm đúng như vậy)
Hàm inference nhận các cột ngày đã tính sẵn; nếu BE tự tính từ dữ liệu thô thì phải theo đúng quy tắc sau (lấy từ notebook, phần A3-A4):
- **Ngày lịch:** gom theo ngày địa phương của người đeo (mặc định UTC+7, cấu hình riêng theo user nếu khác múi giờ).
- **Nhịp tim:** bỏ mẫu ngoài [30, 220] bpm. `heart_rate_mean/min/max` tính trên các mẫu còn lại; `resting_hr_p10` là phân vị 10% của các mẫu trong ngày.
  Ngày có < 60 mẫu hoặc các mẫu phủ < 8 giờ khác nhau thì cả 4 chỉ số nhịp tim là `null` (không nội suy) và `hr_ok = false`.
- **Bước chân:** `TotalSteps` = tổng số bước trong ngày. Nếu không có ghi nhận hoặc bằng 0 thì `null` và `steps_ok = false`.
- **Giấc ngủ (phút):** mỗi giấc được gán vào **ngày thức dậy**, bỏ bản ghi trùng (cùng giờ ngủ và giờ dậy).
  `TotalMinutesAsleep` = deep + light + rem nếu có đủ ba giai đoạn, nếu không dùng `duration`. `TotalTimeInBed` = (giờ dậy - giờ ngủ) / 60.
  Loại giấc có thời gian trên giường < 60 hoặc > 900 phút, hoặc thời gian ngủ < 60 hoặc > 960 phút. Ngày có nhiều giấc thì cộng lại. `sleep_ok` khi tổng > 0.
- **`valid_day`** = `hr_ok` **và** `steps_ok` **và** `sleep_ok`.
- Không nội suy ngày thiếu: để `valid_day = false` hoặc không gửi hàng đó.

## 3. Logic theo tier
Số ngày hợp lệ = số ngày `valid_day=true` đến hôm nay (ngày không hợp lệ không làm tăng tier).
| Tier | Ngày hợp lệ | Tín hiệu |
|---|---|---|
| 1 | 1-3 | z của `resting_hr_p10` so với chuẩn dân số theo tuổi (`population_baseline.json`); SOFT nếu `abs(z) > 2.5` |
| 2 | 4-13 | z của `resting_hr_p10` so với baseline tạm (các ngày hợp lệ trước, cần >= 3); SOFT nếu `abs(z) > 2.5` |
| 3 | >= 14 | Cửa sổ 7 ngày hợp lệ **liên tiếp** kết thúc hôm nay, z-score cá nhân (6 feature) -> LSTM-AE và IF; SOFT nếu `mse > lstm_threshold` **hoặc** `isof_score > isof_threshold` |

- Luật cứng chạy ở **mọi tier và mọi ngày có nhịp tim**: HR trung bình > 100, HR nghỉ (p10) < 50, SpO2 min < 90 -> `CRITICAL`.
- ML chỉ tạo `SOFT`. `CRITICAL` chỉ đến từ luật cứng.
- Tier 3 mà thiếu 7 ngày liên tiếp: chỉ áp dụng luật cứng, `ml.status = "no_consecutive_window"`.
- Mọi hằng số nằm trong `config_week7.json` (`cold_start`, `clinical`, `lstm_ae.threshold_value`, `isolation_forest.score_threshold`). BE không hard-code.

## 4. Feature và giới hạn
- Tier 3 dùng đúng 6 feature: TotalSteps, TotalMinutesAsleep, TotalTimeInBed, heart_rate_mean/min/max.
- **Không có `stress_index_z`** (Xiaomi thiếu SedentaryMinutes). Không so sánh trực tiếp với stress của Fitbit.
- Nếu UI cần hiển thị "stress" cho người dùng cuối, dùng `activity_index` (ước lượng) và gắn nhãn "Chỉ số hoạt động bất thường (ước lượng)".
- Dữ liệu huấn luyện/kiểm tra không phải người cao tuổi, không có nhãn thật: kết quả là cảnh báo tham khảo, không phải chẩn đoán.

## 5. Gói file (`xiaomi_tier3_package_final/`)
`config_week7.json`, `lstm_autoencoder_v8.pth`, `isolation_forest_v8.joblib`, `population_baseline.json`,
`feature_names_vi.json`, `inference_week7.py`, `inference_spec_week7.md`, `requirements.txt` (pin phiên bản torch/scikit-learn/joblib/numpy/pandas).
`isolation_forest_optimized.joblib` và `shap_values.npy` **không** nằm trong gói bàn giao.

## 6. Lưu ý vận hành
- Cần cùng phiên bản `scikit-learn` với lúc huấn luyện để nạp `.joblib` (xem `requirements.txt`).
- z-score lúc huấn luyện dùng toàn bộ ngày hợp lệ của user; lúc inference dùng các ngày hợp lệ đến hôm nay, nên điểm có thể lệch nhẹ so với kết quả notebook.
