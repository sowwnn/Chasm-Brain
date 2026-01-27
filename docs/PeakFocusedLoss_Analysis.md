# Phân Tích PeakFocusedLoss

## Câu hỏi: Loss có tính cả đỉnh dưới 0 không?

### Trả lời: **CÓ** ✅

PeakFocusedLoss **KHÔNG phân biệt** đỉnh dương (>0) hay đỉnh âm (<0). Cả hai loại đỉnh đều được coi là quan trọng.

## Cách hoạt động hiện tại

```python
# Trong ARMNet/loss.py, dòng 40
deviation = torch.abs(target - mean_fmri)
```

Hàm sử dụng `torch.abs()` để tính **độ lệch tuyệt đối** so với mean:
- Đỉnh dương (target > mean): `deviation = target - mean`
- Đỉnh âm (target < mean): `deviation = mean - target`

Cả hai đều được tính như nhau!

### Ví dụ:
```
mean_fmri = 0.0
tau = 0.5

Voxel A: target = +2.0  → deviation = |2.0 - 0.0| = 2.0 → weight = 1 + alpha * (2.0 - 0.5)
Voxel B: target = -2.0  → deviation = |-2.0 - 0.0| = 2.0 → weight = 1 + alpha * (2.0 - 0.5)
```

**Kết luận**: Voxel A và B có cùng trọng số, nghĩa là đỉnh dương và đỉnh âm được coi là **quan trọng như nhau**.

## Tại sao thiết kế như vậy?

Trong fMRI:
- **Đỉnh dương** (positive activation): Vùng não hoạt động mạnh
- **Đỉnh âm** (negative activation/deactivation): Vùng não bị ức chế

Cả hai đều mang **thông tin quan trọng** về hoạt động não bộ!

## Nếu muốn phân biệt đỉnh dương/âm

Nếu bạn muốn ưu tiên đỉnh dương hơn đỉnh âm (hoặc ngược lại), có thể sửa như sau:

### Option 1: Chỉ tính đỉnh dương
```python
# Chỉ phạt nặng khi target > mean (đỉnh dương)
deviation_positive = torch.relu(target - mean_fmri)
weights = 1.0 + self.alpha * torch.relu(deviation_positive - self.tau)
```

### Option 2: Chỉ tính đỉnh âm
```python
# Chỉ phạt nặng khi target < mean (đỉnh âm)
deviation_negative = torch.relu(mean_fmri - target)
weights = 1.0 + self.alpha * torch.relu(deviation_negative - self.tau)
```

### Option 3: Trọng số khác nhau cho đỉnh dương/âm
```python
# Thêm tham số alpha_positive và alpha_negative
deviation_positive = torch.relu(target - mean_fmri)
deviation_negative = torch.relu(mean_fmri - target)

weight_positive = 1.0 + self.alpha_positive * torch.relu(deviation_positive - self.tau)
weight_negative = 1.0 + self.alpha_negative * torch.relu(deviation_negative - self.tau)

# Kết hợp
weights = torch.where(target > mean_fmri, weight_positive, weight_negative)
```

## Khuyến nghị

**Giữ nguyên thiết kế hiện tại** (tính cả đỉnh dương và âm) vì:
1. ✅ Cả activation và deactivation đều quan trọng trong fMRI
2. ✅ Đơn giản và hiệu quả
3. ✅ Phù hợp với hầu hết các task fMRI reconstruction

Chỉ nên thay đổi nếu:
- Bạn có lý do cụ thể từ domain knowledge
- Phân tích cho thấy model đang bỏ qua một loại đỉnh nào đó
- Có ground truth cho thấy đỉnh dương quan trọng hơn (hoặc ngược lại)

## Visualization để kiểm tra

Bạn có thể thêm visualization để xem model predict đỉnh dương/âm như thế nào:

```python
# Trong visualization
positive_peaks = target > (mean_fmri + tau)
negative_peaks = target < (mean_fmri - tau)

# Plot riêng cho đỉnh dương và âm
plt.scatter(target[positive_peaks], pred[positive_peaks], label='Positive Peaks', alpha=0.5)
plt.scatter(target[negative_peaks], pred[negative_peaks], label='Negative Peaks', alpha=0.5)
```

Điều này giúp bạn thấy được model có bias về một loại đỉnh nào không.
