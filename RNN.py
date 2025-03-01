import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. Xây dựng mô hình RNN
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        # Lớp RNN
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        # Lớp fully connected để dự đoán đầu ra
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Lấy đầu ra và trạng thái ẩn từ RNN
        out, _ = self.rnn(x)
        # Lấy giá trị của bước cuối cùng trong chuỗi
        out = self.fc(out[:, -1, :])
        return out

# 2. Tạo dữ liệu giả cho bài toán dự đoán chuỗi số
def generate_data(seq_length=10):
    data = np.array([i for i in range(1, seq_length + 1)])
    X, y = [], []
    for i in range(len(data) - 1):
        X.append(data[i:i+1])  # Tạo dữ liệu input (1 bước)
        y.append(data[i+1])    # Mục tiêu là giá trị tiếp theo
    return np.array(X), np.array(y)

# Tạo dữ liệu
X, y = generate_data()

# Chuyển dữ liệu thành tensor PyTorch
X = torch.Tensor(X).view(-1, 1, 1)  # (batch_size, sequence_length, input_size)
y = torch.Tensor(y).view(-1, 1)     # (batch_size, output_size)

# 3. Khởi tạo mô hình, hàm mất mát và tối ưu hóa
input_size = 1  # Dữ liệu đầu vào là giá trị duy nhất tại mỗi bước
hidden_size = 5 # Số lượng nơ-ron trong lớp ẩn
output_size = 1 # Dự đoán một giá trị
num_epochs = 100
learning_rate = 0.01

model = RNNModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()  # Hàm mất mát: Mean Squared Error
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 4. Huấn luyện mô hình
losses = []

for epoch in range(num_epochs):
    model.train()

    # Tiến hành forward và backward
    optimizer.zero_grad() # Xóa các gradient cũ trước khi tính toán gradient mới không backward bị cộng dồn
    outputs = model(X)  # Dự đoán đầu ra
    loss = criterion(outputs, y)  # Tính toán lỗi
    loss.backward()  # Lan truyền ngược
    # tính toán gradient của hàm mất mát đối với tất cả các tham số trong mô hình 
    optimizer.step()  # Cập nhật trọng số

    losses.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# 5. Dự đoán giá trị
model.eval()
with torch.no_grad():
    test_input = torch.Tensor([[9]])  # Dự đoán giá trị tiếp theo của chuỗi [1, 2, 3, 4, ..., 9]
    test_input = test_input.view(-1, 1, 1)  # Đảm bảo input đúng dạng (batch_size, sequence_length, input_size)
    predicted = model(test_input)
    print(f'Giá trị dự đoán tiếp theo là: {predicted.item()}')

# 6. Đánh giá và vẽ đồ thị
plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

