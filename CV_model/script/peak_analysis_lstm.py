import csv
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


# 依據 boundary_value (預設為 '0.800') 將 CSV 資料切割成多個 segments。
def segment_data_by_value(file_path, boundary_value='0.800'):
    
    segments = []
    current_segment = []

    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        
        for row in reader:
            # 檢查該行是否有內容，且該行第一欄是否為 boundary_value
            if row and row[0] == boundary_value:
                # 如果 current_segment 有資料，代表要先結束前一段
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []
                # 開啟新 segment 從當前這一行 (含 0.800) 開始
                current_segment.append(row)
            else:
                # 若非 0.800 的行，就持續加入 current_segment
                current_segment.append(row)

        # 迴圈結束後，如果 current_segment 裡還有資料，也要加入
        if current_segment:
            segments.append(current_segment)

    return segments


def is_negative_scan(segment):
    
    try:
        start_potential = float(segment[0][0])
        end_potential = float(segment[-1][0])
        return start_potential > end_potential
    except:
        return False


def find_negative_peak_in_range(segment, v_lower=-0.4, v_upper=-0.1):
    
    potentials = []
    currents = []
    
    for row in segment:
        try:
            v = float(row[0])
            i = float(row[1])
        except:
            continue

        # 篩選落在指定區間內的點
        if v_lower <= v <= v_upper:
            potentials.append(v)
            currents.append(i)

    if not potentials:
        return None

    # 找到電流最小值之索引
    min_i_idx = np.argmin(currents)
    return (potentials[min_i_idx], currents[min_i_idx])


def extract_peaks_from_file(file_path, v_lower=-0.4, v_upper=-0.1):
    
    segments = segment_data_by_value(file_path, boundary_value="0.800")

    peaks = []
    for seg in segments:
        if is_negative_scan(seg):
            peak = find_negative_peak_in_range(seg, v_lower, v_upper)
            if peak is not None:
                peaks.append(peak)
    return peaks

# 繪製一個 segment，並儲存圖片(可選)。
def plot_segment(segment, save_path=None):
    
    
    
    x = []
    y = []
    for row in segment:
        try:
            x.append(float(row[0]))
            y.append(float(row[1]))
        except:
            continue

    plt.figure()
    plt.plot(x, y)
    plt.xlabel('Potential/V')
    plt.ylabel('Current/A')
    if save_path:
        plt.savefig(save_path)
    plt.close()


# LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=50, num_layers=1):
        
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 建立 LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # 輸出層 (從 hidden state -> 2 維 (V, I))
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        
        # LSTM 輸出: (batch, seq_len, hidden_size), (h_n, c_n)
        out, (h_n, c_n) = self.lstm(x)
        # 我們只取 seq_len 最後一個時刻的隱藏狀態 out[:, -1, :]
        out = out[:, -1, :]
        out = self.fc(out)
        return out


def build_and_train_lstm(peak_sequence, epochs=100, lr=1e-3, device="cpu"):
   
    data = np.array(peak_sequence, dtype=np.float32)  # shape (N, 2)
    if len(data) < 2:
        print("資料點太少，無法進行 LSTM 訓練。")
        return None

    # 單步預測 X(t) -> y(t+1)
    X_data = data[:-1]  # (N-1, 2)
    y_data = data[1:]   # (N-1, 2)

    # 將 X reshape 成 (N-1, seq_len=1, input_size=2)
    X_data = X_data.reshape(-1, 1, 2)  # (N-1, 1, 2)

    # 轉成 Tensor
    X_tensor = torch.tensor(X_data)  # shape (N-1, 1, 2)
    y_tensor = torch.tensor(y_data)  # shape (N-1, 2)

    # 建立模型
    model = LSTMModel(input_size=2, hidden_size=50)
    model = model.to(device)
    X_tensor = X_tensor.to(device)
    y_tensor = y_tensor.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 訓練
    for ep in range(epochs):
        model.train()
        optimizer.zero_grad()

        output = model(X_tensor)         # (N-1, 2)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()

        if (ep+1) % 10 == 0:
            print(f"Epoch [{ep+1}/{epochs}], Loss: {loss.item():.6f}")

    return model


def predict_next_peak(model, last_peak, device="cpu"):

    model.eval()
    with torch.no_grad():
        inp = np.array(last_peak, dtype=np.float32).reshape(1, 1, 2)  # (batch=1, seq_len=1, input_size=2)
        inp_tensor = torch.tensor(inp).to(device)
        pred = model(inp_tensor)  # shape (1,2)
        pred = pred.cpu().numpy()[0]  # 轉回 numpy, shape (2,)
    return pred



# main
# 示意用

if __name__ == "__main__":

    csv_file_path = "../../CV_data/Electrode1/0 mM/1_10_0mM.csv"  # 檔案路徑

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # 1. 從檔案擷取負掃描時的峰值
    peaks = extract_peaks_from_file(csv_file_path, v_lower=-0.4, v_upper=-0.1)
    print("在負掃描的 -0.4V ~ -0.1V 區間找到的峰值列表：")
    for idx, (v, i) in enumerate(peaks):
        print(f"Cycle {idx+1}: V={v:.4f}, I={i:.6e}")

    # 2. 使用 PyTorch LSTM 進行下一個峰值預測 (單步預測)
    model = build_and_train_lstm(peaks, epochs=50, lr=1e-3, device=device)
    if model is not None and len(peaks) > 1:
        last_peak = peaks[-1]
        next_peak_pred = predict_next_peak(model, last_peak, device=device)
        print(f"\n根據最後一個峰值 {last_peak}，預測下一個峰值：{next_peak_pred}")
    

