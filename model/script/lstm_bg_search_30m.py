from itertools import product
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from math import sqrt

# (1) 讀取與前處理函式 


def load_single_patient_data(file_path):
    """
    讀取單一病患的 CSV 時序資料 (例如包含欄位 SensorGLU)。
    假設檔案中已根據時間排序，或不需要排序。
    若需要排序，可自行在此函式中排序。
    回傳: 該病患的血糖時序資料 (np.array)
    """
    df = pd.read_csv(file_path)
    # 若尚未排序，可依實際情況對時間欄位排序，如:
    # df = df.sort_values(by="ReadingDt")  # 或其它時間欄位

    # 取出 SensorGLU 欄位，去除 NaN
    sensor_values = df["SensorGLU"].dropna().values
    return sensor_values

def create_sequences(data, seq_length, horizon=6):
    """
    給定一維資料 data,
    seq_length: 用多少歷史長度來預測,
    horizon:    要往後幾個時間步預測(若資料5分鐘一筆, horizon=6代表30分鐘後).
    
    回傳 X, y:
        X shape: [batch_size, seq_length, 1]
        y shape: [batch_size, 1]
    """
    Xs, Ys = [], []
    # 留出 horizon，確保有足夠時間步可做預測
    for i in range(len(data) - seq_length - horizon + 1):
        x = data[i : i + seq_length]
        y = data[i + seq_length + horizon - 1]
        Xs.append(x)
        Ys.append(y)
    X = np.array(Xs).reshape(-1, seq_length, 1)
    y = np.array(Ys).reshape(-1, 1)
    return X, y

def create_dataloader(X, y, batch_size=16, shuffle=True):
    tensor_X = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(tensor_X, tensor_y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def count_rows_in_csv_files(directory):
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    files = []
    for csv_file in csv_files:
        file_path = os.path.join(directory, csv_file)
        df = pd.read_csv(file_path)

        files.append({csv_file: len(df)})
    files.sort(key=lambda x: list(x.values())[0], reverse=True)
    return files

# (2) 模型與訓練函式  


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, fc_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, fc_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_dim, 1)

    def forward(self, x):
        # x shape: [batch_size, seq_length, input_dim]
        lstm_out, (h_n, c_n) = self.lstm(x)
        # 取最後一個時間步的輸出
        last_time_step = lstm_out[:, -1, :]  # shape: [batch_size, hidden_dim]
        x = self.fc1(last_time_step)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def evaluate_model(model, dataloader, device):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        if len(dataloader) == 0:
            print("Validation dataloader is empty!")
            return float('nan')

        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch)

            if outputs.shape[0] == 0:  # 確保有輸出
                print("Empty outputs from model!")
                continue

            preds.append(outputs.cpu().numpy())
            trues.append(y_batch.cpu().numpy())

    if len(preds) == 0:  # 確保至少有一個批次被處理
        print("No predictions were made. Returning NaN.")
        return float('nan')

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    rmse_val = np.sqrt(((preds - trues) ** 2).mean())
    return float(rmse_val)



# (3) Step 1,2,3 搜尋 


def search_best_LU(sub_train_data, val_data, fixed_DU, fixed_SL,
                   device='cpu', num_epochs=10, batch_size=16):
    """
    Step 1: 固定 DU, SL，搜尋最佳 LU
    回傳 (best_LU, best_val_rmse)
    """
    # 先建立序列資料
    X_sub_train, y_sub_train = create_sequences(sub_train_data, fixed_SL)
    X_val, y_val = create_sequences(val_data, fixed_SL)

    train_loader = create_dataloader(X_sub_train, y_sub_train, batch_size=batch_size)
    val_loader = create_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False)

    candidate_LU = [5, 10, 20, 30, 40, 50, 60, 70]
    best_LU = None
    best_rmse = float('inf')

    for LU in candidate_LU:
        model = LSTMModel(input_dim=1, hidden_dim=LU, fc_dim=fixed_DU).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 訓練
        for epoch in range(num_epochs):
            _ = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # 驗證
        val_rmse = evaluate_model(model, val_loader, device)
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_LU = LU

    return best_LU, best_rmse

def search_best_DU(sub_train_data, val_data, best_LU, fixed_SL,
                   device='cpu', num_epochs=10, batch_size=16):
    """
    Step 2: 固定 LU, SL，搜尋最佳 DU
    回傳 (best_DU, best_val_rmse)
    """
    X_sub_train, y_sub_train = create_sequences(sub_train_data, fixed_SL)
    X_val, y_val = create_sequences(val_data, fixed_SL)

    train_loader = create_dataloader(X_sub_train, y_sub_train, batch_size=batch_size)
    val_loader = create_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False)

    candidate_DU = [10, 20, 30, 40, 50]
    best_DU = None
    best_rmse = float('inf')

    for DU in candidate_DU:
        model = LSTMModel(input_dim=1, hidden_dim=best_LU, fc_dim=DU).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            _ = train_one_epoch(model, train_loader, criterion, optimizer, device)

        val_rmse = evaluate_model(model, val_loader, device)
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_DU = DU

    return best_DU, best_rmse

def search_best_SL(sub_train_data, val_data, best_LU, best_DU,
                   device='cpu', num_epochs=10, batch_size=16):
    """
    Step 3: 固定 LU, DU，搜尋最佳 SL
    回傳 (best_SL, best_val_rmse)
    """
    candidate_SL = [5, 10, 15, 20, 30, 40, 50, 100]
    best_SL = None
    best_rmse = float('inf')

    for SL in candidate_SL:
        X_sub_train, y_sub_train = create_sequences(sub_train_data, SL)
        X_val, y_val = create_sequences(val_data, SL)

        train_loader = create_dataloader(X_sub_train, y_sub_train, batch_size=batch_size)
        val_loader = create_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False)

        model = LSTMModel(input_dim=1, hidden_dim=best_LU, fc_dim=best_DU).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            _ = train_one_epoch(model, train_loader, criterion, optimizer, device)

        val_rmse = evaluate_model(model, val_loader, device)
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_SL = SL

    return best_SL, best_rmse

def search_best_LU_across_patients(patient_list,
                                   candidate_LU,
                                   fixed_DU,
                                   fixed_SL,
                                   device='cpu',
                                   num_epochs=10,
                                   batch_size=16,
                                   val_ratio=0.2,
                                   patience=20,
                                   lr=0.001):
    """
    Step 1: 固定 DU, SL，搜尋 LU。
    會回傳:
      best_LU
      best_rmse (平均驗證 RMSE)
      candidate_to_patient_rmse: dict，key = LU, value = list of rmse per patient
    """

    best_LU = None
    best_rmse = float('inf')

    # 用來存每個候選 LU 在每位病患上的驗證 RMSE（方便後續畫圖）
    # 結構: { LU值: [病患1的RMSE, 病患2的RMSE, ..., 病患N的RMSE] }
    candidate_to_patient_rmse = {lu: [] for lu in candidate_LU}

    # 取得總病患數
    num_patients = len(patient_list)

    for lu in candidate_LU:
        rmse_list = []  # 收集所有病患的RMSE(用於計算該lu的平均)

        for idx, data in enumerate(patient_list):
            # (1) 先做 train/val 切分
            n = len(data)
            train_size = int(0.66 * n)
            train_data = data[:train_size]
            val_size = int(val_ratio * len(train_data))
            sub_train_data = train_data[:-val_size]
            val_data       = train_data[-val_size:]

            # (2) 產生序列
            if len(sub_train_data) < fixed_SL + 1 or len(val_data) < fixed_SL + 1:
                # 若資料不足，給個 NaN 或 直接跳過
                candidate_to_patient_rmse[lu].append(np.nan)
                continue

            X_sub_train, y_sub_train = create_sequences(sub_train_data, fixed_SL)
            X_val,       y_val       = create_sequences(val_data, fixed_SL)

            train_loader = create_dataloader(X_sub_train, y_sub_train, batch_size=batch_size)
            val_loader   = create_dataloader(X_val,       y_val,       batch_size=batch_size, shuffle=False)

            # (3) 建立並訓練模型
            model = LSTMModel(input_dim=1, hidden_dim=lu, fc_dim=fixed_DU).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            # Early Stopping 相關變數
            best_val_loss = float('inf')
            patience_counter = 0

            for epoch in range(num_epochs):
                _ = train_one_epoch(model, train_loader, criterion, optimizer, device)

                # 計算驗證 RMSE
                val_rmse = evaluate_model(model, val_loader, device)

                # 檢查 Early Stopping 條件
                if val_rmse < best_val_loss:
                    best_val_loss = val_rmse
                    patience_counter = 0  # 重置 patience_counter
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping triggered for LU={lu} at epoch {epoch + 1}")
                    break

            # (4) 計算驗證RMSE
            val_rmse = evaluate_model(model, val_loader, device)
            rmse_list.append(val_rmse)

            # 儲存到 candidate_to_patient_rmse
            candidate_to_patient_rmse[lu].append(val_rmse)

        # 計算此 lu 的平均RMSE
        if len(rmse_list) == 0:
            avg_val_rmse = float('inf')
        else:
            avg_val_rmse = np.mean(rmse_list)

        print(f"LU={lu}, Average Val RMSE={avg_val_rmse:.4f}")
        if avg_val_rmse < best_rmse:
            best_rmse = avg_val_rmse
            best_LU = lu

    return best_LU, best_rmse, candidate_to_patient_rmse

def search_best_DU_across_patients(patient_list,
                                   candidate_DU,
                                   Best_LU,
                                   fixed_SL,
                                   device='cpu',
                                   num_epochs=10,
                                   batch_size=16,
                                   val_ratio=0.2,
                                   patience=20,
                                   lr=0.001):
    """
    Step 2: 固定 LU, SL，搜尋 DU。
    會回傳:
      best_DU
      best_rmse (平均驗證 RMSE)
      candidate_to_patient_rmse: dict，key = DU, value = list of rmse per patient
    """

    best_DU = None
    best_rmse = float('inf')

    # 用來存每個候選 LU 在每位病患上的驗證 RMSE（方便後續畫圖）
    # 結構: { LU值: [病患1的RMSE, 病患2的RMSE, ..., 病患N的RMSE] }
    candidate_to_patient_rmse = {du: [] for du in candidate_DU}

    # 取得總病患數
    num_patients = len(patient_list)

    for du in candidate_DU:
        rmse_list = []  # 收集所有病患的RMSE(用於計算該lu的平均)

        for idx, data in enumerate(patient_list):
            # (1) 先做 train/val 切分
            n = len(data)
            train_size = int(0.66 * n)
            train_data = data[:train_size]
            val_size = int(val_ratio * len(train_data))
            sub_train_data = train_data[:-val_size]
            val_data       = train_data[-val_size:]

            # (2) 產生序列
            if len(sub_train_data) < fixed_SL + 1 or len(val_data) < fixed_SL + 1:
                # 若資料不足，給個 NaN 或 直接跳過
                candidate_to_patient_rmse[du].append(np.nan)
                continue

            X_sub_train, y_sub_train = create_sequences(sub_train_data, fixed_SL)
            X_val,       y_val       = create_sequences(val_data, fixed_SL)

            train_loader = create_dataloader(X_sub_train, y_sub_train, batch_size=batch_size)
            val_loader   = create_dataloader(X_val,       y_val,       batch_size=batch_size, shuffle=False)

            # (3) 建立並訓練模型
            model = LSTMModel(input_dim=1, hidden_dim=Best_LU, fc_dim=du).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # Early Stopping 相關變數
            best_val_loss = float('inf')
            patience_counter = 0

            for epoch in range(num_epochs):
                _ = train_one_epoch(model, train_loader, criterion, optimizer, device)

                # 計算驗證 RMSE
                val_rmse = evaluate_model(model, val_loader, device)

                # 檢查 Early Stopping 條件
                if val_rmse < best_val_loss:
                    best_val_loss = val_rmse
                    patience_counter = 0  # 重置 patience_counter
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping triggered for DU={du} at epoch {epoch + 1}")
                    break

            # (4) 計算驗證RMSE
            val_rmse = evaluate_model(model, val_loader, device)
            rmse_list.append(val_rmse)

            # 儲存到 candidate_to_patient_rmse
            candidate_to_patient_rmse[du].append(val_rmse)

        # 計算此 du 的平均RMSE
        if len(rmse_list) == 0:
            avg_val_rmse = float('inf')
        else:
            avg_val_rmse = np.mean(rmse_list)

        print(f"DU={du}, Average Val RMSE={avg_val_rmse:.4f}")
        if avg_val_rmse < best_rmse:
            best_rmse = avg_val_rmse
            best_DU = du

    return best_DU, best_rmse, candidate_to_patient_rmse
    
def search_best_SL_across_patients(patient_list,
                                   candidate_SL,
                                   Best_LU,
                                   Best_DU,
                                   device='cpu',
                                   num_epochs=10,
                                   batch_size=16,
                                   val_ratio=0.2,
                                   patience=20,
                                   lr=0.001):
    """
    Step 3: 固定 DU, LU，搜尋 SL。
    會回傳:
      best_SL
      best_rmse (平均驗證 RMSE)
      candidate_to_patient_rmse: dict，key = SL, value = list of rmse per patient
    """

    best_SL = None
    best_rmse = float('inf')

    # 用來存每個候選 LU 在每位病患上的驗證 RMSE（方便後續畫圖）
    # 結構: { LU值: [病患1的RMSE, 病患2的RMSE, ..., 病患N的RMSE] }
    candidate_to_patient_rmse = {sl: [] for sl in candidate_SL}

    # 取得總病患數
    num_patients = len(patient_list)

    for sl in candidate_SL:
        rmse_list = []  # 收集所有病患的RMSE(用於計算該lu的平均)

        for idx, data in enumerate(patient_list):
            # (1) 先做 train/val 切分
            n = len(data)
            train_size = int(0.66 * n)
            train_data = data[:train_size]
            val_size = int(val_ratio * len(train_data))
            sub_train_data = train_data[:-val_size]
            val_data       = train_data[-val_size:]

            # (2) 產生序列
            if len(sub_train_data) < sl + 1 or len(val_data) < sl + 1:
                # 若資料不足，給個 NaN 或 直接跳過
                candidate_to_patient_rmse[sl].append(np.nan)
                continue

            X_sub_train, y_sub_train = create_sequences(sub_train_data, sl)
            X_val,       y_val       = create_sequences(val_data, sl)

            train_loader = create_dataloader(X_sub_train, y_sub_train, batch_size=batch_size)
            val_loader   = create_dataloader(X_val,       y_val,       batch_size=batch_size, shuffle=False)

            # (3) 建立並訓練模型
            model = LSTMModel(input_dim=1, hidden_dim=Best_LU, fc_dim=Best_DU).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # Early Stopping 相關變數
            best_val_loss = float('inf')
            patience_counter = 0
            for epoch in range(num_epochs):
                _ = train_one_epoch(model, train_loader, criterion, optimizer, device)

                # 計算驗證 RMSE
                val_rmse = evaluate_model(model, val_loader, device)

                # 檢查 Early Stopping 條件
                if val_rmse < best_val_loss:
                    best_val_loss = val_rmse
                    patience_counter = 0  # 重置 patience_counter
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping triggered for SL={sl} at epoch {epoch + 1}")
                    break

            # (4) 計算驗證RMSE
            val_rmse = evaluate_model(model, val_loader, device)
            rmse_list.append(val_rmse)

            # 儲存到 candidate_to_patient_rmse
            candidate_to_patient_rmse[sl].append(val_rmse)

        # 計算此 SL 的平均RMSE
        if len(rmse_list) == 0:
            avg_val_rmse = float('inf')
        else:
            avg_val_rmse = np.mean(rmse_list)

        print(f"SL={sl}, Average Val RMSE={avg_val_rmse:.4f}")
        if avg_val_rmse < best_rmse:
            best_rmse = avg_val_rmse
            best_SL = sl

    return best_SL, best_rmse, candidate_to_patient_rmse

# (4) 最終訓練與測試  

def final_train_and_test_for_each_patient(patient_list,
                                          best_LU, best_DU, best_SL,
                                          horizon=6,
                                          device='cpu',
                                          num_epochs=30,
                                          batch_size=16,
                                          patience=20,
                                          min_delta=0.001,
                                          lr=0.001):
    """
    用找到的 (best_LU, best_DU, best_SL) + horizon，
    逐位病患進行最終訓練(在train_data=66%)並在 test_data=34% 上評估 RMSE。
    """
    patient_results = []

    for idx, data in enumerate(patient_list):
        n = len(data)
        train_size = int(0.66 * n)
        train_data = data[:train_size]
        test_data = data[train_size:]

        # 若資料量不足以生成序列
        if len(train_data) < (best_SL + horizon) or len(test_data) < (best_SL + horizon):
            print(f"Patient {idx}: 資料不足以產生 (SL+Horizon) = {best_SL + horizon}，跳過...")
            continue

        # 建立序列
        X_train, y_train = create_sequences(train_data, best_SL, horizon=horizon)
        X_test, y_test = create_sequences(test_data, best_SL, horizon=horizon)

        train_loader = create_dataloader(X_train, y_train, batch_size=batch_size)
        test_loader = create_dataloader(X_test, y_test, batch_size=batch_size, shuffle=False)

        # 建模
        model = LSTMModel(input_dim=1, hidden_dim=best_LU, fc_dim=best_DU).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # 訓練
        import time
        start_time = time.time()

        best_loss = float('inf')
        no_improvement = 0

        for epoch in range(num_epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            
            # 評估驗證損失
            val_rmse = evaluate_model(model, test_loader, device)  # 使用驗證集
            val_loss = val_rmse ** 2  # 將 RMSE 轉換為 MSE

            # 判斷是否有改進
            if best_loss - val_loss > min_delta:
                best_loss = val_loss
                no_improvement = 0  # 重置等待次數
            else:
                no_improvement += 1

            # print(f"[Patient {idx}] Epoch {epoch + 1}/{num_epochs}, Train Loss = {train_loss:.4f}, "
            #       f"Val RMSE = {val_rmse:.4f}, No Improvement = {no_improvement}")

            # 判斷是否早停
            if no_improvement >= patience:
                print(f"[Patient {idx}] Early stopping at epoch {epoch + 1}")
                break

        end_time = time.time()

        # 測試
        test_rmse = evaluate_model(model, test_loader, device)
        train_time = end_time - start_time

        print(f"[Patient {idx}] Test RMSE = {test_rmse:.4f}, TrainingTime = {train_time:.2f} s")

        patient_results.append({
            'patient_index': idx,
            'test_rmse': test_rmse,
            'train_time': train_time
        })

    return patient_results

def plot_rmse_for_candidates(candidate_to_patient_rmse, step_name="LU", save_path=None):
    """
    將每個候選參數對應到各病患的RMSE畫在同一張圖上。
    candidate_to_patient_rmse: dict, key=候選參數, value=病患RMSE list
    step_name: 例如 "LU", "DU", "SL", 用來在圖的標題/圖例上區分
    save_path: 若指定路徑，則 plt.savefig(save_path)
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))

    # x 軸: 病患索引
    # 一條線: 一個參數
    # y 值: RMSE

    candidate_keys = list(candidate_to_patient_rmse.keys())

    # 取得最多病患的數量（假設 list 的長度都相同，若有 nan 也允許）
    # 或直接用 len(patient_list)，視您如何保留資訊
    # 這裡以取 candidate_to_patient_rmse 第一個 key的 list 做例子:
    first_key = candidate_keys[0]
    num_patients = len(candidate_to_patient_rmse[first_key])

    x_vals = np.arange(num_patients)

    for param in candidate_keys:
        rmse_list = candidate_to_patient_rmse[param]
        # 畫出折線
        plt.plot(x_vals, rmse_list, marker='o', label=f"{step_name}-{param}")

    plt.xlabel("Patient Index")
    plt.ylabel("RMSE (mg/dL)")
    plt.title(f"{step_name} Grid Search: RMSE across Patients")
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

    plt.close()

# (5) main

if __name__ == "__main__":
    horizon = 6  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 先載入資料
    files = count_rows_in_csv_files("../../real_data/conti")
    # 取前 10 位病患
    top10_files = files[:10]
    patient_list = []
    for file in top10_files:
        file_path = os.path.join("../../real_data/conti", list(file.keys())[0])
        all_data = load_single_patient_data(file_path)
        patient_list.append(all_data)
    
    # 設定候選的hyperparameters
    LU_candidates = [5, 10, 20, 30, 40, 50, 60, 70]
    DU_candidates = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    SL_candidates = [5, 10, 15, 20, 50, 100]

    # 定義 5 組隨機種子
    seeds = [42, 7, 123, 2025, 0]
    results_across_seeds = []

    # 對每一組隨機種子進行實驗
    for seed in seeds:
        print(f"\n\n========== 使用隨機種子 {seed} 進行實驗 ==========")
        # 固定隨機種子
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # ---- Step 1: 搜尋最佳 LU ----
        best_LU, best_rmse, candidate_to_patient_rmse = search_best_LU_across_patients(
            patient_list, LU_candidates, 60, 20,
            device=device, num_epochs=300, batch_size=16, val_ratio=0.35, patience=30
        )
        print(f"Step 1: Best LU = {best_LU}, Val RMSE = {best_rmse:.4f}")
        plot_rmse_for_candidates(candidate_to_patient_rmse, 
                                 step_name=f"LU_seed{seed}", 
                                 save_path=f"../result/30m/LU_search_seed{seed}.png")  

        # ---- Step 2: 搜尋最佳 DU ----
        best_DU, best_rmse, candidate_to_patient_rmse = search_best_DU_across_patients(
            patient_list, DU_candidates, best_LU, 20,
            device=device, num_epochs=300, batch_size=16, val_ratio=0.35, patience=30
        )
        print(f"Step 2: Best DU = {best_DU}, Val RMSE = {best_rmse:.4f}")
        plot_rmse_for_candidates(candidate_to_patient_rmse, 
                                 step_name=f"DU_seed{seed}", 
                                 save_path=f"../result/30m/DU_search_seed{seed}.png")

        # ---- Step 3: 搜尋最佳 SL ----
        best_SL, best_rmse, candidate_to_patient_rmse = search_best_SL_across_patients(
            patient_list, SL_candidates, best_LU, best_DU,
            device=device, num_epochs=300, batch_size=16, val_ratio=0.35, patience=30
        )
        print(f"Step 3: Best SL = {best_SL}, Val RMSE = {best_rmse:.4f}")
        plot_rmse_for_candidates(candidate_to_patient_rmse, 
                                 step_name=f"SL_seed{seed}", 
                                 save_path=f"../result/30m/SL_search_seed{seed}.png")

        print("\n=== Step 1,2,3 搜尋完畢 ===")
        print(f"平均驗證 RMSE = {best_rmse:.4f}")

        # ---- 最終訓練與測試 ----
        best_params = (best_LU, best_DU, best_SL)
        print("\n=== 最終訓練與測試 ===")
        print(f"最佳 (LU, DU, SL) = {best_params}")
        print("進行最終訓練與測試...")
        test_rmse_list = []
        result = final_train_and_test_for_each_patient(
            patient_list, best_LU, best_DU, best_SL,
            device=device, num_epochs=300, batch_size=16, patience=30
        )
        for r in result:
            print(f"Patient {r['patient_index']}: Test RMSE = {r['test_rmse']:.4f}, Training Time = {r['train_time']:.2f} s")
            test_rmse_list.append(r['test_rmse'])
        avg_rmse = np.mean(test_rmse_list)
        print(f"平均 Test RMSE = {avg_rmse:.4f}")

        # 保存此組種子的實驗結果
        results_across_seeds.append({
            "seed": seed,
            "best_params": best_params,
            "avg_rmse": avg_rmse,
            "details": result
        })

    # ---- 全部種子的結果摘要 ----
    print("\n\n========== 全部種子實驗結果 ==========")
    for res in results_across_seeds:
        print(f"Seed {res['seed']}: Best Params = {res['best_params']}, Avg Test RMSE = {res['avg_rmse']:.4f}")
    overall_avg_rmse = np.mean([res["avg_rmse"] for res in results_across_seeds])
    print(f"所有種子的平均 Test RMSE = {overall_avg_rmse:.4f}")
    # 存至txt檔
    with open("../result/30m/my_hyperparam/overall_results.txt", "w") as f:
        f.write("Seed, Best Params, Avg Test RMSE\n")
        for res in results_across_seeds:
            f.write(f"{res['seed']}, {res['best_params']}, {res['avg_rmse']:.4f}\n")
        f.write(f"Overall Avg Test RMSE = {overall_avg_rmse:.4f}\n")



