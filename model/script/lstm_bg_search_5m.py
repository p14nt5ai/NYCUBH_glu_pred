from itertools import product
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from math import sqrt

# (1) 讀取與前處理函式 

def load_single_patient_data(file_path):
    # 讀取單一病患的 CSV 時序資料 
    df = pd.read_csv(file_path)
    sensor_values = df["SensorGLU"].dropna().values
    return sensor_values

def create_sequences(data, seq_length):
    """
    給定一維資料 data 和序列長度 seq_length，
    產生 (X, y) 形式的時序資料。
    X shape: [batch_size, seq_length, 1]
    y shape: [batch_size, 1]
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    X = np.array(xs).reshape(-1, seq_length, 1)
    y = np.array(ys).reshape(-1, 1)
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
        self.dropout = nn.Dropout(p=0) # dropout
        self.fc1 = nn.Linear(hidden_dim, fc_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_dim, 1)

    def forward(self, x):
        # x shape: [batch_size, seq_length, input_dim]
        lstm_out, (h_n, c_n) = self.lstm(x)
        # 取最後一個時間的輸出
        last_time_step = lstm_out[:, -1, :]  # shape: [batch_size, hidden_dim]
        x = self.dropout(last_time_step) 
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
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
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch)
            preds.append(outputs.cpu().numpy())
            trues.append(y_batch.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    rmse_val = np.sqrt(((preds - trues) ** 2).mean())
    return float(rmse_val)


# (3) Step 1,2,3 搜尋 

import matplotlib.pyplot as plt
import numpy as np

def search_best_LU_across_patients(patient_list,
                                   candidate_LU,
                                   fixed_DU,
                                   fixed_SL,
                                   device='cpu',
                                   num_epochs=10,
                                   batch_size=16,
                                   val_ratio=0.2,
                                   patience=20):
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
            optimizer = optim.Adam(model.parameters(), lr=0.001)
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
                                   patience=20):
    """
    Step 2: 固定 LU, SL，搜尋 DU。
    會回傳:
      best_DU
      best_rmse (平均驗證 RMSE)
      candidate_to_patient_rmse: dict，key = DU, value = list of rmse per patient
    """

    best_DU = None
    best_rmse = float('inf')

    candidate_to_patient_rmse = {du: [] for du in candidate_DU}

    num_patients = len(patient_list)

    for du in candidate_DU:
        rmse_list = []  

        for idx, data in enumerate(patient_list):
            
            n = len(data)
            train_size = int(0.66 * n)
            train_data = data[:train_size]
            val_size = int(val_ratio * len(train_data))
            sub_train_data = train_data[:-val_size]
            val_data       = train_data[-val_size:]

            if len(sub_train_data) < fixed_SL + 1 or len(val_data) < fixed_SL + 1:
                candidate_to_patient_rmse[du].append(np.nan)
                continue

            X_sub_train, y_sub_train = create_sequences(sub_train_data, fixed_SL)
            X_val,       y_val       = create_sequences(val_data, fixed_SL)

            train_loader = create_dataloader(X_sub_train, y_sub_train, batch_size=batch_size)
            val_loader   = create_dataloader(X_val,       y_val,       batch_size=batch_size, shuffle=False)

            model = LSTMModel(input_dim=1, hidden_dim=Best_LU, fc_dim=du).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            best_val_loss = float('inf')
            patience_counter = 0

            for epoch in range(num_epochs):
                _ = train_one_epoch(model, train_loader, criterion, optimizer, device)

                val_rmse = evaluate_model(model, val_loader, device)

                if val_rmse < best_val_loss:
                    best_val_loss = val_rmse
                    patience_counter = 0 
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping triggered for DU={du} at epoch {epoch + 1}")
                    break

            val_rmse = evaluate_model(model, val_loader, device)
            rmse_list.append(val_rmse)

            candidate_to_patient_rmse[du].append(val_rmse)

        if len(rmse_list) == 0:
            avg_val_rmse = float('inf')
        else:
            avg_val_rmse = np.mean(rmse_list)

        print(f"DU={du}, Average Val RMSE={avg_val_rmse:.4f}")
        if avg_val_rmse < best_rmse:
            best_rmse = avg_val_rmse
            best_dU = du

    return best_dU, best_rmse, candidate_to_patient_rmse
    
def search_best_SL_across_patients(patient_list,
                                   candidate_SL,
                                   Best_LU,
                                   Best_DU,
                                   device='cpu',
                                   num_epochs=10,
                                   batch_size=16,
                                   val_ratio=0.2,
                                   patience=20):
    """
    Step 3: 固定 DU, LU，搜尋 SL。
    會回傳:
      best_SL
      best_rmse (平均驗證 RMSE)
      candidate_to_patient_rmse: dict，key = SL, value = list of rmse per patient
    """

    best_SL = None
    best_rmse = float('inf')

    candidate_to_patient_rmse = {sl: [] for sl in candidate_SL}

    num_patients = len(patient_list)

    for sl in candidate_SL:
        rmse_list = []  

        for idx, data in enumerate(patient_list):
            n = len(data)
            train_size = int(0.66 * n)
            train_data = data[:train_size]
            val_size = int(val_ratio * len(train_data))
            sub_train_data = train_data[:-val_size]
            val_data       = train_data[-val_size:]

            if len(sub_train_data) < sl + 1 or len(val_data) < sl + 1:
                candidate_to_patient_rmse[sl].append(np.nan)
                continue

            X_sub_train, y_sub_train = create_sequences(sub_train_data, sl)
            X_val,       y_val       = create_sequences(val_data, sl)

            train_loader = create_dataloader(X_sub_train, y_sub_train, batch_size=batch_size)
            val_loader   = create_dataloader(X_val,       y_val,       batch_size=batch_size, shuffle=False)

            model = LSTMModel(input_dim=1, hidden_dim=Best_LU, fc_dim=Best_DU).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            best_val_loss = float('inf')
            patience_counter = 0
            for epoch in range(num_epochs):
                _ = train_one_epoch(model, train_loader, criterion, optimizer, device)

                val_rmse = evaluate_model(model, val_loader, device)

                if val_rmse < best_val_loss:
                    best_val_loss = val_rmse
                    patience_counter = 0 
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping triggered for SL={sl} at epoch {epoch + 1}")
                    break

            val_rmse = evaluate_model(model, val_loader, device)
            rmse_list.append(val_rmse)

            candidate_to_patient_rmse[sl].append(val_rmse)

        if len(rmse_list) == 0:
            avg_val_rmse = float('inf')
        else:
            avg_val_rmse = np.mean(rmse_list)

        print(f"SL={sl}, Average Val RMSE={avg_val_rmse:.4f}")
        if avg_val_rmse < best_rmse:
            best_rmse = avg_val_rmse
            best_SL = sl

    return best_SL, best_rmse, candidate_to_patient_rmse


# (4) 最終訓練與測試、繪圖

def final_train_and_test_for_each_patient(
    patient_list,
    best_LU, best_DU, best_SL,
    horizon=1,
    device='cpu',
    num_epochs=30,
    batch_size=16,
    patience=20,
    min_delta=0.001,
    lr=0.001,
    val_ratio_within_train=0.2  # e.g. 在train_data中拿20%做validation
):
    """
     - 66% 全部叫 train_data, 34% 叫 test_data
     - train_data中再切一小段做 validation
     - 最後只用 test_data 做最終測試
    """

    patient_results = []

    for idx, data in enumerate(patient_list):
        n = len(data)
        train_size = int(0.66 * n)
        train_data = data[:train_size]
        test_data  = data[train_size:]

        if len(train_data) < (best_SL + horizon) or len(test_data) < (best_SL + horizon):
            print(f"Patient {idx}: 資料不足以產生 (SL+Horizon) = {best_SL + horizon}，跳過...")
            continue

        # --- Step A: 在 train_data 內再切出一小段做 validation
        val_size = int(val_ratio_within_train * len(train_data))
        # 確保 val_size 不會太小而無法產生序列
        if val_size < (best_SL + horizon):
            print(f"Patient {idx}: validation set 太小，請調整 val_ratio_within_train.")
            continue

        sub_train_data = train_data[:-val_size]  # 前面做真正的train
        val_data       = train_data[-val_size:]  # 後面做validation

        # --- Step B: 產生 X, y 序列
        X_sub_train, y_sub_train = create_sequences(sub_train_data, best_SL)
        X_val,       y_val       = create_sequences(val_data,       best_SL)
        X_test,      y_test      = create_sequences(test_data,      best_SL)

        train_loader = create_dataloader(X_sub_train, y_sub_train, batch_size=batch_size)
        val_loader   = create_dataloader(X_val,       y_val,       batch_size=batch_size, shuffle=False)
        test_loader  = create_dataloader(X_test,      y_test,      batch_size=batch_size, shuffle=False)

        # --- Step C: 建模與訓練
        model = LSTMModel(input_dim=1, hidden_dim=best_LU, fc_dim=best_DU).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_val_loss = float('inf')
        no_improvement = 0

        import time
        start_time = time.time()

        for epoch in range(num_epochs):
            # (1) 在 sub_train 上做一個 epoch 的參數更新
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

            # (2) 在 val set 上測試 RMSE，做 Early Stopping 判斷
            val_rmse = evaluate_model(model, val_loader, device)
            val_loss = val_rmse ** 2

            if (best_val_loss - val_loss) > min_delta:
                best_val_loss = val_loss
                no_improvement = 0
            else:
                no_improvement += 1
            
            if no_improvement >= patience:
                print(f"[Patient {idx}] Early stopping at epoch {epoch+1}")
                break
        
        end_time = time.time()

        # --- Step D: 最終用 test_loader 做測試
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
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))

    # x 軸: 病患索引
    # 一條線: 一個參數
    # y 值: RMSE

    candidate_keys = list(candidate_to_patient_rmse.keys())


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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [42, 7, 123, 2025, 0]
    files = count_rows_in_csv_files("../../real_data/conti")

    top10_files = [
        "pt_1_segment_2.csv", "pt_2_segment_1.csv", "pt_3_segment_1.csv",
        "pt_4_segment_6.csv", "pt_5_segment_1.csv", "pt_6_segment_2.csv",
        "pt_7_segment_2.csv", "pt_8_segment_2.csv", "pt_9_segment_1.csv",
        "pt_10_segment_1.csv"
    ]
    
    patient_list = []
    for file in top10_files:
        file_path = os.path.join("../../real_data/conti", file)
        all_data = load_single_patient_data(file_path)
        patient_list.append(all_data)

    # 定義候選超參數
    LU_candidates = [5, 10, 20, 30, 40, 50, 60, 70]
    DU_candidates = [10, 20, 30, 40, 50]
    SL_candidates = [5, 10, 15, 20]

    # 儲存每個種子的實驗結果
    results_across_seeds = []

    # 逐一對每個種子進行實驗
    for seed in seeds:
        print(f"\n\n========== 使用隨機種子 {seed} 進行實驗 ==========")

        # 固定隨機種子
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        print("=== Step 1,2,3 搜尋 ===")

        # ---- Step 1: 搜尋最佳 LU ----
        best_LU, best_rmse_LU, candidate_to_patient_rmse = search_best_LU_across_patients(
            patient_list, LU_candidates,
            fixed_DU=30,  
            fixed_SL=10,
            device=device, num_epochs=300, batch_size=16, val_ratio=0.2, patience=50
        )
        print(f"Step 1: Best LU = {best_LU}, Val RMSE = {best_rmse_LU:.4f}")
        plot_rmse_for_candidates(
            candidate_to_patient_rmse,
            step_name=f"LU_seed{seed}",
            save_path=f"../result/5m/my_hyperparam/LU_search_seed{seed}.png"
        )

        # ---- Step 2: 搜尋最佳 DU ----
        best_DU, best_rmse_DU, candidate_to_patient_rmse = search_best_DU_across_patients(
            patient_list, DU_candidates,
            Best_LU=best_LU, fixed_SL = 10,
            device=device, num_epochs=300, batch_size=16, val_ratio=0.2, patience=50
        )
        print(f"Step 2: Best DU = {best_DU}, Val RMSE = {best_rmse_DU:.4f}")
        plot_rmse_for_candidates(
            candidate_to_patient_rmse,
            step_name=f"DU_seed{seed}",
            save_path=f"../result/5m/my_hyperparam/DU_search_seed{seed}.png"
        )

        # ---- Step 3: 搜尋最佳 SL ----
        best_SL, best_rmse_SL, candidate_to_patient_rmse = search_best_SL_across_patients(
            patient_list, SL_candidates,
            Best_LU=best_LU, Best_DU=best_DU,
            device=device, num_epochs=300, batch_size=16, val_ratio=0.2, patience=50
        )
        print(f"Step 3: Best SL = {best_SL}, Val RMSE = {best_rmse_SL:.4f}")
        plot_rmse_for_candidates(
            candidate_to_patient_rmse,
            step_name=f"SL_seed{seed}",
            save_path=f"../result/5m/my_hyperparam/SL_search_seed{seed}.png"
        )

        # 取得最終搜尋到的參數 (LU, DU, SL) 與最後 RMSE
        best_rmse = best_rmse_SL  # 此處可視為第三步的最佳 RMSE
        print("\n=== Step 1,2,3 搜尋完畢 ===")
        print(f"Seed {seed} 最佳搜尋結果 => (LU, DU, SL) = ({best_LU}, {best_DU}, {best_SL}), Val RMSE = {best_rmse:.4f}")

        # ---- 最終訓練與測試 ----
        print("\n=== 最終訓練與測試 ===")
        print(f"最佳 (LU, DU, SL) = ({best_LU}, {best_DU}, {best_SL})")
        print("進行最終訓練與測試...")

        result = final_train_and_test_for_each_patient(
            patient_list,
            best_LU=best_LU,
            best_DU=best_DU,
            best_SL=best_SL,
            device=device, num_epochs=400, batch_size=16, patience=80
        )

        test_rmse_list = []
        for r in result:
            print(f"Patient {r['patient_index']}: Test RMSE = {r['test_rmse']:.4f}, Training Time = {r['train_time']:.2f} s")
            test_rmse_list.append(r['test_rmse'])

        # 計算平均測試 RMSE
        avg_test_rmse = np.mean(test_rmse_list)
        print(f"平均 Test RMSE = {avg_test_rmse:.4f}")

        # 收集當前種子的結果
        results_across_seeds.append({
            "seed": seed,
            "best_params": (best_LU, best_DU, best_SL),
            "val_rmse": best_rmse,
            "test_rmse_list": test_rmse_list,
            "avg_test_rmse": avg_test_rmse
        })

    # ---- 最後彙整所有種子的實驗結果 ----
    print("\n\n========== 全部種子實驗結果彙整 ==========")
    for res in results_across_seeds:
        print(
            f"Seed {res['seed']}: "
            f"Best Params = {res['best_params']}, "
            f"Val RMSE (Step3) = {res['val_rmse']:.4f}, "
            f"Avg Test RMSE = {res['avg_test_rmse']:.4f}"
        )

    # 平均測試 RMSE
    overall_avg_rmse = np.mean([res['avg_test_rmse'] for res in results_across_seeds])
    print(f"\n[總結] 所有種子的平均 Test RMSE = {overall_avg_rmse:.4f}")
    # 保存結果 in txt
    with open("../result/5m/my_hyperparam/hyperparam_search_result.txt", "w") as f:
        f.write(f"Overall Avg Test RMSE = {overall_avg_rmse:.4f}\n")
        for res in results_across_seeds:
            f.write(
                f"Seed {res['seed']}: "
                f"Best Params = {res['best_params']}, "
                f"Val RMSE (Step3) = {res['val_rmse']:.4f}, "
                f"Avg Test RMSE = {res['avg_test_rmse']:.4f}\n"
            )






    # Test single 
    print('(50, 30, 10) 之實驗')
    results_across_seeds503010 = []
    for seed in seeds:
        print(f"\n\n========== 使用隨機種子 {seed} 進行實驗 ==========")

        # 固定隨機種子
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        result = final_train_and_test_for_each_patient(
            patient_list,
            best_LU=50,
            best_DU=30,
            best_SL=10,
            device=device, num_epochs=400, batch_size=16, patience=80
        )

        test_rmse_list = []
        for r in result:
            print(f"Patient {r['patient_index']}: Test RMSE = {r['test_rmse']:.4f}, Training Time = {r['train_time']:.2f} s")
            test_rmse_list.append(r['test_rmse'])

        # 計算平均測試 RMSE
        avg_test_rmse = np.mean(test_rmse_list)
        print(f"平均 Test RMSE = {avg_test_rmse:.4f}")

        # 收集當前種子的結果
        results_across_seeds503010.append({
            "seed": seed,
            "best_params": (50, 30, 10),
            "val_rmse": None,
            "test_rmse_list": test_rmse_list,
            "avg_test_rmse": avg_test_rmse
        })
    # 平均測試 RMSE
    overall_avg_rmse = np.mean([res['avg_test_rmse'] for res in results_across_seeds503010])
    print(f"\n[總結] 所有種子的平均 Test RMSE = {overall_avg_rmse:.4f}")
    # 保存結果 in txt
    with open("../result/5m/classical_hyperparam/hyperparam_search_result.txt", "w") as f:
        f.write(f"Overall Avg Test RMSE = {overall_avg_rmse:.4f}\n")
        for res in results_across_seeds503010:
            f.write(
                f"Seed {res['seed']}: "
                f"Best Params = {res['best_params']}, "
                f"Avg Test RMSE = {res['avg_test_rmse']:.4f}\n"
            )





