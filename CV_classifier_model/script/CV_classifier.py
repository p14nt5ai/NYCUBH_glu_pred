import csv
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def segment_data_by_value(file_path, boundary_value='0.800'):
    # 回傳 segments (list of rows)。
    
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
                # 開啟新 segment 從當前這一行 (含 boundary_value) 開始
                current_segment.append(row)
            else:
                # 若非 boundary_value 的行，就持續加入 current_segment
                current_segment.append(row)

        # 迴圈結束後，如果 current_segment 裡還有資料，也要加入
        if current_segment:
            segments.append(current_segment)

    return segments



def extract_features(segment, n_subsegments=5):
    """
    對單一 segment 多段電流值取樣。
    1. 將該 segment 裡的 (電位, 電流) 轉為 float
    2. 找出最小電位 minV、最大電位 maxV
    3. 將 (minV, maxV) 等分成 n_subsegments 段，計算每段平均電流
    
    回傳: [feature_1, feature_2, ..., feature_n_subsegments]
    """
    # 將 segment 轉成 np.array，濾除無效行
    potential = []
    current = []
    for row in segment:
        try:
            p = float(row[0])
            i = float(row[1])
            potential.append(p)
            current.append(i)
        except ValueError:
            continue

    potential = np.array(potential)
    current = np.array(current)

    if len(potential) == 0:
        return [0]*n_subsegments

    min_v, max_v = np.min(potential), np.max(potential)
    
    # 防呆
    if min_v == max_v:
        
        return [np.mean(current)]*n_subsegments


    # 拿到 n_subsegments+1 個切割點
    sub_range_edges = np.linspace(min_v, max_v, n_subsegments+1)

    features = []

    for idx in range(n_subsegments):
        # 取第 idx 段的電位範圍 [sub_range_edges[idx], sub_range_edges[idx+1])
        v_start = sub_range_edges[idx]
        v_end = sub_range_edges[idx+1]

        # 找到屬於這段電位範圍內的 current
        mask = (potential >= v_start) & (potential < v_end)
        currents_in_range = current[mask]

        if len(currents_in_range) > 0:
            avg_i = np.mean(currents_in_range)  # 計算該段平均電流
        else:
            avg_i = 0  # 如果這段沒有資料點，就給 0 或其他預設值

        features.append(avg_i)

    return features



def main():
    """
    1. 讀 csv 檔
    2. 每個檔案取 segment
    3. 每個 segment 取特徵後，指定其 Label (濃度)
    4. 收集所有 (X, y) 資料後，使用機器學習進行分類
    """
    data_dir = "../../CV_data"  
    boundary_value = '0.800'
    n_subsegments = 5

    X = []
    y = []

    # recusively read all files in the directory

    for element in os.listdir(data_dir):
        if not os.path.isdir(os.path.join(data_dir, element)):
            continue
        for dirnames in os.listdir(os.path.join(data_dir, element)):
            for filename in os.listdir(os.path.join(data_dir, element, dirnames)):
                
                if filename.endswith(".csv"):
                    file_path = os.path.join(data_dir, element, dirnames, filename)

                    segments = segment_data_by_value(file_path, boundary_value)

                    # 砍了頭兩個 segment
                    segments = segments[2:]

                    
                    dict_label = {"0mM.csv": 0, "0.5mM.csv": 0.5, "1mM.csv": 1, "2mM.csv": 2, "4mM.csv": 4, "8mM.csv": 8, "10mM.csv": 10, "15mM.csv": 15, "20mM.csv": 20, "25mM.csv": 25,
                                "0mm.csv": 0, "0.5mm.csv": 0.5, "1mm.csv": 1, "2mm.csv": 2, "4mm.csv": 4, "8mm.csv": 8, "10mm.csv": 10, "15mm.csv": 15, "20mm.csv": 20, "25mm.csv": 25}
                    
                    
                    
                    label_for_this_file = str(dict_label[(filename.split("_")[2])])

                    # 針對每個 segment 萃取特徵
                    for seg in segments:
                        features = extract_features(seg, n_subsegments=n_subsegments)
                        X.append(features)
                        y.append(label_for_this_file)
                    
                

    X = np.array(X)
    y = np.array(y)

    valid_mask = (y != -1)
    X = X[valid_mask]
    y = y[valid_mask]

    # Train / Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # RandomForest 分類器
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # 5. 測試並輸出結果
    y_pred = clf.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # 6. save result and report
    result_path = "../result/discrete/CV_classifier_result.csv"
    with open(result_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["True", "Predicted"])
        for true, pred in zip(y_test, y_pred):
            writer.writerow([true, pred])
    print(f"Result saved to {result_path}")

    report_path = "../result/discrete/CV_classifier_report.txt"
    with open(report_path, 'w') as f:
        f.write(classification_report(y_test, y_pred))
    print(f"Classification report saved to {report_path}")
    

    # 7. 儲存模型
    model_path = "../result/discrete/CV_classifier_model.pkl"
    import joblib
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
