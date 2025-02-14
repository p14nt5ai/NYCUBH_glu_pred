import csv
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

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
    
    if min_v == max_v:
        return [np.mean(current)]*n_subsegments

    sub_range_edges = np.linspace(min_v, max_v, n_subsegments+1)

    features = []

    for idx in range(n_subsegments):
        v_start = sub_range_edges[idx]
        v_end = sub_range_edges[idx+1]

        mask = (potential >= v_start) & (potential < v_end)
        currents_in_range = current[mask]

        if len(currents_in_range) > 0:
            avg_i = np.mean(currents_in_range)  
        else:
            avg_i = 0  

        features.append(avg_i)

    return features



def main():

    # 使用回歸進行訓練與預測
    
    data_dir = "../../CV_data"  
    boundary_value = '0.800'
    n_subsegments = 5

    X = []
    y = []

    for element in os.listdir(data_dir):
        if not os.path.isdir(os.path.join(data_dir, element)):
            continue
        for dirnames in os.listdir(os.path.join(data_dir, element)):
            for filename in os.listdir(os.path.join(data_dir, element, dirnames)):
                
                if filename.endswith(".csv"):
                    file_path = os.path.join(data_dir, element, dirnames, filename)

                    segments = segment_data_by_value(file_path, boundary_value)
                    # 砍了頭兩個 segment (依需求)
                    segments = segments[2:]

                    dict_label = {
                        "0mM.csv": 0, "0.5mM.csv": 0.5, "1mM.csv": 1, "2mM.csv": 2,
                        "4mM.csv": 4, "8mM.csv": 8, "10mM.csv": 10, "15mM.csv": 15,
                        "20mM.csv": 20, "25mM.csv": 25,
                        "0mm.csv": 0, "0.5mm.csv": 0.5, "1mm.csv": 1, "2mm.csv": 2,
                        "4mm.csv": 4, "8mm.csv": 8, "10mm.csv": 10, "15mm.csv": 15,
                        "20mm.csv": 20, "25mm.csv": 25
                    }

                    
                    float_label = dict_label[(filename.split("_")[2])]  # e.g. 0.5 (float)

                    for seg in segments:
                        features = extract_features(seg, n_subsegments=n_subsegments)
                        X.append(features)
                        y.append(float_label)
    
    X = np.array(X)
    y = np.array(y, dtype=float)  

   
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Regression Metrics:")
    print(f"MSE: {mse}")
    print(f"R^2: {r2}")

    result_path = "../result/continuous/CV_regressor_result.csv"
    with open(result_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["True", "Predicted"])
        for true_val, pred_val in zip(y_test, y_pred):
            writer.writerow([true_val, pred_val])
    print(f"Regression result saved to {result_path}")

    report_path = "../result/continuous/CV_regressor_report.txt"
    with open(report_path, 'w') as f:
        report_str = f"MSE: {mse}\nR^2: {r2}\n"
        f.write(report_str)
    print(f"Regression report saved to {report_path}")
    
    model_path = "../result/continuous/CV_regressor_model.pkl"
    import joblib
    joblib.dump(reg, model_path)
    print(f"Model saved to {model_path}")



if __name__ == "__main__":
    main()



