import pandas as pd
import os

def export_5min_or_same_time_segments_to_csv(input_csv, output_dir='segments_output'):
    """
    將每位病患 (PtID) 的量測資料中，相鄰兩筆資料的時間差是 5 分鐘「或」0 分鐘
    (表示同一時刻重複量測) 的資料當作同一段，並各自輸出成獨立 CSV 檔。
    """
    
    # 1. 讀取原始資料
    df = pd.read_csv(input_csv)
    
    # 2. 合併 ReadingDt 與 ReadingTm 成 datetime (依你實際格式調整)
    df['ReadingDatetime'] = pd.to_datetime(df['ReadingDt'] + ' ' + df['ReadingTm'])
    
    # 3. 依 PtID、RecID 排序
    df = df.sort_values(by=['PtID', 'RecID']).reset_index(drop=True)

    # 若沒有資料夾就新建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 4. 針對每位病患做分組
    for pt_id, group in df.groupby('PtID'):
        # 再依時間做一次排序(保險)
        group = group.sort_values(by='RecID').reset_index(drop=True)
        
        # 計算相鄰筆之間的時間差(分鐘)
        group['TimeDiff'] = group['ReadingDatetime'].diff().dt.total_seconds().div(60)

        segments = []
        current_segment = []

        for idx, row in group.iterrows():
            if idx == 0:
                # 第一筆一定是新片段的開始
                current_segment = [row]
            else:
                diff_mins = row['TimeDiff']
                if diff_mins == 5:
                    current_segment.append(row)
                else:
                    # 遇到不符合(0或5分鐘)就結束前一段，開新段
                    segments.append(pd.DataFrame(current_segment))
                    current_segment = [row]

        # for 迴圈結束後，把最後一個 segment 也加入
        if current_segment:
            segments.append(pd.DataFrame(current_segment))

        # 5. 輸出 segment
        for i, seg_df in enumerate(segments, start=1):
            # 如果你需要至少含多少筆才輸出，可以在這裡加判斷
            if len(seg_df) < 20: continue  # 範例：只輸出>=2筆的片段

            # 組出檔名 pt_{pt_id}_segment_{i}.csv
            filename = f"pt_{pt_id}_segment_{i}.csv"
            out_path = os.path.join(output_dir, filename)
            
            # 輸出 csv
            seg_df.to_csv(out_path, index=False)

if __name__ == "__main__":
    # 使用範例：假設原始檔叫 'raw_data.csv'
    # 輸出到 segments_output 資料夾
    export_5min_or_same_time_segments_to_csv("../../real_data/tblADataCGMS.csv", "../../real_data/conti")
