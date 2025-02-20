# NYCUBH_glu_pred
2025/01/20




## model
### script

- **主要流程**
   
  + **資料準備**  
      從檔案中讀取資料，並將其存為時序資料列表。  
  + **參數搜尋**  
      按順序進行 LU、DU、SL 搜尋。  
      每步驟紀錄 RMSE 並繪圖。  
  + **最終測試**  
      使用最佳參數組合對每位病患資料進行訓練與測試。  
      計算平均測試 RMSE。  

1. lstm_bg_search_5m.py :
預測5分鐘後血糖，資料同復現論文  
使用有限的參數做 grid_search  
在 Training set 找到最適合的 ( LU, DU, SL ) 後，在 Testing set 上測試

2. lstm_bg_search_30m.py :  
預測30分鐘後血糖，資料來源同復現論文(與5m使用資料不同)  
一樣使用有限的參數做 grid_search
使用 grid_search 在 Training set 找到最適合的 ( LU, DU, SL ) 後，在 Testing set 上測試

### result
存放實驗結果

### preprocessing  
  + extract_conti_data.py :  
切割tblADataCGMS.csv成可用的連續片段檔案

## CNNLSTM_model
流程與 model 相似，但在原有模型上加上了卷積層

## CV_model  
### script  
  + peak_analysis_lstm.py :  
  挑選 CV 圖中的峰值，並進行峰值預測  
  
## real_data
1. tblADataCGMS.csv:  
[原始資料](https://public.jaeb.org/direcnet/stdy/155)


2. conti  
存放預處理後的 CGM 的 csv 檔

3. some_graph  
展示一些圖像化的資料

## CV_classifier_model  
從CV資料轉成葡萄糖濃度


