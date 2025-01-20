# NYCUBH_glu_pred
2025/01/20

## model
### script
1. train_by_torch_real_5m.py :  
預測5分鐘後血糖，資料同復現論文  
使用有限的參數做 grid_search  
在 Training set 找到最適合的(LU, DU, SL)後，在 Testing set 上測試

3. train_by_torch_real_30m.py :  
預測30分鐘後血糖，資料來源同復現論文  
一樣使用有限的參數做 grid_search
使用 grid_search 在 Training set 找到最適合的(LU, DU, SL)後，在 Testing set 上測試

### result
存放實驗結果
## real_data
1. tblADataCGMS.csv:  
原始資料

2. conti  
存放預處理後的CGM的csv檔

3. some_graph  
展示一些圖像化的資料

4. conti_85  
/some_graph/e47674de-34a9-4c10-a910-64ae85580c57.png 之原始資料
