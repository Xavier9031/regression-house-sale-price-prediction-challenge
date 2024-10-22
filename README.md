# regression-house-sale-price-prediction-challenge-NYCU312707024

## Description

這個項目的目標是開發一個回歸模型來預測房價。我們使用 `train.csv` 和 `valid.csv` 資料集來訓練模型，這些數據集包含 id、price 和 19 種房屋參數的交易記錄。然後，我們使用訓練好的模型來預測 `test.csv` 中的房價，並將結果提交至系統以評估模型的 Mean Absolute Error (MAE) 表現。

## Getting Started

### Dependencies

Python 3.10.13

安裝相依套件：
```bash
pip install -r requirment.txt
```

## 文件結構

- `main.py`: 主腳本,用來加载數據,預處理,調參,訓練模型,以及生成提交文件
- `models.py`: 包含模型訓練和評估的函數
- `data/`: 數據集目錄,包含`train.csv`、`valid.csv` 和 `test.csv`

## 使用方法

主腳本`main.py`負責數據加載、預處理、模型訓練、評估和提交預測結果。

運行整個流程:

```
python main.py
```

這將會:

- 加載`train.csv`、`valid.csv`和`test.csv`
- 預處理數據(特徵工程、特徵選擇等)
- 使用Optuna調參
- 訓練最終的XGBoost模型
- 在訓練集和驗證集上進行評估
- 生成提交文件`submission.csv`,裡面包含預測結果

可以自定義提交文件名:

```
python main.py --submission_path my_submission.csv
```


### Train the Model

使用以下命令訓練模型：
```bash
sh train.sh
```

### Test the Model

使用以下命令測試模型並生成提交文件：
```bash
sh test.sh
```

## 數據預處理

- 刪除无用特徵如ID和郵編
- 生成特徵如`is_basement`和`house_year`
- 基於相關性、方差和KBest進行特徵選擇
- 將訓練數據分割為訓練集(80%)和驗證集(20%)

詳情見`models.py`中的`data_preprocessing()`。

## 模型訓練

- 使用XGBoost回歸模型
- 使用Optuna調參
- 在訓練集上訓練,在驗證集上評估
- 畫出學習曲線
- 保存最終模型

詳情見`main.py`中的`train()`和`tune_hyperparameters()`。

## 評估

- 在訓練集和驗證集上計算RMSE

詳情見`models.py`中的`validate()`。

## 預測

- 將測試集數據輸入訓練好的模型
- 將預測結果寫入提交文件

詳情見`models.py`中的`get_submission()`。


## Analysis and Discussion

### Results Analysis

![image](https://github.com/Machine-Learning-NYCU/regression-house-sale-price-prediction-challenge-NYCU312707024/blob/main/learning_curve.png)

MAE of training data:  16219.406195773427

MAE of validation data:  67202.47720794778

### Error Analysis

模型在訓練數據上的 MAE 為 16219.41，而在驗證數據上的 MAE 升高到 67202.48。這個差異較大的誤差提示模型可能過度擬合了訓練數據。過度擬合通常表現為模型在訓練數據上表現出色，但在未見過的數據（如驗證數據）上表現不佳。可能的原因包括：

- **模型複雜度過高**：模型可能太複雜，捕捉到了訓練數據中的噪聲，而非其真實的分布。
- **數據質量問題**：數據可能存在一些異常值或錯誤的標籤，這些都會對模型的學習造成影響。
- **特征選擇**：模型所使用的特征可能不是最優的，可能包括了一些不相關或噪聲特征。

### Improvements

為了改善模型的性能和泛化能力，我們可以採取以下幾個策略：

- **正則化**：使用正則化技術，比如 L1 或 L2 正則化，來減小模型的複雜度並防止過度擬合。
- **交叉驗證**：使用交叉驗證來更可靠地估計模型的性能並選擇最佳的超參數。
- **特征工程**：進一步優化特征選擇和特征處理，例如進行更多的特征組合、使用不同的編碼方法等。
- **數據預處理**：對數據進行進一步的清理和預處理，例如處理異常值、填補缺失值等。
- **使用不同的模型或模型融合**：嘗試使用其他的回歸模型或者將多個模型進行融合，如使用集成學習方法（例如 Bagging 或 Boosting）來提高預測性能。

## Submission

提交文件應該包含一個標頭並且具有以下格式：
    
```
id,price
2,0
5,0
6,0
etc.
```

## Evaluation

提交的預測結果將基於與觀測到的銷售價格之間的平均絕對誤差（MAE）進行評估。

## 個人心得

這個項目提供了一個實際的挑戰，即使用機器學習模型來預測房價。雖然我們已經得到了一些初步的結果，但從模型在驗證數據上的性能來看，還有很多可以優化和改進的地方。這個過程不僅讓我更深入地理解了回歸模型和機器學習的一些挑戰（例如過度擬合、特征選擇等），也讓我意識到實際應用中模型的部署和優化是一個持續迭代和優化的過程。我期待在未來的學習和項目中繼續深化我的理解和技能。
