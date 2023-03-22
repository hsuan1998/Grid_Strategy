# Grid_Strategy
網格交易結合LSTM模型預測價格範圍
## 使用方式舉例  
### 讀入資料與總資金
```python
grid = GridStrategy(wealth = 1000000,
                    min_data_path = 'BTCUSDT-Minute-Trade.txt',
                    day_data_path = 'BTCUSDT-Day-Trade.txt')
```
### 訓練預測價格模型
```python
grid.train_lstm()
```
### 驗證集預測圖
```python
grid.plot_valid()
```
### 預測資料輸入
```python
grid.inputdata('input.txt')
```
### 產生價格範圍與格數
```python
grid.output()
```
### 交易策略
```python
grid.train_grid()
```
### 走勢圖
```python
grid.plot_Grid()
```
### 進出場圖
```python
grid.plot_candlestick()
```
### 績效表現
```python
grid.plot_performance()
```
Yearly Cumulative Return: 54.0%
Weekly Cumulative Return: 0.83%
Yearly Max Cumulative Return: 103%
MDD: 1.22%
Return on MDD: 0.68
Sharpe Ratio: 4.9
Trade Volume: 10
### 交易詳細資料
```python
grid.tradedata(save=True)
```
