# 網格交易結合LSTM模型預測價格範圍
## 策略介紹
1. 利用收盤價LSTM模型預測隔日收盤價  
2. 使用預測價格與7日波動度建構價格區間  
3. 網格策略:  
  * 以收盤價為標準,碰上格則放空,碰下格則買進  
  * 每筆交易資金:總資金/格數  
  * 超過價格邊界則持有部位直到回復價格範圍內  

## 使用方式舉例  
#### 讀入資料與總資金
```python
grid = GridStrategy(wealth = 1000000,
                    min_data_path = 'BTCUSDT-Minute-Trade.txt',
                    day_data_path = 'BTCUSDT-Day-Trade.txt')
```  
#### 訓練預測價格模型
```python
grid.train_lstm()
```  
#### 驗證集預測圖
```python
grid.plot_valid()
```
![plot_valid](https://github.com/hsuan1998/Grid_Strategy/blob/main/images/valid_plot.png)  
### 預測資料輸入
```python
grid.inputdata('input.txt')
```  
#### 產生價格範圍與格數
```python
grid.output()
```
Highest Price: [40891.047]  
Lowest Price: [28698.086]  
Quantity: 11  
#### 交易策略
```python
grid.train_grid()
```  
#### 走勢圖
```python
grid.plot_Grid()
```
![plot_Grid](https://github.com/hsuan1998/Grid_Strategy/blob/main/images/gird_plot.png)  
#### 進出場圖
```python
grid.plot_candlestick()
```
![plot_candlestick](https://github.com/hsuan1998/Grid_Strategy/blob/main/images/trade_plot.png)  
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
![plot_performance](https://github.com/hsuan1998/Grid_Strategy/blob/main/images/performance.png)  
#### 交易詳細資料
```python
grid.tradedata(save=True)
```
![tradedata](https://github.com/hsuan1998/Grid_Strategy/blob/main/images/trade_chart.png)
