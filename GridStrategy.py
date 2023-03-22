import pandas as pd
import numpy as np
import operator
import mplfinance as mpf
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt
plt.style.use('seaborn')

class GridStrategy:

    def __init__(self, wealth, min_data_path, day_data_path):
        
        #導入訓練資料
        self.data = pd.read_csv(day_data_path,parse_dates=['Date'])
        self.fold=int(len(self.data)*0.9)
        
        #導入回測資料
        mindata = pd.read_csv(min_data_path, parse_dates=[['Date','Time']])
        mindata = mindata.rename(columns={'Date_Time':'Date'})
        self.mindata = mindata.set_index('Date')
        
        self.wealth = wealth
        
    def min_data_time(self, start_time, end_time):
        
        ###################用時間切'分'資料###################
        
        start_time=pd.to_datetime(start_time)
        end_time=pd.to_datetime(end_time)
        
        self.mindata=self.mindata[(self.mindata.index>=start_time) & (self.mindata.index<=end_time)]

    def train_lstm(self):
        
        ###################訓練LSTM模型###################

        #creating train and test sets
        dataset = self.data[['Date','Close']].set_index('Date').values

        train = dataset[:self.fold,:]
        self.valid = dataset[self.fold:,:]

        #converting dataset into x_train and y_train
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scaler.fit_transform(dataset)

        x_train, y_train = [], []
        for i in range(7,len(train)):
            x_train.append(scaled_data[i-7:i,0])
            y_train.append(scaled_data[i,0])
        x_train, y_train = np.array(x_train), np.array(y_train)

        x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

        # create and fit the LSTM network
        self.model = Sequential()
        self.model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
        self.model.add(LSTM(units=50))
        self.model.add(Dense(1))

        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)
        

        #預測valid
        inputs = self.data['Close'][len(self.data) - len(self.valid) - 7:].values
        inputs = inputs.reshape(-1,1)
        self.inputs  = self.scaler.transform(inputs)

        X_test = []
        for i in range(7,self.inputs.shape[0]):
            X_test.append(self.inputs[i-7:i,0])
        X_test = np.array(X_test)

        X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
        closing_price = self.model.predict(X_test)
        self.closing_price = self.scaler.inverse_transform(closing_price)
        
    
    def rms_valid(self):
        
        ###################valid集的rms###################
        
        rms=np.sqrt(np.mean(np.power((self.valid-self.closing_price),2)))
        
        return(rms)

    
    def plot_valid(self):
        
        ###################畫valid圖###################
    
        plt.figure(figsize=(16,8))

        #for plotting
        #train = data[['Date','Close']][:fold]
        valid = self.data[['Date','Close']][self.fold:]
        valid['Predictions'] = self.closing_price
        #plt.plot(train['Close'])
        plt.plot(valid[['Close','Predictions']])
    
    def inputdata(self, path=None, start_time=None, end_time=None):
        
        ###################導入預測input資料（非必要）###################
        #沒導入則使用訓練資料的最後7日資料
        
        if path!=None:
            
            data=pd.read_csv(path, parse_dates=['Date'])

            inputs = data['Close'][-7:].values
        
        if start_time!=None and end_time!=None:
            
            start_time=pd.to_datetime(start_time)
            end_time=pd.to_datetime(end_time)
        
            inputs=self.data['Close'][(self.data['Date']>=start_time) & (self.data['Date']<=end_time)].values
            
        inputs = inputs.reshape(-1,1)
        self.inputs  = self.scaler.transform(inputs)
        
        
    def output(self, multitrain='Return'):
        
        ###################產生最佳價格範圍、格數###################
        #利用建構完成之LSTM模型，導入前七日收盤（input）預測未來一日價格
        #使用前七日之‘分’收盤資料，計算1日移動標準差，取最大*6倍即為上下範圍
        #固定範圍，最大化報酬 ->最佳化格數（10<=格數<=200，<10交易量少，報酬不穩定）
        
        #predict 7/24, input=7/16-7/23 day close
        X_test = np.array([self.inputs[-7:,0]])
        X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
        closing_price = self.model.predict(X_test)
        predict_price = self.scaler.inverse_transform(closing_price)
        
        #get range
        crit=4
        price=predict_price[-1]
        rolling_Std = self.mindata['Close'].iloc[-7*60*24:].std()
        self.highest = (price+crit*rolling_Std)[0]
        self.lowest = (price-crit*rolling_Std)[0]
        
        #最佳化格數,最大化報酬
        self.performance={num:self.train_grid(quantity=num, multitrain_cum = multitrain) for num in range(10,100)}
        self.quantity = max(self.performance.items(), key=operator.itemgetter(1))[0]
        
        print('Highest Price: {}'.format(self.highest))
        print('Lowest Price: {}'.format(self.lowest))
        print('Quantity: {}'.format(self.quantity))
    
    
    def train_grid(self, wealth=None, lowest=None, highest=None, quantity=None, unit_return=None, func='等比', multitrain_cum=None):
        
        ###################網格策略回測###################
        #multitrain_cum = True 則返回累積報酬率
        
        #初始化
        
        if lowest == None:
            lowest = self.lowest
        if highest == None:
            highest = self.highest
            
        if quantity == None:
            if unit_return != None:
                self.quantity = round(math.log((highest/lowest),(1+unit_return)))
                
            quantity = self.quantity
        else:
            if unit_return != None:
                print('Error:不可同時給定quantity與unit_return')

        if wealth == None:
            wealth = self.wealth
        
        self.intervals = (lowest-highest)/quantity
        
        self.signal_df=[]
        self.last_price_index = None
        self.last_price_index_df=[]
        
        #產生網格
        if func == '等差':
            self.price_levels = [x for x in np.arange(highest, lowest+self.intervals/2, self.intervals)]
        elif func == '等比':
            self.price_levels = [highest * pow(lowest/highest,1/quantity)**i for i in range(quantity)]
        else:
            print('func=等差\等比')
        
        #產生訊號
        for close in self.mindata['Close']:
            if self.last_price_index == None:
                for i in range(len(self.price_levels)):
                    if close < self.price_levels[i]:
                        self.last_price_index = i
            signal = 0 
            while True:
                upper = None
                lower = None
                if self.last_price_index > 0:
                    upper = self.price_levels[self.last_price_index - 1]
                if self.last_price_index < len(self.price_levels) - 1:
                    lower = self.price_levels[self.last_price_index + 1]
                # 还不是最轻仓，继续涨，就再卖一档
                if upper != None and close > upper:
                    self.last_price_index = self.last_price_index - 1
                    signal = -1
                    continue
                # 还不是最重仓，继续跌，再买一档
                if lower != None and close < lower:
                    self.last_price_index = self.last_price_index + 1
                    signal = 1
                    continue
                break
            self.signal_df.append(signal)
        self.signal_df=pd.Series(self.signal_df,index=self.mindata.index)


        #position
        self.position_df=self.signal_df.cumsum()
        
        #交易股數
        trade_stock_df=pd.DataFrame(index=self.mindata.index)
        trade_stock_df=self.signal_df.shift(1)*((wealth/quantity)/self.mindata['Close'].shift(1))
        self.trade_stock_df=trade_stock_df.fillna(0).cumsum()

        return_df=self.mindata['Close'].diff(1)
        #交易成本
        self.cost_df = (abs(self.trade_stock_df-self.trade_stock_df.shift(-1))*self.mindata['Close']*0.00036).fillna(0)
        #報酬
        trade_return_df=(return_df*self.trade_stock_df)-self.cost_df
        cum_trade_return_df=trade_return_df.cumsum()

        #performance
        self.cum_trade_percent_return=(cum_trade_return_df/wealth)*100
        self.MDD_series=self.cum_trade_percent_return.cummax()-self.cum_trade_percent_return
        self.high_index=self.cum_trade_percent_return[self.cum_trade_percent_return.cummax()==self.cum_trade_percent_return].index

        self.MDD=round(self.MDD_series.max(),2)
        self.Cumulative_Return=round(self.cum_trade_percent_return.iloc[-1],2)
        self.Return_on_MDD=round(self.cum_trade_percent_return.iloc[-1]/self.MDD_series.max(),2)
        daily_return=self.cum_trade_percent_return.diff(1)
        try:
            self.Sharpe_Ratio=round((daily_return.mean()/daily_return.std())*pow(525600,0.5),2)
        except:
            self.Sharpe_Ratio=0
        self.trade_volume=abs(self.signal_df).cumsum().iloc[-1]
        
        #返回累積報酬率
        if multitrain_cum != None:
            
            if multitrain_cum == 'Return':
                
                return(self.Cumulative_Return)
            
            elif multitrain_cum == 'SR':
                
                return(self.Sharpe_Ratio)
        
    def tradedata(self, save = False):
        
        ###################交易細項###################
        #save = True 則儲存成csv檔
        
        trade_data=pd.concat([self.mindata['Close'],self.signal_df,self.position_df,self.cum_trade_percent_return,abs(self.trade_stock_df.shift(-1)-self.trade_stock_df),self.cost_df],axis=1,keys=['CLOSE','SIGNAL','POSITION','CUM RETURN%','VOLUME','COST'],names=['wealth: {}'.format(self.wealth)])
        self.trade_data=trade_data[trade_data['COST']!=0].dropna()
        
        #儲存csv
        if save == True:
            
            self.trade_data.to_csv('交易細項.csv')
        
        return(self.trade_data)
    
    def plot_Grid(self):
        
        ###################網格收盤走勢圖###################

        plt.style.use('seaborn')
        
        plt.figure(figsize=(12,6))
        plt.title('Grid Strategy',fontsize=16)

        self.mindata['Close'].plot(label='Close')
        plt.ylim([self.mindata['Close'].min(axis=0)*0.99,self.mindata['Close'].max(axis=0)*1.01])

        for price in self.price_levels:
            plt.axhline(price,ls='--',c='grey',linewidth=1)

        plt.legend()
    
    def plot_performance(self):
        
        ###################回測報酬圖###################
        
        fig,ax=plt.subplots(figsize=(16,6))

        (self.cum_trade_percent_return).plot(label='Total Return',ax=ax,c='r')
        plt.fill_between(self.MDD_series.index,-self.MDD_series,0,facecolor='r',label='DD')
        plt.scatter(self.high_index,self.cum_trade_percent_return.loc[self.high_index],c='#02ff0f',label='High')

        plt.legend()
        plt.ylabel('Return%')
        plt.xlabel('Date')
        plt.title('Return & MDD',fontsize=16);

        #performance data
        print('Price Range: {:.2f} to {:.2f}, Quantity: {}'.format(self.lowest, self.highest, self.quantity))
        print('Yearly Cumulative Return: {}%'.format(round((pow(self.Cumulative_Return/100+1,365/27)-1)*100),2))
        print('Weekly Cumulative Return: {}%'.format(self.Cumulative_Return))
        print('Yearly Max Cumulative Return: {}%'.format(round((pow(self.cum_trade_percent_return.max()/100+1,365/27)-1)*100),2))
        print('MDD: {}%'.format(self.MDD))
        print('Return on MDD: {}'.format(self.Return_on_MDD))
        print('Sharpe Ratio: {}'.format(self.Sharpe_Ratio))
        print('Trade Volume: {}'.format(self.trade_volume))
        
    def plot_candlestick(self):
        
        ###################k棒進出場圖###################
        
        signal_long=self.mindata['Close'][self.signal_df==1]
        signal_short=self.mindata['Close'][self.signal_df==-1]
        
        Total_data=pd.concat([self.mindata,self.signal_df,signal_long,signal_short], axis=1, sort=False)
        Total_data.columns=['Open','High','Low','Close','TotalVolume','signal','signal_long','signal_short']
        
        title='candle stick plot'
        
        add_plot = [mpf.make_addplot(Total_data['signal_long'], scatter=True,markersize=80,marker='^',color='pink'),
                    mpf.make_addplot(Total_data['signal_short'], scatter=True,markersize=80,marker='v',color='lightgreen')]

        mc = mpf.make_marketcolors(up='r',
                                   down='g',
                                   edge='',
                                   wick='inherit',
                                   volume='inherit')
        
        s = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc, gridaxis= 'vertical')
        
        mpf.plot(self.mindata,
                 hlines=dict(hlines=self.price_levels,colors='#BEBEBE',linestyle='--'), 
                 type='candle', style=s, addplot=add_plot, figsize=(12,8),
                 ylim=(self.mindata['Close'].min(axis=0)*0.99, self.mindata['Close'].max(axis=0)*1.01))
        