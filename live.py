import numpy as np
from hilbert import decode, encode
import requests
from tqdm import tqdm
import pandas as pd
import torch
from comet_ml import Experiment
import subprocess
from comet_ml.integration.pytorch import load_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import datetime
import websocket
import json
device = torch.device('cpu')
print(device)
os.environ["COMET_API_KEY"] = "uM0HPEvEu6OyX3dTEuB4Fihgz"

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(5, 8, 5,dilation=4,padding="same")
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.pool = nn.MaxPool2d(3, 2)
        self.pool2 = nn.AvgPool2d(3, 2)
        
        self.conv2 = nn.Conv2d(8, 16, 5,dilation=3,padding="same")
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        
        self.conv3 = nn.Conv2d(16, 24, 3,dilation=2)
        self.conv4 = nn.Conv2d(24, 48, 2)
        self.conv5 = nn.Conv2d(48, 64, 2)
        self.conv6 = nn.Conv2d(64, 128, 1)
        
        self.fc1 = nn.Linear(1152, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 16)
        self.fc6 = nn.Linear(16, 8)
        
        self.batch1=nn.BatchNorm2d(8)
        self.batch2=nn.BatchNorm2d(16)
        self.batch3=nn.BatchNorm2d(24)
        self.batch4=nn.BatchNorm2d(64)
        
        self.dropout1=nn.Dropout2d(p=0.2)
        

    def forward(self, x):
        x = self.dropout1(self.batch1(F.relu(self.conv1(x))))
        x = self.pool(self.batch2(F.relu(self.conv2(x))))
        x = self.pool2(self.batch3(F.relu(self.conv3(x))))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        x = self.fc6(x)
        return x
    
class LiveTrader():
    def __init__(self):
        self.model = Net()
        print("Loading Model")

        checkpoint = load_model("experiment://c9d790db2d0544c7a0fcd25ae43ecb90/4_layer_w_volume_classic_CNN_mse_target3",map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        self.model.to(device)
        print("Model Loaded")

        self.open_position=False
        self.position_history = []
        self.output_history = []
        self.current_position={}
        self.pos_dict_open = {}
        self.pos_dict_closed={}
        self.odf_s = np.array([0.00088979, 0.00134634, 0.00088979, 0.00134634, 0.00088979,
        0.00134634, 0.00088979, 0.00134634])
        self.odf_m = np.array([-0.00102187,  0.00315814, -0.00102187,  0.00315814, -0.00102187,
         0.00315814, -0.00102187,  0.00315814])
        self.ohlc_mean=27993.054512542625
        self.ohlc_std=17832.786089817815
        self.vol_mean=248.0463029065656
        self.vol_std=426.867918968892
        self.long_short=0
        self.verbose=0
        self.data_s = np.array(range(32*32))
        self.locs = decode(self.data_s, 2, 5)
        self.margin=3
        self.profit_margin=1
        self.last_timestamp=0
        self.trade_available=True
        print("Initialized")
    def send_slack(self,msg):
        webhook_url = "https://hooks.slack.com/services/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"
        slack_data = {
            "text": "New Action!!!",
            "attachments": [
                {"text":f"{i}:{message[i]}"} for i in message 
            ]
        }

        response = requests.post(
            webhook_url, data=json.dumps(slack_data),
            headers={'Content-Type': 'application/json'}
        )

    def live_call(self,_timestamp,_open,_high,_low,_close):
        #print(f"Open Position is {self.open_position}")
        if not self.trade_available:
            resid = _timestamp%60000
            if (resid>58000) and (self.last_timestamp+60000<_timestamp):
                self.trade_available=True
                
        if self.trade_available:
            if self.open_position:
                self.in_position(_timestamp,_open,_high,_low,_close)
            else:
                print("time",_timestamp,_timestamp%60000)
                self.out_of_position(_timestamp,_open,_high,_low,_close)
        
    def out_of_position(self,_timestamp,_open,_high,_low,_close):
        cols=[
                "OpenTime",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "CloseTime",
                "QuoteAsset",
                "NumberOfTrades",
                "TakerBuybaseassetvolume",
                "TakerBuyquoteassetvolume",
                "Ignore"
             ]
        print("Evaluating Position")
        market = 'BTCUSDT'
        tick_interval = '1m'
        url = 'https://fapi.binance.com/fapi/v1/klines?symbol='+market+'&interval='+tick_interval+'&limit=1024'
        data = requests.get(url).json()
        df = pd.DataFrame(data,columns=cols)
        #current_ts = df.iloc[-1].copy(deep=True)
        #df = df.iloc[:-1]
        print(f"Fetched {len(df)} rows")
        pos_dict_open = {}
        o1 = np.zeros((5,32,32))
        vals = df[["Open","High","Low","Close","Volume"]].values
        for j in range(len(self.data_s)):
            o1[:,self.locs[j][0],self.locs[j][1]] = vals[j]
        print("Rearranged to hilbert curve")
        ohlc = o1[:4,:,:]
        vol = o1[-1:,:,:]
        ohlc = (ohlc-self.ohlc_mean)/self.ohlc_std
        vol = (vol-self.vol_mean)/self.vol_std
        print("Normalized")
        on1 = torch.Tensor(np.concatenate((ohlc,vol),axis=0).reshape((1,*o1.shape)))
        output=self.model(on1.to(device)).tolist()[0]
        print(f"Predicted Tensor of shape {on1.shape}")
        
        self.output_history.append(output)
        short = min(output[::2])
        
        long = max(output[1::2])
        
        short_index = output.index(short)//2
        
        long_index = output.index(long)//2
        
        norm_max = np.abs((np.array(output) - self.odf_m)/self.odf_s).argmax()
        
        if norm_max%2:
            self.long_short=1
        else:
            self.long_short=-1
        print(f"Long Short Direction {self.long_short}")
        _type=None
        _tp=None
        if self.long_short==1:
            _type="long"
            self.open_position=True
            long=long*1.25
            if long_index==0:
                stop_loss=output[0]/3
            else:
                stop_loss = min(output[::2][:long_index])/3
            _tp=long
            self.pos_dict_open["status"]="Opened"
            self.pos_dict_open["time"]=int(_timestamp)
            self.pos_dict_open["price"]=float(_close)
            self.pos_dict_open["type"]="long"
            self.current_position["type"]="long"
            self.pos_dict_open["take_profit"]=float(_close)*(1+long)
            self.current_position["take_profit"]=float(_close)*(1+long)
            self.pos_dict_open["stop_loss"]=float(_close)*(1+(stop_loss))
            self.current_position["stop_loss"]=float(_close)*(1+(stop_loss))
            self.pos_dict_open["take_profit_ratio"]=(1+long*self.margin)
            self.current_position["take_profit_ratio"]=(1+long*self.margin)
            self.pos_dict_open["stop_loss_ratio"]=(1+(stop_loss)*self.margin)
            self.current_position["stop_loss_ratio"]=(1+(stop_loss)*self.margin)
            self.pos_dict_open["result"]=None
            if self.verbose==2:
                print(_open,_high,_low,_close)
                print("\nOpened Position",self.pos_dict_open)
        
        elif self.long_short==-1:
            _type="short"
            self.open_position=True
            
            short=short*1.25
            
            if short_index == 0:
                stop_loss = output[1]/4
            else:
                stop_loss = max(output[1::2][:short_index])/4
            _tp=short
            self.pos_dict_open["status"]="Opened"
            self.pos_dict_open["time"]=int(_timestamp)
            self.pos_dict_open["price"]=float(_close)
            self.pos_dict_open["type"]="short"
            self.current_position["type"]="short"
            self.pos_dict_open["take_profit"]=float(_close)*(1+short)
            self.current_position["take_profit"]=float(_close)*(1+short)
            self.pos_dict_open["stop_loss"]=float(_close)*(1+(stop_loss))
            self.current_position["stop_loss"]=float(_close)*(1+(stop_loss))
            self.pos_dict_open["take_profit_ratio"]=(1-short*self.margin)
            self.current_position["take_profit_ratio"]=(1-short*self.margin)
            self.pos_dict_open["stop_loss_ratio"]=(1-(stop_loss)*self.margin)
            self.current_position["stop_loss_ratio"]=(1-(stop_loss)*self.margin)
            self.pos_dict_open["result"]=None
            if self.verbose==2:
                print(_open,_high,_low,_close)
                print("\nOpened Position",self.pos_dict_open)

        self.position_history.append(self.pos_dict_open)
        msg={
            
                            "Action": "Open Position",
                            "Type":f" {self.current_position['type']}",
                            "Margin":f" {self.margin}",
                            "Time":f" {pd.to_datetime(datetime.datetime.fromtimestamp(_timestamp/1000.0))}",
                            "Price":f" {_close}",
                            "StopLoss":f" {self.current_position['stop_loss']}",
                            "TakeProfit":f" {self.current_position['take_profit']}",
                            "StopLossRatio":f" {self.current_position['stop_loss_ratio']}",
                            "TakeProfitRatio":f" {self.current_position['take_profit_ratio']}",
                        }
        self.send_slack(msg)
        self.pos_dict_open={}
        
    def in_position(self,_timestamp,_open,_high,_low,_close):
        if self.current_position["type"]=="long":
            if float(_close)<self.current_position["stop_loss"]:
                print("long loss condition satisfied")
                self.pos_dict_closed["status"]="Closed"
                self.pos_dict_closed["time"]=int(_timestamp)
                self.pos_dict_closed["price"]=float(_low)
                self.pos_dict_closed["result"] = "loss"
                self.profit_margin*=self.current_position["stop_loss_ratio"]*0.99985
                self.open_position=False
            elif float(_close)>self.current_position["take_profit"]:
                print("long profit condition satisfied")
                self.pos_dict_closed["status"]="Closed"
                self.pos_dict_closed["time"]=int(_timestamp)
                self.pos_dict_closed["price"]=float(_high)
                self.pos_dict_closed["result"] = "profit"
                self.profit_margin*=self.current_position["take_profit_ratio"]*0.99985
                self.open_position=False
        elif self.current_position["type"]=="short":
            if float(_close)>self.current_position["stop_loss"]:
                print("short loss condition satisfied")
                self.pos_dict_closed["status"]="Closed"
                self.pos_dict_closed["time"]=int(_timestamp)
                self.pos_dict_closed["price"]=float(_high)
                self.pos_dict_closed["result"] = "loss"
                self.profit_margin*=self.current_position["stop_loss_ratio"]*0.99985
                self.open_position=False
            elif float(_close)<self.current_position["take_profit"]:
                print("short profit condition satisfied")
                self.pos_dict_closed["status"]="Closed"
                self.pos_dict_closed["time"]=int(_timestamp)
                self.pos_dict_closed["price"]=float(_low)
                self.pos_dict_closed["result"] = "profit"
                self.profit_margin*=self.current_position["take_profit_ratio"]*0.99985
                self.open_position=False

        
        if not self.open_position:
            self.last_timestamp=int(_timestamp)
            
            print("Position Closed")
            self.pos_dict_closed["type"] = self.current_position["type"]
            self.pos_dict_closed["take_profit"] = self.current_position["take_profit"]
            self.pos_dict_closed["stop_loss"] = self.current_position["stop_loss"]
            self.pos_dict_closed["take_profit_ratio"] = self.current_position["take_profit_ratio"]
            self.pos_dict_closed["stop_loss_ratio"] = self.current_position["stop_loss_ratio"]
            self.position_history.append(self.pos_dict_closed)
            msg={
            
                            "Action": "Close Position",
                            "Type":f" {self.current_position['type']}",
                            "Margin":f" {self.margin}",
                            "Result":f"{self.pos_dict_closed['result']}"
                            "Time":f" {pd.to_datetime(datetime.datetime.fromtimestamp(_timestamp/1000.0))}",
                            "Price":f" {_close}",
                            

                        }
            self.send_slack(msg)
            msg={
                "Profit Margin":self.profit_margin,
                            }
                        
            self.send_slack(msg)
            if self.verbose==2:
                print("\nClosed Position", self.pos_dict_closed)
                print("\nProfit Margin: ",self.profit_margin)
            self.pos_dict_closed={}
            self.current_position={}
            self.trade_available=False
    

def on_open(ws):
    print("opened")
    subscribe_message = {
        "method": "SUBSCRIBE",
        "params":
        [
         "btcusdt@kline_1m",
         ],
         "id": 1
         }

    ws.send(json.dumps(subscribe_message))

def on_message(ws, message):
    global lt
    #print("received a message")
    #print(json.loads(message))    
    msg=json.loads(message)
    #print(msg["k"]["t"])
    lt.live_call(msg["E"],msg["k"]["o"],msg["k"]["h"],msg["k"]["l"],msg["k"]["c"])
def on_close(ws):
    print("closed connection")

lt=LiveTrader()
socket='wss://fstream.binance.com/ws/'
ws = websocket.WebSocketApp(socket, on_open=on_open, on_message=on_message, on_close=on_close)
ws.run_forever()
