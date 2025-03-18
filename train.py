import ccxt
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import tweepy
from stable_baselines3 import PPO

# Настройка X API
auth = tweepy.OAuthHandler("gHSLlDReQY5sqmA7k6AIwEiz2", "nVTo3aTBatpeDdYK2btKrohXjIpIkLe1zO4I8bZuXuFHu7bjdT")
auth.set_access_token("1901993121099182080-e0er6I0nEyhKg3yH4CqnYlB6hDyVfr", "SL3ebxft9xcSRBR9wpgQ0kkoZRdFQUxBX75iYc8VdVfGH")
api = tweepy.API(auth)

class FuturesTradingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'urls': {'api': {'fapi': 'https://testnet.binancefuture.com'}},
            'options': {'defaultType': 'future'}
        })
        self.exchange.load_markets()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.balance = 1000
        self.position = 0
        self.entry_price = 0
        self.step_count = 0

    def reset(self, seed=None):
        self.balance = 1000
        self.position = 0
        self.entry_price = 0
        self.step_count = 0
        return self._get_observation(), {}

    def _get_observation(self):
        ohlcv = self.exchange.fetch_ohlcv('BTCUSDT', '1h', limit=10)
        df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        sma = df['close'].rolling(5).mean().iloc[-1]
        atr = df['close'].rolling(5).std().iloc[-1]
        tweets = api.search_tweets(q="Bitcoin", lang="en", count=10)
        sentiment = sum(1 if "bull" in t.text.lower() else -1 if "bear" in t.text.lower() else 0 for t in tweets)
        return np.array([df['close'].iloc[-1], sma, atr, df['volume'].iloc[-1], sentiment])

    def step(self, action):
        obs = self._get_observation()
        current_price = obs[0]
        reward = 0

        if self.position != 0:
            if (self.position > 0 and current_price < self.entry_price * 0.95) or \
               (self.position < 0 and current_price > self.entry_price * 1.05):
                profit = (current_price - self.entry_price) * abs(self.position) * 10
                self.balance += profit
                reward = profit
                self.position = 0
                self.entry_price = 0

        if action == 1 and self.position == 0:
            position_size = min(self.balance * 0.1, self.balance)
            self.position = position_size / current_price
            self.balance -= position_size
            self.entry_price = current_price
        elif action == 2 and self.position == 0:
            position_size = min(self.balance * 0.1, self.balance)
            self.position = -position_size / current_price
            self.balance -= position_size
            self.entry_price = current_price

        self.step_count += 1
        done = self.step_count >= 100 or self.balance <= 0
        return obs, reward, done, False, {}

# Обучение
env = FuturesTradingEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
model.save("ppo_futures")

# Скачивание
import shutil
shutil.make_archive("ppo_futures", 'zip', "ppo_futures")
print("Модель сохранена как ppo_futures.zip, скачай через кнопку 'Download' вверху")
