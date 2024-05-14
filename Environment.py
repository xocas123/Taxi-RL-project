# taxi_environment.py
import gymnasium as gym

def create_taxi_environment():
    return gym.make('Taxi-v3')
