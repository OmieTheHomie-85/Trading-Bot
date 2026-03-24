import os
from dotenv import load_dotenv

load_dotenv()
KEY = os.getenv("KEY")
SECRET = os.getenv("SECRET")
DISCORD = os.getenv("DISCORD")
channel_id = 1485476029337440387
symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "PLTR", "ARM", "SMCI", "QQQ", "SOXX", "ARKK", "TLT", "GLD"]