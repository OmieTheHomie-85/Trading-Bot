import os
from dotenv import load_dotenv

load_dotenv()
KEY = os.getenv("KEY")
SECRET = os.getenv("SECRET")
DISCORD = os.getenv("DISCORD")
channel_id = 1485476029337440387
symbols = ["SPY", "QQQ", "VTI", "IWN", "EFA", "EEM", "TLT", "GLD"]