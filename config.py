import os
from dotenv import load_dotenv

load_dotenv()
KEY = os.getenv("KEY")
SECRET = os.getenv("SECRET")
symbols = ["SPY", "QQQ", "VTI", "IWN", "EFA", "EEM", "TLT", "GLD"]