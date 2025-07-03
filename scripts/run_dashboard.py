import sys
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
from streamlit.web import cli as stcli

import numpy as np
import pandas as pd
from scipy.stats import beta

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

    target_script = "src/app/dashboard.py"

    # Streamlit CLI를 프로그램적으로 호출
    sys.argv = ["streamlit", "run", target_script]
    sys.exit(stcli.main())
