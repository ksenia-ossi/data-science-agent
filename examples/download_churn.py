"""Download the IBM Telco customer churn dataset (7,043 customers, 21 columns) to data/."""

from pathlib import Path
from urllib.request import urlretrieve

URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

if __name__ == "__main__":
    out = Path("data/telco_churn.csv")
    out.parent.mkdir(exist_ok=True)
    urlretrieve(URL, out)
    print(f"Saved {out} ({out.stat().st_size:,} bytes)")
