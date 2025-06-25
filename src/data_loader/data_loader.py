import re
import typing as tp
from datetime import datetime

import pandas as pd
from email.utils import parsedate_tz, mktime_tz

MAX_SIZE = 50000


class EmailsDataLoader:

    def __init__(self, dataset_path: str) -> None:
        self.dataset: tp.Optional[pd.DataFrame] = None
        self.dataset_path: str = dataset_path

    def _extract_email_data(self, raw_email: str) -> tp.Dict[str, tp.Any]:
        """Extract message ID, content, and date from raw email"""

        # Extract Message ID
        message_id_match = re.search(
            r"Message-ID:\s*<([^>]+)>", raw_email, re.IGNORECASE
        )
        message_id = message_id_match.group(1) if message_id_match else ""

        # Extract Date
        date_match = re.search(r"Date:\s*(.+)", raw_email)
        date = None
        if date_match:
            try:
                date_str = date_match.group(1).strip()
                parsed = parsedate_tz(date_str)
                if parsed:
                    date = datetime.fromtimestamp(mktime_tz(parsed))
            except:
                date = None

        # Extract email content
        if "\n\n" in raw_email:
            content = raw_email.split("\n\n", 1)[1]
            content = re.sub(r"-{10,}.*?-{10,}", "", content, flags=re.DOTALL)
            content = re.sub(r"\n{3,}", "\n\n", content.strip())
        else:
            content = ""

        return {"message_id": message_id, "date": date, "content": content}

    def preprocess(self, raw_email_col: str) -> None:
        if self.dataset is None:
            raise ValueError("Dataset not loaded")

        extracted_data = self.dataset[raw_email_col].apply(
            self._extract_email_data
        )

        self.dataset["message_id"] = extracted_data.apply(lambda x: x["message_id"])
        self.dataset["email_date"] = extracted_data.apply(lambda x: x["date"])
        self.dataset["email_content"] = extracted_data.apply(lambda x: x["content"])

    @classmethod
    def load(cls, dataset_path: str) -> "EmailsDataLoader":
        loader = cls(dataset_path)

        try:
            dataset: pd.DataFrame = pd.read_csv(dataset_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        except pd.errors.EmptyDataError:
            raise ValueError(f"Dataset file is empty: {dataset_path}")

        if len(dataset) > MAX_SIZE:
            dataset = dataset.sample(n=MAX_SIZE, random_state=42)

        loader.dataset = dataset
        return loader
