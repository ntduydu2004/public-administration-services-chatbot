import csv
from config import logger

class CSVToSampleConverter:
    def __init__(self, csv_file_path: str):
        """
        Initialize the converter with the path to the CSV file.
        :param csv_file_path: Path to the CSV file.
        """
        self.csv_file_path = csv_file_path

    def convert_to_samples(self) -> list:
        """
        Convert the CSV file into a list of dictionaries with keys:
        'user_input', 'response', and 'retrieved_contexts'.
        :return: A list of dictionaries.
        """
        samples = []
        try:
            with open(self.csv_file_path, mode='r', encoding='utf-8') as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    samples.append({
                        "user_input": row["instruction"],
                        "retrieved_contexts": [row["output"]],  # Assuming output is used as retrieved context
                    })
        except FileNotFoundError:
            logger.debug(f"Error: File not found at {self.csv_file_path}")
        except KeyError as e:
            logger.debug(f"Error: Missing column in CSV file - {e}")
        except Exception as e:
            logger.debug(f"An unexpected error occurred: {e}")
        return samples
