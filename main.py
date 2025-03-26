from utils.parser import parse
from src.torch_translator import Translator
import logging


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    translator = Translator()
    translator.build()
