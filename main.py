from utils.parser import parse
from src.torch_translator_letter_by_letter import PyTorch_Letter_By_Letter_Translation
import logging


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    args = parse()
    if args.framework == "pytorch" and args.type == "letter_by_letter":
        translator = PyTorch_Letter_By_Letter_Translation()
        translator.build()
