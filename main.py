import logging
from data.retrieve import Retriever
from utils.parser import parse


class Translation:
    def __init__(self, framework: str = "pytorch", type: str = "letter_by_letter"):
        self.framework = framework
        self.type = type

    def build(self):
        logging.info("Retrive data from hugging face")
        retriever = Retriever()
        logging.info(retriever.data)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    args = parse()
    translator = Translation(framework=args.framework, type=args.type)
    translator.build()
