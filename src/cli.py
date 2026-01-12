import fire

from src.evaluat import evaluate_model
from src.training import train


class App:
    def train(self, **kwargs):
        return train(**kwargs)

    def evaluate(self, **kwargs):
        return evaluate_model(**kwargs)


def main():
    fire.Fire(App)


if __name__ == "__main__":
    main()
