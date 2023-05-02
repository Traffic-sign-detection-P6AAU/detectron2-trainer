from trainer.trainer import prepare_dataset, train, evaluate
from detectron2.config import get_cfg

CATEGORIES_PATH = 'data_handler/categories.json'

def main():
    cfg = get_cfg()
    print('---Menu list---')
    print('Type: 1 to prepare dataset and train')
    print('Type: 2 to evaluate the model')
    choice = input()
    if choice == '1':
        prepare_dataset()
        train(cfg)
        evaluate(cfg)
    elif choice == '2':
        evaluate(cfg)
    else:
        print('Input was not 1, 2, 3 or 4.')

if __name__ == '__main__':
    main()
