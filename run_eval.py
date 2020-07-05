import argparse

from eval_2 import main

parser = argparse.ArgumentParser(description='Evaluate few-shot prototypical networks')

default_model_path = 'results/best_model.pt'
parser.add_argument('--model.model_path', type=str, default=default_model_path, metavar='MODELPATH',
                    help="location of pretrained model to evaluate (default: {:s})".format(default_model_path))

parser.add_argument('--data.test_way', type=int, default=5, metavar='TESTWAY',
                    help="number of classes per episode in test. 0 means same as model's data.test_way (default: 0)")
parser.add_argument('--data.test_shot', type=int, default=1, metavar='TESTSHOT',
                    help="number of support examples per class in test. 0 means same as model's data.shot (default: 0)")
parser.add_argument('--data.test_query', type=int, default=0, metavar='TESTQUERY',
                    help="number of query examples per class in test. 0 means same as model's data.query (default: 0)")
parser.add_argument('--data.test_episodes', type=int, default=10000, metavar='NTEST',
                    help="number of test episodes per epoch (default: 1000)")
parser.add_argument('--dist.qbits', type=int, default=0, metavar='QBITS',
                    help="number of quantization bits, no quantization if not specified (default: 0)")
parser.add_argument('--dist.distance', type=str, default="euclidean", metavar='QBITS',
                    help="distance metric (default: euclidean)")


args = vars(parser.parse_args())

main(args)
