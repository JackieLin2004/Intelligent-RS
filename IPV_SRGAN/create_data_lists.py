from utils import create_data_lists


if __name__ == '__main__':
    create_data_lists(train_folders=['../dataset/super-resolution/train/train_HR'],
                      test_folders=['../dataset/super-resolution/test/test_HR'],
                      min_size=100,
                      output_folder='./data/')
