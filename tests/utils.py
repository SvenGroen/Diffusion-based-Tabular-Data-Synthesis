import os
import lib

def get_test_data():
    cwd = os.getcwd()
    if "tests" in cwd:
        data_path = os.path.normpath("data/adult")
    else:
        data_path = os.path.normpath("tests/data/adult")

    x_num, x_cat, y = lib.read_pure_data(data_path, "train")
    info = lib.load_json(os.path.join(data_path, 'info.json'))["dataset_config"]
    return x_cat, x_num, y, info

class ProcessorFactory():
    x_cat, x_num, y, info = get_test_data()

    @staticmethod
    def get_data_sample(size=100):
        return ProcessorFactory.x_cat[:size], ProcessorFactory.x_num[:size], ProcessorFactory.y[:size]

    @staticmethod
    def get_instance(type):
        if type == "ft":
            from tabular_processing.ft_processor import FTProcessor
            processor = FTProcessor(
                x_cat=ProcessorFactory.x_cat,
                x_num=ProcessorFactory.x_num,
                y=ProcessorFactory.y,
                cat_columns=ProcessorFactory.info['cat_columns'],
                problem_type=ProcessorFactory.info['problem_type'],
                target_column=ProcessorFactory.info['target_column'])
            return processor

