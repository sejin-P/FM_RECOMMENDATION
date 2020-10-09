import csv
import sys

sys.path.extend(['/Users/parksejin/Line'])

import numpy as np
import pickle as pkl

from criteo_conversion_logs.real.fm import FM


def training(line_num: int, end_num: int, category_end: int, data_set: int):
    with open('data.txt') as f:
        lines = f.readlines()
    train_data = lines[:line_num]
    test_data = lines[line_num:end_num]
    model = FM(train_data, category_end)
    print("pre-processing start..")
    model.pre_process()
    print("setting test data....")
    model.pre_process_test(test_data)
    print("weight setting start...")
    model.set_weight()
    print("training start...")
    model.train(epoch=2, lambd=2e-5, delta=0.05, h=1)
    model.save(f"model{data_set}.pkl")
    for i, result in enumerate(model.test_evaluate()):
        print("label", model.test_label[i])
        print(result)
        if i == 100:
            break
    print(model.test_logloss())


def save_csv(model_name: str, file_name: str, line_num_start: int, line_num_end: int, is_train: bool):
    csv_file = np.array([list(range(line_num_start+1, line_num_end + 1))])
    with open(model_name, "rb") as f:
        load_model = pkl.load(f)

    with open('data.txt') as f:
        lines = f.readlines()
    datas = lines[line_num_start:line_num_end]
    total_click = []
    for data in datas:
        click = data.split("\t")[0]
        total_click.append(click)
    assert len(total_click) == len(csv_file[0])
    csv_file = np.vstack((csv_file, np.array(total_click)))
    if is_train:
        assert len(total_click) == len(load_model.train_label)
        csv_file = np.vstack((csv_file, load_model.train_label))
        csv_file = np.vstack((csv_file, load_model.train_evaluate()))
    else:
        assert len(total_click) == len(load_model.test_label)
        csv_file = np.vstack((csv_file, load_model.test_label))
        csv_file = np.vstack((csv_file, load_model.test_evaluate()))
    with open(file_name, "w", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        for line_no, click_timestamp, target_label, prob in zip(csv_file[0].tolist(), csv_file[1].tolist(),
                                                                csv_file[2].tolist(), csv_file[3].tolist()):
            writer.writerow([line_no, click_timestamp, target_label, prob])


def main():
    training(1000000, 1500000, 6, 1) # if you want to train dataset1 same as me, you should set train category index 2 to 6
    training(10653765, 15898883, 4, 2) # if you want to train dataset2 same as me, you should set train category index 2 to 4

    # save_csv("model1.pkl", "results_DS1_train.csv", 0, 1000000, True)
    # save_csv("model1.pkl", "results_DS1_test.csv", 1000000, 1500000, False)
    # save_csv("model2.pkl", "results_DS2_train.csv", 0, 10653765, True)
    # save_csv("model2.pkl", "results_DS2_test.csv", 10653765, 15898883, False)

if __name__ == '__main__':
     main()

