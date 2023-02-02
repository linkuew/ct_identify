from library import *
import os, sys

def main():
    if len(sys.argv) != 13:
        usage(sys.argv[0])
        exit(1)

    data_index, train_perc, test_perc, feat, n1, n2, num_feat, func = process_args(sys.argv[0])

    # check to make sure the LOCO partition exists
    if not os.path.exists('../data/LOCO_partition.json'):
        partition_dataset()

    with open('../data/LOCO_partition.json') as f:
        data = json.load(f)

    #bigfoot, climate, flat, pizza, vaccine
    ct = split_data(data)
    ct = ct[data_index]

    # get x and y data from the ct
    xdata, ydata = x_y_split(ct)

    # find best features for this ct
    feature_selection(xdata, ydata, num_feat, n1, n2, func, feat, data_index, 10, False)

if __name__ == "__main__":
    main()
