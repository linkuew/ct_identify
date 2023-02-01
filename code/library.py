import json
import getopt,sys

glob_dict = {'bf' : 0, 'fe' : 1, 'cc' : 2, 'va' : 3, 'pg' : 4}

##
# Prints out the usage statements
##
def usage():
    print("Usage: python main.py -d [bf|fe|cl|va|pg] -t x,y -f [char|word] -r i,j")
    print()
    print("-d, dataset to use")
    print("\t bf = bigfoot")
    print("\t fe = flat earth")
    print("\t cc = climate change")
    print("\t va = vaccines")
    print("\t pg = pizzagate")
    print("-t, train percent, test percent, e.g.: 70,30")
    print("-f, feature set, either 'word' or 'char'")
    print("-r, n-gram range for features, e.g. 1,3")
    print("-h, print this help page")
    return 0

def process_args():
    args = []

    try:
        optlist, _ = getopt.getopt(sys.argv[1:], "hd:t:f:r:")

        for arg, val in optlist:
            if arg == "-h":
                usage()
            elif arg == "-d":
                dataset = glob_dict.get(val)
            elif arg == "-t":
                tmp = val.split(",")
                train = int(tmp[0])
                test = int(tmp[1])
            elif arg == "-r":
                tmp = val.split(",")
                low = int(tmp[0])
                upp = int(tmp[1])
            elif arg == "-f":
                feat = val
    except Exception as e:
        print(e)
        usage()
        exit(-1)

    return dataset, train, test, feat, low, upp

##
# function to partition dataset according to our pre-determined
#
# input: none - just make sure the LOCO.json file is the correct directory
#
# output: none - there should be a new file which is just the conspiracy theories we want
##
def partition_dataset():
    f = open('../data/LOCO.json')
    data = json.load(f)

    new_data = []

    for i in range(len(data)):
        if data[i]['seeds'].__contains__('big.foot') \
            or data[i]['seeds'].__contains__('flat.earth') \
            or data[i]['seeds'].__contains__('climate') \
            or data[i]['seeds'].__contains__('vaccine') \
            or data[i]['seeds'].__contains__('pizzagate'):
            new_data.append(data[i])

    with open('../data/LOCO_partition.json', 'w') as of:
        of.write('[')
        for i in range(len(new_data)):
            tmp_json = json.dumps(new_data[i], indent = 4)
            if i == (len(new_data) - 1):
                of.write(tmp_json)
            else:
                of.write(tmp_json + ",\n")
        of.write(']')

##
# Split the data into the different areas which we want to test on
#
# input: entire corpus
#
# returns: subcorpora
##
def split_data(data):
    vaccine = []
    bigfoot = []
    flat = []
    pizza = []
    climate = []
    for entry in data:
        if entry['seeds'].__contains__('big.foot'):
            bigfoot.append(entry)
        if entry['seeds'].__contains__('vaccine'):
            vaccine.append(entry)
        if entry['seeds'].__contains__('flat.earth'):
            flat.append(entry)
        if entry['seeds'].__contains__('pizzagate'):
            pizza.append(entry)
        if entry['seeds'].__contains__('climate'):
            climate.append(entry)

    return [bigfoot, climate, flat, pizza, vaccine]

##
# A helper function for the conspiracy select, this returns the text entry from the corpus
#
# input: a corpus
#
# output: a breakdown of the input corpus into text (X) and label (Y) - note that 1 marks conspiracy and 0 marks non-conspiracy
##
def x_y_split(data):
    X = []
    Y = []

    for entry in data:
        if entry['subcorpus'] == 'conspiracy':
            X.append(entry['txt'])
            Y.append(1)
        else:
            X.append(entry['txt'])
            Y.append(0)

    return X, Y
