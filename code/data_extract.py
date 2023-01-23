import json

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
