import math

datasets = [
    ('bonne humeur', [False, True, True, False, True, True, False, True]),
    ('beau temps', [True, False, True, True, True, False, False, False]),
    ('gouter pris', [False, True, False, True, True, False, True, False]),
]

decision = [True, True, True, True, False, False, False, False]

def main():
    for name, data in datasets:
        print(name)
        trues = []
        falses = []
        for i in range(len(decision)):
            if data[i]:
                trues.append(decision[i])
            else:
                falses.append(decision[i])

        in_true_out_true = trues.count(True) / len(trues)
        in_true_out_false = trues.count(False) / len(trues)
        in_false_out_true = falses.count(True) / len(falses)
        in_false_out_false = falses.count(False) / len(falses)
        j_true = -(in_true_out_true * math.log2(in_true_out_true)) - in_true_out_false * math.log2(in_true_out_false)
        j_false = -(in_false_out_true * math.log2(in_false_out_true)) - in_false_out_false * math.log2(in_false_out_false)
        h = len(trues)/len(decision) * j_true + len(falses) / len(decision) * j_false
        print("J true", j_true)
        print("J false", j_false)
        print("H", h)

if __name__ == '__main__':
    main()
