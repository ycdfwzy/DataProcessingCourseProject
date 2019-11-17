import csv
import classifiers.decision_tree as dec_tree

def read_data(filepath):
    with open(filepath) as fin:
        csv_data = csv.reader(fin)
        headings = next(csv_data)
        data_x = []
        labels = []
        for row in csv_data:
            raw_data = [int(row(1 + i)) for i in range(len(row[1:-1]))]
            data_x.append(raw_data)
            labels.append(int(row[-1]))

if __name__ == "__main__":
    
    pass