from data.swarmset import SwarmDataset

def scrape_controllers(dataset_path):
    controllers = []
    data = SwarmDataset(dataset_path)
    for i in range(len(data)):
        controllers.append(list(data[i][1]))
    return controllers

if __name__ == "__main__":
    print(scrape_controllers("../../data/mrs-q1-samples"))
