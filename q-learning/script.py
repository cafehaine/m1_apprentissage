from island import Island
from algos.epsilongreedy import epsilongreedy

def main():
    island = Island(6, 6, 5)
    score = epsilongreedy(island, print_island=True)
    print("Score final:", score)

if __name__ == "__main__":
    main()
