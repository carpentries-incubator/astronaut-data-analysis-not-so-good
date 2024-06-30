import pandas as pd
import matplotlib.pyplot as plt
import sys



if __name__ == '__main__':

    if len(sys.argv) < 3:
        data_f = './eva-data.json'
        data_t = './eva-data.csv'
        print(f'Using default input and output filenames')
    else:
        data_f = sys.argv[1]
        data_t = sys.argv[2]
        print('Using custom input and output filenames')

    g_file = './cumulative_eva_graph.png'

    print(f'Reading JSON file {data_f}')
    data = pd.read_json(data_f, convert_dates=['date'])
    data['eva'] = data['eva'].astype(float)
    data.dropna(axis=0, inplace=True)
    data.sort_values('date', inplace=True)

    print(f'Saving to CSV file {data_t}')
    data.to_csv(data_t, index=False)

    print(f'Plotting cumulative spacewalk duration and saving to {g_file}')
    data['duration_hours'] = data['duration'].str.split(":").apply(lambda x: int(x[0]) + int(x[1])/60)
    data['cumulative_time'] = data['duration_hours'].cumsum()
    plt.plot(data.date, data.cumulative_time, 'ko-')
    plt.xlabel('Year')
    plt.ylabel('Total time spent in space to date (hours)')
    plt.tight_layout()
    plt.savefig(g_file)
    plt.show()
    print("--END--")
