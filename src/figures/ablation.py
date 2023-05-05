import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import rc


if __name__ == '__main__':
    rc('axes', titlesize=25)     # fontsize of the axes title
    rc('axes', labelsize=20)    # fontsize of the x and y labels
    rc('xtick', labelsize=16)    # fontsize of the tick labels
    rc('ytick', labelsize=16)    # fontsize of the tick labels
    rc('legend', fontsize=14)    # legend fontsize
    rc('figure', titlesize=28)  # fontsize of the figure title
    rc('font', size=18)
    # rc('font', family='Times New Roman')
    rc('text', usetex=True)
    
    df = pd.read_csv('ablation.csv')
    print(df.head())

    fig = plt.figure(figsize=(20, 3))
    plt.rcParams['image.cmap'] = "Set2"
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set2.colors)

    i = 1
    for kernel in df.kernel.unique():
        for sigma in sorted(df.sigma.unique()):
            ax = fig.add_subplot(1, 4, i)
            data = df[(df.kernel == kernel) & (df.sigma == sigma)]
            data = data.sort_values(by='method')
            print(data)

            int_mae = data['ramp/int_mae'].values
            bacc = data['ramp/bacc'].values
            ext_mae = data['ramp/ext_mae'].values
            score = data['ramp/score'].values
            methods = data['method'].values

            width = 0.4
            x = np.arange(4)*2
            labels = ['Int. MAE', 'BAcc', 'Ext. MAE', 'Score']

            data = np.array([[int_mae[i], bacc[i], ext_mae[i], score[i]] for i in range(3)])

            if kernel == "rbf":
                ax.set_title(f"{kernel} ($\sigma$={sigma})")
            else:
                ax.set_title(f"{kernel} ($\gamma$={sigma})")
            
            alpha = 0.8
            ax.bar(x - width, data[0], width, label=methods[0], alpha=alpha)
            ax.bar(x, data[1], width, label=methods[1], alpha=alpha)
            ax.bar(x + width, data[2], width, label=methods[2], alpha=alpha)
            ax.set_ylim(0, 10)
            # ax.bar(x + width, data[3], width, label=methods[3])

            if i == 4:
                ax.legend()
            ax.set_xticks(x, labels, rotation=45)

            i += 1
    # fig.tight_layout()
    plt.savefig('ablation.pdf', dpi=200, bbox_inches='tight', pad_inches=0)
    plt.show()