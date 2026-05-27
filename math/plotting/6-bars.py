##!/usr/bin/env python3
'''stacked bar chart'''
import numpy as np
import matplotlib.pyplot as plt


def bars():
    '''stacked bar chart of fruit sales'''
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))
    bar_width = 0.5
    x = np.arange(fruit.shape[1])
    plt.bar(x, fruit[0], width=bar_width, label='apples', color='red')

    plt.bar(x, fruit[1], width=bar_width, label='bananas',
            bottom=fruit[0], color='yellow')

    plt.bar(x, fruit[2], width=bar_width, label='oranges',
            bottom=fruit[0] + fruit[1], color='#ff8000')

    plt.bar(x, fruit[3], width=bar_width, label='peaches',
            bottom=fruit[0] + fruit[1] + fruit[2], color='#ffe5b4')

    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    plt.xticks(x, ['Farrah', 'Fred', 'Felicia'])
    plt.legend()
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))
    plt.show()
