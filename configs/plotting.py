import matplotlib.pyplot as plt

_prop_cycle = plt.rcParams['axes.prop_cycle']
_colors = _prop_cycle.by_key()['color']
_algorithms = ['carefl', 'careflns', 'lrhyv', 'reci', 'anm', 'anm-nn']

color_dict = {a: c for (a, c) in zip(_algorithms + ['linear'], _colors)}
color_dict['gp'] = color_dict['anm']

label_dict = {'carefl': 'CAREFL',
              'careflns': 'CAREFL-NS',
              'lrhyv': 'Linear LR',
              'reci': 'RECI',
              'anm': 'ANM',
              'gp': 'ANM-GP',
              'linear': 'ANM-linear',
              'anm-nn': 'ANM-NN'}

font_dict = {'title': 14, 'xlabel': 12, 'ylabel': 12}
