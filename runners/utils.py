import os


def check_sim_mech(mech, arch, net, nl, nh):
    path = os.path.join('results', 'simulations', mech, 'carefl', arch)
    files = os.listdir(path)
    acc = []
    for i in [25, 50, 75, 100, 150, 250, 500]:
        if 'sim_{}_{}_{}_{}_{}.p'.format(i, arch, net, nl, nh) not in files:
            acc.append(i)
    # print('[SIM] mech: {}, arch: {}, net: {}, nl: {}, nh: {}'.format(mech, arch, net, nl, nh), acc)
    return acc


def check_sim(arch, net, nl, nh):
    mechs = ['linear', 'hoyer2009', 'nueralnet_l1']
    acc = {mech: check_sim_mech(mech, arch, net, nl, nh) for mech in mechs}
    return acc


def check_int(arch, net, nl, nh, r=True):
    path = os.path.join('results', 'interventions', 'carefl', arch)
    files = os.listdir(path)
    acc = []
    for i in [250, 500, 750, 1000, 1250, 1500, 2000, 2500]:
        if 'int_{}{}_{}_{}_{}_{}.p'.format(i, 'r' * r, arch, net, nl, nh) not in files:
            acc.append(i)
    # print('[INT] arch: {}, net: {}, nl: {}, nh: {}, r: {}'.format(arch, net, nl, nh, r), acc)
    return acc
