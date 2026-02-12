# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: dq_cuda_251
#     language: python
#     name: python3
# ---

# %%
# # !pip install bayesian-optimization

# %%
import deepquantum.photonic.circuit as circuit
import numpy as np
from deepquantum.optimizer import OptimizerBayesian, OptimizerFourier, OptimizerSPSA
from deepquantum.photonic.decompose import UnitaryDecomposer
from scipy.stats import unitary_group

np.set_printoptions(precision=8, floatmode='fixed', suppress=True)  # to make the print info aligned

N = 8
u8x8 = unitary_group.rvs(N)
decomp_rlt = UnitaryDecomposer(u8x8, 'cssr').decomp()
mzi_info = decomp_rlt[0]


def zero_init(mzi_info):
    for i in range(len(mzi_info['MZI_list'])):
        mzi_info['MZI_list'][i][2] = 0.0
        mzi_info['MZI_list'][i][3] = 0.0
    mzi_info['phase_angle_ori'] = np.zeros(N)
    mzi_info['phase_angle'] = np.zeros(N)


def bar_securing(mzi_info):
    adjustable_ids = []
    for i in range(len(mzi_info['MZI_list'])):
        if i not in adjustable_ids:
            mzi_info['MZI_list'][i][-1] = np.pi
            mzi_info['MZI_list'][i][-2] = np.pi


def xx_planting(mzi_info):
    for i in [20, 24]:
        mzi_info['MZI_list'][i][-1] = np.pi / 2


def yy_planting(mzi_info):
    for i in [20, 24]:
        mzi_info['MZI_list'][i][-1] = np.pi / 2
        mzi_info['MZI_list'][i][-2] = np.pi


def zz_planting(mzi_info):
    for i in [20, 24]:
        mzi_info['MZI_list'][i][-1] = np.pi
        mzi_info['MZI_list'][i][-2] = np.pi


def zx_planting(mzi_info):
    mzi_info['MZI_list'][20][-1] = np.pi
    mzi_info['MZI_list'][20][-2] = np.pi
    mzi_info['MZI_list'][24][-1] = np.pi / 2


def trial_planting(mzi_info, angle_list):
    change_list = [[1, -1], [4, -1], [5, -1]]
    for i in range(len(angle_list)):
        a, b = change_list[i]
        mzi_info['MZI_list'][a][b] = angle_list[i]


zero_init(mzi_info)
bar_securing(mzi_info)
zz_planting(mzi_info)


def post_selection(rlt_strkey):
    # print(rlt_strkey)
    label_list = ['|00101000>', '|00100100>', '|00011000>', '|00010100>']
    value = np.zeros(4)
    for i in range(4):
        value[i] = rlt_strkey.get(label_list[i], 0.0)
    return value


def estimate_energy(post_selection_rlt, basis='ZZ'):
    norm = post_selection_rlt / (post_selection_rlt.sum() + 1e-16)
    if basis == 'ZZ':  # ZZ basis
        return norm[0] + norm[3] - norm[1] - norm[2]
    if basis == 'ZI':  # ZI basis
        return norm[0] + norm[1] - norm[2] - norm[3]
    if basis == 'IZ':  # IZ basis
        return -norm[0] - norm[1] + norm[2] + norm[3]


def compute_process(mzi_info):
    cir = circuit.QumodeCircuit(
        nmode=8, init_state=[0, 0, 0, 1, 0, 1, 0, 0], name='test', cutoff=3, basis=True
    )  # basis=True, using state list
    for info in mzi_info['MZI_list']:
        cir.ps(inputs=info[2], wires=[info[0]])
        cir.bs_theta(inputs=np.pi / 4, wires=[info[0], info[1]])
        cir.ps(inputs=info[3], wires=[info[0]])
        cir.bs_theta(inputs=np.pi / 4, wires=[info[0], info[1]])
    rlt = cir(is_prob=True)
    rlt_strkey = dict()
    for key in rlt:
        rlt_strkey[str(key)] = np.abs(rlt[key].detach().numpy().squeeze()) ** 2

    # rltm = cir.measure(shots=10000)[0]
    # rltm_strkey = dict()
    # for key in rltm.keys():
    #     rltm_strkey[str(key)] = rltm[key]
    # return post_selection(rltm_strkey)

    return post_selection(rlt_strkey)


def estimate_eigval(angle_list):
    # to minimize
    zero_init(mzi_info)
    bar_securing(mzi_info)
    trial_planting(mzi_info, angle_list)

    zz_planting(mzi_info)
    post_selection_rlt = compute_process(mzi_info)
    e_zz = estimate_energy(post_selection_rlt)

    xx_planting(mzi_info)
    post_selection_rlt = compute_process(mzi_info)
    e_xx = estimate_energy(post_selection_rlt)

    yy_planting(mzi_info)
    post_selection_rlt = compute_process(mzi_info)
    e_yy = estimate_energy(post_selection_rlt)

    return (1 + e_zz - e_xx - e_yy) / 2

    # zz_planting(mzi_info)
    # post_selection_rlt = compute_process(mzi_info)
    # e_zi = estimate_energy(post_selection_rlt,basis='ZI')

    # xx_planting(mzi_info)
    # post_selection_rlt = compute_process(mzi_info)
    # e_ix = estimate_energy(post_selection_rlt,basis='IZ')

    # zx_planting(mzi_info)
    # post_selection_rlt = compute_process(mzi_info)
    # e_zx = estimate_energy(post_selection_rlt,basis='ZZ')

    # return (1 + e_zi + e_ix - e_zx) / 2


def target_function(angle_list):
    # to maximize
    return -estimate_eigval(angle_list)


def void_target():
    return -1


# %%

# %%
# 测试BO算法
print('=' * 50)
print('运行BO')
param_init = {'t1': 0, 't2': 0, 't3': 0}
bo_optimizer = OptimizerBayesian(target_func=void_target, param_init=param_init, random_state=0)

print('轮次', '\t', '参数值', '\t', ' ' * 31, '目标值')
for _ in range(100):
    # 待片上计算的参数
    p1 = bo_optimizer.param_suggest()
    # 模拟芯片计算过程，之后需要替换成接口
    f1 = -estimate_eigval(p1.tolist())  #  BO 内置用法是最大化目标；但是接下来打印时再添一个符号即可
    # 测试结果传回，更新param_dict
    bo_optimizer.param_register([p1], [f1])
    f1_to_print = -f1
    mark = '*' if bo_optimizer.best_target == f1 else ' '
    if f1_to_print < 0:
        print(bo_optimizer.iter, '\t', p1, '\t', f'{f1_to_print:8.10f} ', mark)
    else:
        print(bo_optimizer.iter, '\t', p1, '\t ', f'{f1_to_print:8.10f} ', mark)

print('BO结果：', estimate_eigval(list(bo_optimizer.best_param_dict.values())))

# %%
# 测试SPSA算法
print('=' * 50)
print('运行SPSA')
param_init = {'t1': np.random.randn(), 't2': np.random.randn(), 't3': np.random.randn()}
spsa_optimizer = OptimizerSPSA(target_func=void_target, param_init=param_init, random_state=0)

print('轮次', '\t', '参数值', '\t', ' ' * 31, '目标值')
for _ in range(200):
    # 待片上计算的两组参数
    p1, p2 = spsa_optimizer.param_suggest()
    # 以下两行模拟芯片计算过程，之后需要替换成接口
    f1 = estimate_eigval(p1.tolist())
    f2 = estimate_eigval(p2.tolist())
    # 测试结果传回，更新param_dict
    spsa_optimizer.param_register([p1, p2], [f1, f2])
    # spsa_optimizer.param_register(np.array([p1,p2]),np.array([f1,f2]))

    if f1 < 0:
        print(spsa_optimizer.iter, '\t', p1, '\t', f'{f1:8.10f} ')
    else:
        print(spsa_optimizer.iter, '\t', p1, '\t ', f'{f1:8.10f} ')
    if f2 < 0:
        print(spsa_optimizer.iter, '\t', p2, '\t', f'{f2:8.10f} ')
    else:
        print(spsa_optimizer.iter, '\t', p2, '\t ', f'{f2:8.10f} ')
    # print(spsa_optimizer.hyperparam['c']/(1+spsa_optimizer.iter)**spsa_optimizer.hyperparam['gamma'])

    if (f1 < -0.92) and (f2 < -0.92):
        # print("!!!进入步长减小阶段!!!")
        spsa_optimizer.hyperparam['c'] = 0.001
    elif (f1 < -0.999) and (f2 < -0.999):
        spsa_optimizer.hyperparam['c'] = 1e-4

print('SPSA结果：', estimate_eigval(list(spsa_optimizer.best_param_dict.values())))

# %%
# 测试Fourier级数方法
print('=' * 50)
print('运行Fourier级数法')
param_init = {'t1': np.random.randn(), 't2': np.random.randn(), 't3': np.random.randn()}
# param_init = dict(zip(param_init.keys(),np.random.randn(3)))
fourier_optimizer = OptimizerFourier(target_func=void_target, param_init=param_init, random_state=0, order=3, lr=0.1)
print('轮次', '\t', '参数值', '\t', ' ' * 31, '目标值')
for _ in range(50):
    # 待片上计算的两组参数
    param_array = fourier_optimizer.param_suggest()
    # 模拟芯片计算过程，之后需要替换成接口
    target = np.zeros(len(param_array))
    for i in range(len(param_array)):
        target[i] = estimate_eigval(param_array[i])

    # 测试结果传回，更新param_dict
    fourier_optimizer.param_register(param_array, target)
    p1 = np.array(list(fourier_optimizer.best_param_dict.values()))
    f1 = fourier_optimizer.best_target
    if f1 < 0:
        print(fourier_optimizer.iter, '\t', p1, '\t', f'{f1:8.10f} ')
    else:
        print(fourier_optimizer.iter, '\t', p1, '\t ', f'{f1:8.10f} ')

print('Fourier结果：', estimate_eigval(list(fourier_optimizer.best_param_dict.values())))
