import copy
from typing import Dict, List

import numpy as np
from bayes_opt import BayesianOptimization, UtilityFunction


hyperparam_spsa = {
    'a': 1e-1,
    'c': 1e-2,
    'A': 200,
    'nepoch': 2000,
    'alpha': 0.602,
    'gamma': 0.101
}

class Optimizer(object):
    def __init__(self, target_func, param_init, random_state = 0):
        """初始化内容包括：
        1）target_func：待优化目标函数（最小化），可以用一个空的函数指代，
           因为实际运行时register函数不需要调用它，只需要target结果给到register函数输入
        2）param_init：目标函数的参数字典
        3）random_state：随机数种子"""
        self.target_func = target_func
        self.param_dict = copy.deepcopy(param_init)
        self.random_state = random_state
    def __str__(self) -> str:
        return "Optimizer"

class OptimizerBayesian(Optimizer):
    def __init__(self, target_func, param_init, random_state = 0):
        """初始化内容包括：
        1）待优化目标函数（func_to_maximize改为最大化）;无实质作用
        2）pbounds（每个参数都是0到2pi）
        3）BO的acquisition function（这里使用默认的UCB）
        4）随机数种子
        5) 最佳参数方案及目标值（最大化目标值）
        """
        super().__init__(target_func, param_init, random_state)
        def func_to_maximize(**param_dict: Dict) -> float:
            return -self.target_func(**param_dict)
        self.pbounds = self.gen_pbounds()
        self.optimizer = BayesianOptimization(
            f = func_to_maximize,
            pbounds = self.pbounds,
            random_state = self.random_state
        )
        self.util = UtilityFunction(
                    kind='ucb',
                    kappa=2.576,
                    xi=0.0,
                    kappa_decay=1,
                    kappa_decay_delay=0)
        self.best_param_dict = copy.deepcopy(self.param_dict)
        self.best_target = -np.inf
        self.iter = 0

    def gen_pbounds(self) -> Dict:
        """用于生成初始化的pbounds"""
        pbounds = dict()
        for key in self.param_dict.keys():
            pbounds[key] = (0,np.pi*2)
        return pbounds

    def param_suggest(self) -> np.ndarray:
        """用于产生待测试的参数方案"""
        self.util.update_params()
        x_probe = self.optimizer.suggest(self.util)
        x = self.optimizer._space._as_array(x_probe) # a list
        param_array = np.asarray(x).reshape(1,-1)
        return param_array

    def param_register(self, param_array: np.ndarray, target: float) -> None:
        """将参数方案与实施后的目标值传回
        param_dict: 施加在芯片上的参数
        target：芯片测量的目标值结果，注意是最大化目标"""
        for i in range(len(param_array)):
            x = param_array[i]
            param_dict = dict(zip(self.param_dict.keys(), x))
            if self.optimizer._space._constraint is None:
                self.optimizer._space.register(x, target[i])
            else:
                constraint_value = self.optimizer._space._constraint.eval(**param_dict)
                self.optimizer._space.register(x, target[i], constraint_value)

            if target[i] > self.best_target:
                self.best_param_dict = copy.deepcopy(param_dict)
                self.best_target = target[i]
        self.iter += 1

    def run(self, nstep: int) -> List:
        for _ in range(nstep):
            p1 = self.param_suggest()
            # BO 内置用法是最大化目标；但是接下来打印时再添一个符号即可
            f1 = [-self.target_func(p1)]
            self.param_register(p1, f1)
        return list(self.best_param_dict.values())


class OptimizerSPSA(Optimizer):
    def __init__(self, target_func, param_init, hyperparam = hyperparam_spsa, random_state = 0):
        """初始化内容包括：
        1）SPSA的超参数，可参见https://www.jhuapl.edu/spsa/Pages/MATLAB.htm
        2）随机数种子
        3）最佳参数方案及目标值（最小化目标值）
        4）待优化目标函数
        """
        super().__init__(target_func, param_init, random_state)
        self.random_state_ori = np.random.get_state()
        np.random.seed(self.random_state)
        self.hyperparam = hyperparam
        self.iter = 0
        self.nparam = len(param_init)
        self.best_param_dict = copy.deepcopy(self.param_dict)
        self.best_target = np.inf

    def param_suggest(self) -> np.ndarray:
        """用于产生待测试的参数方案"""
        tmp_param = np.asarray(list(self.param_dict.values()))
        delta_lr = self.hyperparam['c'] / (1+self.iter)**self.hyperparam['gamma']
        delta = (np.random.randint(0, 2, self.nparam) * 2 - 1) * delta_lr
        param_array = np.zeros((2, self.nparam))
        param_array[0] = tmp_param - delta
        param_array[1] = tmp_param + delta
        return param_array

    def param_register(self, param_array: np.ndarray, target: np.ndarray) -> None:
        """将参数方案与实施后的目标值传回
        param_array: 施加在芯片上的参数；可以用长度2的列表代替行数2的数组
        target：芯片测量目标值结果；可以用长度2的列表代替行数2的数组"""
        assert(len(param_array)==2)
        assert(len(target)==2)
        param_lr = self.hyperparam['a'] / (1+self.iter+self.hyperparam['A'])**self.hyperparam['alpha']
        param1 = param_array[0]
        param2 = param_array[1]
        target1 = target[0]
        target2 = target[1]
        delta = param2 - param1
        grad = (target2-target1) / delta
        param_new = 0.5*(param1+param2) - param_lr * grad
        self.param_dict = dict(zip(self.param_dict.keys(),param_new))
        self.iter += 1

        if target1 < self.best_target:
            self.best_param_dict = dict(zip(self.param_dict.keys(), param1))
            self.best_target = target1

        if target2 < self.best_target:
            self.best_param_dict = dict(zip(self.param_dict.keys(), param2))
            self.best_target = target2

    def ori_random_state(self) -> None:
        """SPSA结束之后，将随机状态回退到设置随机数种子之前"""
        np.random.set_state(self.random_state_ori)

    def run(self, nstep: int) -> List:
        for _ in range(nstep):
            p1, p2 = self.param_suggest()
            f1 = self.target_func(p1)
            f2 = self.target_func(p2)
            self.param_register([p1,p2], [f1,f2])
            if (f1 < -0.92) and (f2 < -0.92):
                self.hyperparam['c'] = 0.001
            elif (f1 < -0.999) and (f2 < -0.999):
                self.hyperparam['c'] = 1e-4
        return list(self.best_param_dict.values())


class OptimizerFourier(Optimizer):
    def __init__(self, target_func, param_init, R = 5, lr = 0.1, random_state = 0):
        """初始化内容包括：
        1）傅里叶级数的个数R
        2）傅里叶级数对响应的拟合 求解Au=v得到
        3）傅里叶级数对导数的拟合
        4）学习率
        """
        super().__init__(target_func, param_init, random_state)
        self.iter = 0
        self.R = R
        self.nparam = len(param_init)
        self.best_param_dict = copy.deepcopy(self.param_dict)
        self.best_target = np.inf
        self.lr = lr
        self.A = self.gen_A()
        self.u = np.zeros((2*R+1)*self.nparam)
        self.iter = 0

    def gen_A(self) -> np.ndarray:
        A = np.zeros((2*self.R+1, 2*self.R+1))
        mu = np.arange(2*self.R+1)
        x_mu = 2*np.pi/(2*self.R+1)*(mu-self.R)
        A[:,0] = 1
        A[:,1:self.R+1] = np.cos(x_mu.reshape(-1,1)@np.arange(1,self.R+1).reshape(1,-1))
        A[:,self.R+1:2*self.R+2] = np.sin(x_mu.reshape(-1,1)@np.arange(1,self.R+1).reshape(1,-1))
        return A

    def param_suggest(self) -> np.ndarray:
        """原理是每次只调1个寻优参数，剩余参数固定，对该寻优参数进行2*R+1个点的尝试"""
        tmp_param = np.asarray(list(self.param_dict.values()),dtype=float).reshape(1,-1)
        mu = np.arange(2*self.R+1)
        varied_param = 2*np.pi/(2*self.R+1)*(mu-self.R)
        param_array = np.repeat(tmp_param,self.nparam*(2*self.R+1),axis=0)
        for param_id in range(self.nparam):
            param_array[param_id*(2*self.R+1):(param_id+1)*(2*self.R+1),param_id] = varied_param
        return  param_array

    def param_register(self, param_array: np.ndarray, target: np.ndarray):
        """与另外两种优化算法不同，Fourier中，
        通过param_register修改当前param_dict，
        让param_suggest产生新的点"""
        assert(len(param_array)==(2*self.R+1)*self.nparam)
        assert(len(target)==(2*self.R+1)*self.nparam)
        # 求解线性方程组得出组合系数
        param = np.asarray(list(self.param_dict.values()))
        for param_id in range(self.nparam):
            idx1 = param_id*(2*self.R+1)
            idx2 = (1+param_id)*(2*self.R+1)
            self.u[idx1:idx2] = np.linalg.solve(self.A,target[idx1:idx2])

        # 根据组合系数计算当前位置处的偏导数
        grad = np.zeros(self.nparam)
        for param_id in range(self.nparam):
            theta = param[param_id]
            idx = 1+param_id*(2*self.R+1)
            grad[param_id] = -(np.arange(1,self.R+1)*np.sin(theta*np.arange(1,self.R+1)))@\
                            self.u[idx:self.R+idx]+(np.arange(1,self.R+1)*\
                            np.cos(theta*np.arange(1,self.R+1)))@self.u[self.R+idx:self.R*2+idx]
        param_new = param - self.lr * grad
        self.param_dict = dict(zip(self.param_dict.keys(),param_new))
        if target.min() < self.best_target:
            self.best_target = target.min()
            self.best_param_dict = dict(zip(self.param_dict.keys(),param_array[target.argmin()]))
        self.iter += 1

    def run(self, nstep: int) -> List:
        for _ in range(nstep):
            param_array = self.param_suggest()
            target = np.zeros(len(param_array))
            for i in range(len(param_array)):
                target[i] = self.target_func(param_array[i])
            self.param_register(param_array, target)
        return list(self.best_param_dict.values())
