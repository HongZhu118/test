"""开发者可参考本案例开发搜索组件入口程序Python代码
需要注意：

初始化参数
1. 算法参数（options）

启动搜索算法的参数
1. 优化问题（optimization_problem）

搜索算法运行时：
1. 搜索算法输出参数，需要组件send给仿真接口组件的逻辑在回调函数中实现：
    sim_req_cb
2. 仿真接口组件返回仿真结果（即本组件输入），
    仿真结果发给优化算法:search_alg.add_simulation_result(simulation_result)
"""

# 学术模式，测试内置问题

import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.append("./components/packages")

from components.packages import platgo as pg  # noqa
from components.packages.common.commons import AlgoResultType
from components.packages.platgo.algorithms import GA
if __name__ == "__main__":
    # 内置问题
    optimization_problem = {
        "name": "SOP_F20",
        "n_var": 6,
        "algoResultType": 0,
        "lower": "0",
        "upper": "1",
    }
    print(optimization_problem)

    pop_size = 100
    max_fe = 10000
    options = {}

    evol_algo = GA(
        pop_size=pop_size,
        options=options,
        optimization_problem=optimization_problem,
        control_cb=None,  # noqa
        max_fe=max_fe,
        name="GA-Thread",
        debug=True,
    )

    evol_algo.set_data_type(AlgoResultType.OBJECTIVE_SPACE)
    evol_algo.start()
    evol_algo.join()
    algo_result = evol_algo.get_external_algo_result()
    print("Done")


