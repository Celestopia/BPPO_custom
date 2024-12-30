# 数据生成
使用`generate_burgers.py`生成数据。训练集包含2000个轨迹，测试集包含100个轨迹。

每条轨迹代表一个演化过程。系统一共根据burgers方程演化10000步，每步对应的物理时间长度是0.0001s。演化完毕后，等距采样10个时间点加入数据集中，因此每条轨迹的长度为10。由于一共划分了128个网格点，所以state和action的维度为128，亦即state_trajectory、action_trajectory的形状均为(10, 128)。

在每条轨迹的演化过程中，action每1000步更新一次，亦即在1000个时间步内（如1000-2000步），action保持不变。

初始状态、控制信号的随机生成方法同diffphycon。

以上对应的函数为`generate_burgers.py`中的`load_burgers_data_sampled`和`get_sampled_trajectory`，细节见函数实现，注释已补充完善。


# 数据加载
在`main_draft.ipynb`中，训练集、测试集分别从不同的.pkl文件中加载，所包含键值对以及数据形状均已打印出来。

具体而言，`burgers_data_dict`是一个字典，其中
- `burgers_data_dict['observations']`包含了state trajectory，形状为(2000, 10, 128)，其中2000是轨迹的数量，10是每个轨迹的长度，128是state的维度。
- `burgers_data_dict['actions']`包含了action trajectory，形状为(2000, 10, 128)，其中2000是轨迹的数量，10是每个轨迹的长度，128是action的维度。
- `burgers_data_dict['Y_f']`包含了各个轨迹的最终状态，形状为(2000, 128)，其中2000是轨迹的数量，128是state的维度。
- `burgers_data_dict['next_observations']`包含了next state trajectory，形状为(2000, 10, 128)。

`test_burgers_data_dict`同上，只是样本数量从2000改为100。

原始数据集中只有state和action数据，无reward信息，reward在`main_draft.ipynb`中计算得到。对于每个时间步，state reward为当前state与本条轨迹的final state的均方差的相反数，action reward为action和零向量的L2距离的相反数，两个reward加权求和得到总reward，权重为超参数`reward_ratio`（设为0.1）。

对于每条轨迹，在其内遍历时间点，将final state拼接到每个时间步的state上，并加上一个代表当前时间步数的标量（0, 1, 2, ..., 9），由此得到257维的`concat_state`。对所有轨迹进行如上操作，并展平0, 1维，则新的state数据集形状为(20000, 257)，其中20000是采样时间点总数，257是拼接后的state维度。

actions数据集也进行展平，新的actions数据集形状为(20000, 128)。


# 模型构建
- `value`：学习每个状态对应的值函数`V(s)`，底层为MLP，输入维度为257，输出维度为1。值函数对应的应该是return（reward的加权期望）而非reward。但diffphycon中直接使用当前state的reward作为目标value，学习这个值，这里沿用此做法。
- `Q_bc`：学习状态动作值函数`Q(s, a)`，底层为MLP，输入维度为257+128，输出维度为1。Q网络学习的应该是Q值，但diffphycon中直接使用当前state的reward与next state的reward值的加权和作为目标Q值，这里沿用此做法。
- `bc`：策略网络，根据当前state给出action（对应方法为`.select_action(s)`。底层为MLP，输入维度为257，输出维度为256，代表128维的高斯分布，即action的均值和方差。训练时，使用概率分布计算loss；预测时，`.select_action()`都传入`is_sample=False`，即使用均值作为输出的action。
- `bppo`：策略网络，根据当前state给出action（对应方法为`.select_action(s)`。底层为MLP，同`bc`。`bppo`的网络参数使用训练好的`bc`的网络参数初始化，然后进一步用BPPO的loss训练。


# 模型训练
各个模型都使用`.update(memory)`方法训练，底层逻辑为每次在memory（即数据集）中随机采样`batch_size`个样本（此处为64），计算loss，更新网络参数。例如训练数据集里state的形状为(20000, 257)，则每次都是从20000个样本中随机采样64个。每个样本对应某个轨迹中的某个时间点，包含state、action、reward、next_state信息。由于state中已经拼接了final state，因此各个样本间较为独立，可以混在一起更新网络参数，而不用担心不同的final state会带来不同的更新方向。


# 模型评价

遍历测试集的100条轨迹，根据每条轨迹的初始state，计算含控制信号（用bc或bppo根据state计算出）时10000个时间步后的最终状态。其中控制信号每1000步更新一次。

作为对照，同时模拟不含控制信号（按照burgers方程自然演化）时10000个时间步后的最终状态。

最终在一个图中画出`target_state`（数据集生成时，随机控制信号作用下的最终状态）、`final_state`（含控制信号的最终状态）和`final_state_`（自然演化的最终状态）的曲线，并分别计算`final_state`、`final_state_`与`target_state`之间的均方误差。


# 各文件作用
- `generate_burgers.py`：生成burgers数据集。最后被注释掉的main函数可以生成一条burgers演化轨迹，并展示动画。
- `main_draft.ipynb`：加载训练集和测试集，并训练模型，评价模型。

- `bppo.py`：BC和BPPO的模型实现。
- `critic.py`：值函数和Q函数学习器的实现。
- `memory.py`：训练中所用离线数据集。
- `net.py`：基础MLP网络的实现。
- `utils.py`：不太重要的辅助函数。


