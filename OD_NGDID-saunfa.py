import numpy as np
from sklearn.datasets import load_breast_cancer
import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler


# 加载数据集
data = load_breast_cancer()
# 将数据转为 DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)

df['class'] = data.target  # 添加目标列（0 为 malignant，1 为 benign）
df['class'] = df['class'].apply(lambda x: 'benign' if x == 1 else 'malignant')
# 假设数据已经读取为 DataFrame `df`
# 数据集包括了 699 个数据样本，9 个数值型属性和一个类别标签

# 1. 将数据集划分为 benign 类和 malignant 类
benign_samples = df[df['class'] == 'benign']
malignant_samples = df[df['class'] == 'malignant']


# 2. 随机删除很多个 malignant 类样本（占比约 83%）36
malignant_reduced = malignant_samples.sample(frac=0.17, random_state=42)
# 3. 从 benign 类中挑选357 个样本
benign_selected = benign_samples

# 4. 合并 benign_selected 和 malignant_reduced 为新的数据集
processed_df = pd.concat([benign_selected, malignant_reduced], ignore_index=True)
processed_df = processed_df.drop(columns=['class'])

# 初始化 MinMaxScaler
scaler = MinMaxScaler()

# 对所有列进行归一化
processed_df[:] = scaler.fit_transform(processed_df)

processed_df.head()

#两个标记位
Golbel_NKGH = 0
Golbel_is = 0
Golbel_is2=0
Golbel_WAD=0
def OD_NGDID(U, A, delta, mu):
    # 初始化：将离群点集合 O 置为空集
    O = []


    # 初始化属性重要度与权重
    sig = {}
    weight = {}

    # 步骤 2-8：计算每个属性的邻域区分指数、重要度及权重
    for a in A:
        # 计算邻域覆盖、邻域知识粒度、区分指数等
        # NKG = calculate_NKG(a, U, delta)
        # H = calculate_H(NKG,len(U))
        #
        # # 邻域粒度区分指数（定义公式）
        # NKGH = NKG * H
        NKGH=0
        # 属性重要度 sig(a) 和属性权重 weight(a) 的计算
        sig[a] = calculate_importance(NKGH,a)
        weight[a] = 1+np.sqrt(sig[a]) / sum(sig.values()) if sig[a]!=0 else 2/3 # 权重为重要度归一化
    #统计总weight
    max=0
    for i in weight.values():
        max+=i
    # 数值型属性集处理
    AN = [a for a in A if is_number("{}".format(U[1][a]))]
    DOFN = {}
    distlie = {}
    for xi in range(len(U)):
        distt=[]
        for yi in range(len(U)):
            dist = 0
            if xi != yi:
                # 计算 xi 和 yi 的数值属性加权距离
                for a in AN:
                    dist += np.sqrt(weight[a] * calculate_numeric_distance(U[xi], U[yi], a))
                # print(dist)
            distt.append(dist)
        # 数值属性上的距离离群因子 DOFN(xi)
        distlie[xi]=distt
    # print(distlie)
    for xi in range(len(U)):
        DOFN[xi] = calculate_DOFN(xi,distlie,U)

    # 符号型属性集处理
    AC = [a for a in A if not is_number("{}".format(U[1][a]))]
    DOFC = {}
    for xi in range(len(U)):
        DOFC[xi] = 0
        coun=0
        for yi in range(len(U)):
            if xi != yi:
                # 计算 xi 和 yi 的符号型属性加权距离
                dist = 0
                for a in AC:
                    dist += weight[a] * calculate_symbolic_distance(U[xi], U[yi], a)
            if dist>0.5:
                # np.random.uniform(0, max)
                coun+=1
        # 符号属性上的距离离群因子 DOFC(xi)
        DOFC[xi] =coun/len(U)

    # 计算最终的离群因子 WDOF(xi)
    for xi in range(len(U)):
        WDOF_xi = DOFN[xi] + DOFC[xi]  # 定义公式
        if WDOF_xi > mu:
            O.append(xi)

    return O

#{x:[]}
def calculate_DOFN(xid,distlie,U):
    global Golbel_WAD,Golbel_is2
    count=0
    sum=0
    if Golbel_is2==0:
        for xi,distt in distlie.items():
            for i in distt:
                sum+=i
        WADist=sum/(len(U)**2)
        Golbel_WAD=WADist
        Golbel_is2=1
    else:
        WADist=Golbel_WAD
    for xi,distt in distlie.items():
        if xid==xi:
            for i in distt:
                if i>WADist:
                    count+=1
            break
    return count/len(U)



# 计算邻域知识粒度 NKG
def calculate_NKG(a, U, delta):
    # 初始化邻域知识粒度 NKG
    n = len(U)
    sum_NKG = 0.0
    num=0
    for xi in U:
        # 计算邻域 n_{B}^ε(x)
        neighborhood = [y for y in U if calculate_HEOM(xi, y, a) <= delta]
        # 计算 NKG 累加项 (1 - |U| / |n_B^ε(x)|)
        num+=len(neighborhood)
        if len(neighborhood) > 0:
            sum_NKG += (1 - len(neighborhood)/n)
    # 最终的邻域知识粒度 NKG
    NKG = sum_NKG / n
    # print(num)
    return NKG


def calculate_HEOM(x, y, a):
    """
    计算对象 x 和 y 在属性集 a 上的异构欧式距离 HEOM
    """
    dist = 0
    if isinstance(a, list):
        for j in a:
            if is_number('{}'.format(x[j])):
                # 数值型属性，使用归一化欧式距离
                dist += (x[j] - y[j]) ** 2
            else:
                # 符号型属性，根据是否相同计算距离
                dist += 0 if x[j] == y[j] else 1
    else:
        if is_number('{}'.format(x[a])):
            # 数值型属性，使用归一化欧式距离
            dist += (x[a] - y[a]) ** 2
        else:
            # 符号型属性，根据是否相同计算距离
            dist += 0 if x[a] == y[a] else 1
    return np.sqrt(dist)


def is_number(s):

    return bool(re.fullmatch(r'[-+]?[0-9]*\.?[0-9]+', s))


# 计算邻域区分指数 H
def calculate_H(NKG,n):
    """
       计算邻域区分指数 H^δ(B)

       参数:
       NKG: float, 由 calculate_NKG 计算得到的邻域知识粒度
       n: int, 数据集 U 的大小

       返回:
       H: float, 邻域区分指数
       """
    # 根据公式计算邻域区分指数 H^δ(B)
    H = np.log(n ** 2 / (NKG * n))  # 等价于 log(n / NKG)
    return H

# 计算属性重要度 sig(a)
def calculate_importance(NKGH,a):
    global Golbel_is,Golbel_NKGH
    """
        计算属性 a 的重要度 Sig_delta(a)

        参数:
        A: list, 属性集合
        a: str, 当前属性名
        U: list, 数据集中的对象集合，每个对象是一个字典
        delta: float, 邻域半径

        返回:
        importance: float, 属性 a 的重要度
        """
    # 计算 NKGH^δ(A): 完整属性集的邻域粒度区分指数
    if Golbel_is==0:
        NKGH_A = calculate_NKGH(A, U, delta)
        Golbel_is=1
        Golbel_NKGH=NKGH_A
    else:
        NKGH_A=Golbel_NKGH

    # 计算 NKGH^δ(A - {a}): 去掉属性 a 的属性集的邻域粒度区分指数
    A_without_a = [attr for attr in A if attr != a]
    NKGH_A_minus_a = calculate_NKGH(A_without_a, U, delta)

    # 使用公式计算属性重要度 Sig_delta(a)
    importance = (NKGH_A - NKGH_A_minus_a) / (NKGH_A + NKGH_A_minus_a)
    # print(importance)
    return importance


def calculate_NKGH(attributes, U, delta):
    """
    计算邻域粒度区分指数 NKGH^δ(attributes)

    参数:
    attributes: list, 属性集合
    U: list, 数据集中的对象集合，每个对象是一个字典
    delta: float, 邻域半径

    返回:
    NKGH: float, 邻域粒度区分指数
    """
    # 根据属性集合计算邻域知识粒度 NKG
    NKG = calculate_NKG(attributes, U, delta)
    # 计算邻域区分指数 H，假设 n 是 U 的大小
    H = calculate_H(NKG, len(U))
    # 邻域粒度区分指数 NKGH = NKG * H
    NKGH = NKG * H
    # print(H)
    # print(NKG)
    return NKGH


# 计算数值型属性加权距离
def calculate_numeric_distance(xi, yi, a):
    # 使用数值型加权距离公式
    sm=0
    sm=(xi[a]-yi[a])**2
    return sm


# 计算符号型属性加权距离
def calculate_symbolic_distance(xi, yi, a):
    # 使用符号型加权距离公式
    if xi[a]==yi[a]:
        return 0
    else:
        return 1


# 判断属性 a 是否为数值型
def is_numeric(a):
    # 根据属性类型判断
    pass



if __name__ == '__main__':
    #超参delt，mu一个 if dist>np.random.uniform(0,max):

    # 去掉标签列，保留数值型属性列
    numeric_attributes = processed_df.columns.tolist()

    # 定义 A 为数值属性的列表
    A = numeric_attributes

    # 将数据集每行转为字典形式组成的列表 U
    U = processed_df[numeric_attributes].to_dict(orient='records')

    # 示例：调用 OD_NGDID 函数
    delta = 0.4 # 假设的 delta 值
    mu = 0.2  # 假设的 mu 值

    # 调用函数
    outliers = OD_NGDID(U, A, delta, mu)
    print("Detected outliers:", len(outliers))