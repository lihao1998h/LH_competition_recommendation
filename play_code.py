print('='*20, 'EDA开始', '='*20)


def easy_look(df):
    print("--------------------------Shape Of Data---------------------------------")
    print(df.shape)
    columns = df.columns.values.tolist()  # 获取所有的变量名
    print('变量列表：', columns)
    print('随机给几个样本')
    # df.head()  # 给前几个样本
    # df.tail()  # 给后几个样本
    print(df.sample(10))
    print("-------------------------------INFO--------------------------------------")
    print(df.info())
    numeric_features = df.select_dtypes(exclude='category')  # [np.number]
    categorical_features = df.select_dtypes(include=[np.object])  # [np.object]
    if numeric_features.shape[1] == 0:
        print('没有连续变量')
    else:
        print('连续变量的一些描述信息，如基本统计量、分布等。')
        print(df.describe())
    if categorical_features.shape[1] == 0:
        print('没有分类变量')
    else:
        print('所有变量的一些描述信息。')
        print(df.describe(include='all'))
    print('重复值统计（todo）')
    is_dup = False
    # idsUnique = len(set(train.Id)) # train['Id'].nunique()
    # idsTotal = train.shape[0]
    # idsDupli = idsTotal - idsUnique
    # print("There are " + str(idsDupli) + " duplicate IDs for " + str(idsTotal) + " total entries")
    # df.duplicated()

    print('--------------------缺失值统计--------------------------')
    is_missing = True
    ### 需要注意的是有些缺失值可能已经被处理过，可以用下条语句进行替换
    # Train_data['notRepairedDamage'].replace('-', np.nan, inplace=True)
    #
    # credit.isnull().sum()/float(len(credit))
    #
    #
    # bar(todo)
    # missing = train.isnull().sum()
    # missing = missing[missing > 0]
    # missing.sort_values(inplace=True)
    # missing.plot.bar()
    #
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print(missing_data)
    # for col in df.columns:
    #     print(col, df[col].isnull().sum())
    #
    if df.isnull().sum().sum() == 0:
        is_missing = False
    return is_dup, is_missing