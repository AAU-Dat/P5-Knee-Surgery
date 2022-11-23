import pandas as pd
import matplotlib
from matplotlib import style
from sklearn import linear_model
import matplotlib.pyplot as plt
matplotlib.use('Agg')
style.use('seaborn')

acl_k_path = "./Test ran on 23-Nov-2022 at 00:53/models/ACL_k/"
acl_epsr_path = "./Test ran on 23-Nov-2022 at 00:53/models/ACL_epsr/"


def create_graph(target, path):
    data = pd.read_csv(f"{path}test-data.csv")
    true_value = data['Expected']
    predicted_value = data['Predicted'].str.strip('[').str.strip(']').astype(float)
    values = linear_model.LinearRegression()
    values.fit(true_value.values.reshape(-1, 1), predicted_value)
    regression_line = values.predict(true_value.values.reshape(-1, 1))

    plt.scatter(true_value, predicted_value, label='Data Points', c='r', alpha=0.6, s=5)
    plt.plot(true_value, regression_line, label='Best Fit Line', c='b', linewidth=2)
    plt.title(f"Expected and Actual {target} Values")
    plt.xlabel('Actual Values')
    plt.ylabel('Expected Values')
    plt.legend()

    plt.savefig(f"{path}scatter-plot.png")
    plt.close()


create_graph('ACL_k', acl_k_path)
create_graph('ACL_epsr', acl_epsr_path)

print("Completed")
