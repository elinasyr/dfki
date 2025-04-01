import pandas as pd
from sklearn.preprocessing import LabelEncoder


def plan_from_trajectories(t, keep_only_order=False):
    plan = []
    prev_goal = None

    for goal in t["goal_status"]:
        if goal != prev_goal:
            plan.append(goal)
            prev_goal = goal
    final_plan = list(dict.fromkeys(plan))
    if keep_only_order:
        plan_by_numbers = []
        for stop in final_plan:
            if stop[:6] == "moving":
                plan_by_numbers.append(int(stop[-1]))
        return plan_by_numbers
    return final_plan


plan_mapping = {
    "[1, 6, 4, 2, 3, 5]": 1,
    "[3, 5, 4, 1, 6, 2]": 2,
    "[5, 4, 6, 3, 1, 2]": 3,
    "[2, 6, 1, 5, 4, 3]": 4,
    "[6, 5, 2, 4, 1, 3]": 5,
}


def get_data():

    data = pd.DataFrame()
    valid_plans = [
        1,
        2,
        3,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        22,
        25,
        32,
        35,
        40,
        55,
    ]
    for i in valid_plans:
        temp = pd.read_csv(f"./output_robot5data/out{i}.csv")
        temp = temp[["goal_status"]]  # 'idle'
        temp.drop_duplicates(subset=["goal_status"], keep="first", inplace=True)
        if temp.shape[0] < 10:
            continue
        plan = plan_from_trajectories(temp, keep_only_order=True)
        if str(plan) not in list(plan_mapping.keys()):
            continue
        temp["id"] = i
        temp["plan"] = plan_mapping[str(plan)]
        if temp.iloc[-1].goal_status == "(unknown)":
            temp = temp[:-1]

        data = pd.concat([data, temp])
    data.reset_index(drop=True, inplace=True)
    enc = LabelEncoder()

    data["goal_status"] = enc.fit_transform(data[["goal_status"]])

    return data
