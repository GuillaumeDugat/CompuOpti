from random import random, sample

import pandas as pd
import gurobipy as grb
from gurobipy import GRB


def find_pref_model(unacceptable, correct, satisfactory):
    model = grb.Model()

    # Define decision variables
    omega_1 = model.addVar(0.0, 1.0, vtype=GRB.CONTINUOUS, name="omega_1")
    omega_2 = model.addVar(0.0, 1.0, vtype=GRB.CONTINUOUS, name="omega_2")
    omega_3 = model.addVar(0.0, 1.0, vtype=GRB.CONTINUOUS, name="omega_3")
    th_1 = model.addVar(0.0, 1.0, vtype=GRB.CONTINUOUS, name="th_1")
    th_2 = model.addVar(0.0, 1.0, vtype=GRB.CONTINUOUS, name="th_2")
    eps = model.addVar(0.0, 1.0, vtype=GRB.CONTINUOUS, name="eps")

    # Add constraints
    model.addConstr(omega_1 + omega_2 + omega_3 == 1, name="normalisation")
    model.addConstrs(
        (
            f1 * omega_1 + f2 * omega_2 + f3 * omega_3 <= th_1 - eps
            for (f1, f2, f3) in unacceptable
        ),
        name="unacceptable_max",
    )
    model.addConstrs(
        (
            f1 * omega_1 + f2 * omega_2 + f3 * omega_3 <= th_2 - eps
            for (f1, f2, f3) in correct
        ),
        name="correct_max",
    )
    model.addConstrs(
        (
            f1 * omega_1 + f2 * omega_2 + f3 * omega_3 >= th_1 + eps
            for (f1, f2, f3) in correct
        ),
        name="correct_min",
    )
    model.addConstrs(
        (
            f1 * omega_1 + f2 * omega_2 + f3 * omega_3 >= th_2 + eps
            for (f1, f2, f3) in satisfactory
        ),
        name="satisfactory_min",
    )

    # Objective
    model.setObjective(eps, GRB.MAXIMIZE)

    model.optimize()

    return omega_1.x, omega_2.x, omega_3.x, th_1.x, th_2.x


def normalize_models(instance, models):
    max_objVal = sum([job["gain"] for job in instance["jobs"]])
    max_max_assigned = len(instance["jobs"])
    max_max_duration = instance["horizon"]
    res = []

    for model in models:
        f1 = model["objVal"] / max_objVal
        f2 = 1 - model["max_assigned"] / max_max_assigned
        f3 = 1 - model["max_duration"] / max_max_duration
        res.append((f1, f2, f3))

    return res


def convert_examples(examples: pd.DataFrame, instance):
    unacceptable = normalize_models(
        instance,
        (
            examples[examples["status"] == "unacceptable"]
            .drop("status", axis=1)
            .to_dict("records")
        ),
    )
    correct = normalize_models(
        instance,
        (
            examples[examples["status"] == "correct"]
            .drop("status", axis=1)
            .to_dict("records")
        ),
    )
    satisfactory = normalize_models(
        instance,
        (
            examples[examples["status"] == "satisfactory"]
            .drop("status", axis=1)
            .to_dict("records")
        ),
    )
    return unacceptable, correct, satisfactory


def simulate_preferences(instance, models, sample_freq=None):
    models = normalize_models(instance, models)
    inacceptable_models = []
    correct_models = []
    satisfying_models = []
    while (
        len(inacceptable_models) == 0
        or len(correct_models) == 0
        or len(satisfying_models) == 0
    ):
        w1, w2, w3, th1, th2 = get_random_weights()
        inacceptable_models = []
        correct_models = []
        satisfying_models = []

        for truc in models:
            f1, f2, f3 = truc
            model_score = w1 * f1 + w2 * f2 + w3 * f3
            if model_score < th1:
                inacceptable_models.append((f1, f2, f3))
            elif model_score < th2:
                correct_models.append((f1, f2, f3))
            else:
                satisfying_models.append((f1, f2, f3))

    print(f"Simulated weights:\t({w1:.2f}, {w2:.2f}, {w3:.2f})")
    print(f"Simulated thresholds:\t({th1:.2f}, {th2:.2f})")

    if sample_freq is None:
        sample_freq = random()

    sample_inacceptable_models = sample(
        inacceptable_models, max(1, int(sample_freq * len(inacceptable_models)))
    )
    sample_correct_models = sample(
        correct_models, max(1, int(sample_freq * len(correct_models)))
    )
    sample_satisfying_models = sample(
        satisfying_models, max(1, int(sample_freq * len(satisfying_models)))
    )

    return sample_inacceptable_models, sample_correct_models, sample_satisfying_models


def get_random_weights():
    w1 = random()
    w2 = random()
    w3 = random()
    th1 = random()
    th2 = random()
    w1, w2, w3 = normalize_weights(w1, w2, w3)
    return w1, w2, w3, min(th1, th2), max(th1, th2)


def normalize_weights(w1, w2, w3):
    sum_w = w1 + w2 + w3
    return w1 / sum_w, w2 / sum_w, w3 / sum_w


def order_solutions(instance, models, params):
    w1, w2, w3 = params[:3]
    th1, th2 = params[3:]
    objVal = [model["objVal"] for model in models]
    max_assigned = [model["max_assigned"] for model in models]
    max_duration = [model["max_duration"] for model in models]
    normalized = normalize_models(instance, models)
    score = [w1 * f1 + w2 * f2 + w3 * f3 for (f1, f2, f3) in normalized]
    res = pd.DataFrame(
        {
            "objVal": objVal,
            "max_assigned": max_assigned,
            "max_duration": max_duration,
            "score": score,
        }
    )
    res = res.sort_values("score", ascending=False)
    res["status"] = res["score"].apply(
        lambda x: "unacceptable"
        if x < th1
        else ("correct" if x < th2 else "satisfactory")
    )

    return res


def preferences(instance, solutions, unacceptable, correct, satisfactory):
    params = find_pref_model(unacceptable, correct, satisfactory)
    print(f"Weights:\t({params[0]:.2f}, {params[1]:.2f}, {params[2]:.2f})")
    print(f"Thresholds:\t({params[3]:.2f}, {params[4]:.2f})")
    return order_solutions(instance, solutions, params)
