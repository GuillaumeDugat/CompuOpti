import os
import pickle

from gurobipy import GRB

from build_model import build_model, get_instance


def compute_non_dominated_surface(
    model,
    data,
    max_assigned_name: str = "max_assigned",
    max_duration_name: str = "max_duration",
):
    model.Params.LogToConsole = 0  # muting the output of model.optimize()

    model.optimize()  # required to optimize to retrieve the variables in getVars
    max_duration = model.getVarByName(max_duration_name)
    max_assigned = model.getVarByName(max_assigned_name)

    non_dominated_solutions = []

    horizon = data["horizon"]  # max value max duration can take
    total_nb_projects = len(data["jobs"])  # max value max assigned can take

    epsilon_c_max_duration = horizon

    while epsilon_c_max_duration >= 0:
        print(epsilon_c_max_duration)

        epsilon_c_max_assigned = total_nb_projects
        next_epsilon_c_max_duration = 0

        model.addConstr(
            (max_duration <= epsilon_c_max_duration),
            name=f"{max_duration_name}_epsilon",
        )

        while epsilon_c_max_assigned >= 0:
            print(epsilon_c_max_assigned)

            model.addConstr(
                (max_assigned <= epsilon_c_max_assigned),
                name=f"{max_assigned_name}_epsilon",
            )

            model.optimize()

            model.remove(model.getConstrByName(f"{max_assigned_name}_epsilon"))

            if model.Status == GRB.OPTIMAL:
                solutions_variable = build_variables_dictionnary(model)
                non_dominated_solutions.append(solutions_variable)

                next_epsilon_c_max_duration = max(
                    solutions_variable[max_duration_name], next_epsilon_c_max_duration
                )
                epsilon_c_max_assigned = solutions_variable[max_assigned_name] - 1

            elif model.Status == GRB.INFEASIBLE:
                break
            elif model.Status == GRB.TIME_LIMIT:
                raise ValueError(
                    "Epsilon constraint method failed because of timeout. We recommend increasing the time limit."
                )

        model.remove(model.getConstrByName(f"{max_duration_name}_epsilon"))

        epsilon_c_max_duration = next_epsilon_c_max_duration - 1

    model.Params.LogToConsole = 1

    return non_dominated_solutions


def build_variables_dictionnary(model):
    variables = {}
    for v in model.getVars():
        variables[v.VarName] = v.X
    variables["objVal"] = model.objVal
    return variables


def save_non_dominated_surface(non_dominated_models, filename, folder="results"):
    pickle.dump(non_dominated_models, open(os.path.join(folder, filename), "wb"))


def load_non_dominated_surface(filename, folder="results"):
    return pickle.load(open(os.path.join(folder, filename), "rb"))


if __name__ == "__main__":
    instance_filename = "toy_instance.json"
    data = get_instance(instance_filename)
    model = build_model(data)
    non_dominated_models = compute_non_dominated_surface(model, data)
    print(non_dominated_models)
