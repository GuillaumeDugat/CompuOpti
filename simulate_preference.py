from random import random, sample


def simulate_preference(instance, non_dominated_models, sample_freq=None):
    w1, w2, w3 = get_random_weights()
    inacceptable_models = []
    correct_models = []
    satisfying_models = []
    max_objVal = sum([job["gain"] for job in instance["jobs"]])
    max_max_duration = instance["horizon"]
    max_max_assigned = len(instance["jobs"])

    for model in non_dominated_models:
        normalized_obj_val = model["objVal"] / max_objVal
        normalized_max_duration = model["max_duration"] / max_max_duration
        normalized_max_assigned = model["max_assigned"] / max_max_assigned
        model_score = (
            w1 * normalized_obj_val
            + w2 * normalized_max_duration
            + w3 * normalized_max_assigned
        )
        if model_score < 1 / 3:
            inacceptable_models.append(
                (normalized_obj_val, normalized_max_duration, normalized_max_assigned)
            )
        elif model_score >= 1 / 3 and model_score < 2 / 3:
            correct_models.append(
                (normalized_obj_val, normalized_max_duration, normalized_max_assigned)
            )
        elif model_score >= 2 / 3:
            satisfying_models.append(
                (normalized_obj_val, normalized_max_duration, normalized_max_assigned)
            )

    if sample_freq is None:
        sample_freq = random()

    sample_inacceptable_models = sample(
        inacceptable_models, int(sample_freq * len(inacceptable_models))
    )
    sample_correct_models = sample(
        correct_models, int(sample_freq * len(correct_models))
    )
    sample_satisfying_models = sample(
        satisfying_models, int(sample_freq * len(satisfying_models))
    )

    return sample_inacceptable_models, sample_correct_models, sample_satisfying_models


def get_random_weights():
    w1 = random()
    w2 = random()
    w3 = random()
    w1, w2, w3 = normalize_weights(w1, w2, w3)
    return w1, w2, w3


def normalize_weights(w1, w2, w3):
    sum_w = w1 + w2 + w3
    return w1 / sum_w, w2 / sum_w, w3 / sum_w


if __name__ == "__main__":
    from non_dominated_surface import load_non_dominated_surface
    from build_model import get_instance

    instance = get_instance("toy_instance.json")
    surface = load_non_dominated_surface("toy_instance.pkl")
    print(simulate_preference(instance, surface))
