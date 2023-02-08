import json
from random import randint, choice, choices, sample
import numpy as np

alphabet = "abcdefghijklmnopqrstuvwxyz"


def create_random_instance(
    horizon=None,
    nb_skills=None,
    nb_workers=None,
    nb_jobs=None,
    save=False,
    filepath=None,
):
    if horizon is None:
        # horizon of at least 5 days, no more than 40, creates exponentially more small projects than big ones
        horizon = min(5 + int(np.random.exponential(20)), 40)
    if nb_skills is None:
        nb_skills = randint(1, 10)
    if nb_workers is None:
        nb_workers = randint(1, 10)
    if nb_jobs is None:
        nb_jobs = randint(1, 10)

    available_skills = create_random_skills(nb_skills)
    staff = create_random_workers(nb_workers, available_skills, horizon)
    jobs = create_random_jobs(nb_jobs, available_skills, horizon)

    instance = {
        "horizon": horizon,
        "qualifications": available_skills,
        "staff": staff,
        "jobs": jobs,
    }
    if save:
        json.dump(instance, open(filepath, "w+"), indent=4)
    return instance


def create_random_skills(nb_skills):
    return sample(list(alphabet.upper()), k=nb_skills)


def create_random_workers(nb_workers, available_skills, horizon):
    return [create_random_worker(available_skills, horizon) for _ in range(nb_workers)]


def create_random_worker(available_skills, horizon):
    skills_nb = randint(1, len(available_skills))
    skills = sample(available_skills, skills_nb)
    name = "".join(choices(list(alphabet), k=randint(3, 10))).capitalize()
    nb_vacation_days = choices(
        population=[i for i in range(horizon)],
        weights=[1 / ((i + 1) ** 2) for i in range(horizon)],
        k=1,
    )[0]
    vacations = sample([i + 1 for i in range(horizon)], nb_vacation_days)
    return {
        "name": name,
        "qualifications": skills,
        "vacations": vacations,
    }


def create_random_jobs(nb_jobs, available_skills, horizon):
    return [create_random_job(available_skills, horizon, i + 1) for i in range(nb_jobs)]


def create_random_job(available_skills, horizon, id):
    gain = randint(10, 80)
    # the following regression and noise were measured empirically on the large instance
    total_working_days = int(
        12 / 60 * gain + 2 + choice([-1, 1]) * np.random.exponential(2)
    )
    nb_skills_required = int(2 / 14 * total_working_days + 1) + choice([-1, 1]) * int(
        np.random.exponential(0.5)
    )

    due_date = randint(min(total_working_days, horizon), horizon)

    skills_required = sample(
        available_skills, min(nb_skills_required, len(available_skills))
    )
    working_days_per_qualification = {}
    for skill in skills_required:
        days = randint(1, total_working_days)
        if days != 0:
            working_days_per_qualification[skill] = days
        total_working_days -= days
        if total_working_days == 0:
            break

    return {
        "name": f"Job{id}",
        "gain": gain,
        "due_date": due_date,
        "daily_penalty": 3,
        "working_days_per_qualification": working_days_per_qualification,
    }
