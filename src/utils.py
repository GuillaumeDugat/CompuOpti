import os
import json

import numpy as np
import pandas as pd
import matplotlib as mpl


def get_instance(instance_filename):
    file = os.path.join("instances", instance_filename)
    data = json.load(open(file, "r"))
    return data


def disply_worker_skills(instance):
    day_length = instance["horizon"]
    qualifications_worker_skill = np.array(
        [
            [
                1 if skill in worker["qualifications"] else 0
                for skill in instance["qualifications"]
            ]
            for worker in instance["staff"]
        ]
    )
    vacations_worker_day = np.array(
        [
            [1 if 1 + day in worker["vacations"] else 0 for day in range(day_length)]
            for worker in instance["staff"]
        ]
    )
    qualifications = instance["qualifications"]
    names = [worker["name"] for worker in instance["staff"]]

    res = pd.DataFrame(
        [
            [
                str([q for (r, q) in zip(row_q, qualifications) if r == 1]).replace(
                    "'", ""
                ),
                str([d for (r, d) in zip(row_d, range(day_length)) if r == 1]),
            ]
            for row_q, row_d in zip(qualifications_worker_skill, vacations_worker_day)
        ],
        index=names,
        columns=["qualifications", "vacations"],
    )

    return res


def highlight_cols(col, instance):
    jobs = [job["name"] for job in instance["jobs"]]
    # Give a hex color to each job
    cmap = mpl.cm.get_cmap("hsv", int(len(jobs) * 1.2))
    job_colors = [mpl.colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)][: len(jobs)]
    return [f"color:black;background-color: {color}" for color in job_colors]


def display_work_days(instance):
    gains_job = np.array([job["gain"] for job in instance["jobs"]])
    penalties_job = np.array([job["daily_penalty"] for job in instance["jobs"]])
    due_dates_job = np.array([job["due_date"] for job in instance["jobs"]])
    work_days_job_skill = np.array(
        [
            [
                job["working_days_per_qualification"][skill]
                if skill in job["working_days_per_qualification"]
                else 0
                for skill in instance["qualifications"]
            ]
            for job in instance["jobs"]
        ]
    )
    qualifications = instance["qualifications"]
    jobs = [job["name"] for job in instance["jobs"]]

    res = pd.DataFrame(work_days_job_skill, index=jobs, columns=qualifications)
    res["due date"] = pd.Series(due_dates_job, index=jobs)
    res["gain"] = pd.Series(gains_job, index=jobs)
    res["penalty"] = pd.Series(penalties_job, index=jobs)
    res = res.style.apply(lambda x: highlight_cols(x, instance), axis=0)
    return res


def find_task(worker, day, instance, model):
    job_length = len(instance["jobs"])
    skill_length = len(instance["qualifications"])
    qualifications = instance["qualifications"]
    for job in range(job_length):
        for skill in range(skill_length):
            tab = model.getVarByName(f"work[{worker},{job},{skill},{day}]").x
            if tab == 1:
                return job, qualifications[skill]
    return None


def color_cells(x, df, instance):
    jobs = [job["name"] for job in instance["jobs"]]
    # Give a hex color to each job
    cmap = mpl.cm.get_cmap("hsv", int(len(jobs) * 1.2))
    job_colors = [mpl.colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)][: len(jobs)]

    df = df.applymap(lambda val: job_colors[val[0]] if val is not None else "")
    df = df.applymap(
        lambda color: f"color:{'' if color == '' else 'black'};background-color: {color}"
    )
    return df


def display_time_table(instance, model):
    worker_length = len(instance["staff"])
    day_length = instance["horizon"]
    names = [worker["name"] for worker in instance["staff"]]

    data = [
        [find_task(worker, day, instance, model) for day in range(day_length)]
        for worker in range(worker_length)
    ]
    df = pd.DataFrame(data, index=names, columns=range(day_length))
    res = df.applymap(lambda x: x[1] if x is not None else None)
    res = res.style.apply(lambda x: color_cells(x, df, instance), axis=None)
    return res
