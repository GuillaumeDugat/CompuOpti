import os
import json

import numpy as np
import gurobipy as grb
from gurobipy import GRB


def build_model(data, with_epsilon_constraint=False):
    model = grb.Model()

    worker_length = len(data["staff"])  # Number of workers
    job_length = len(data["jobs"])  # Number of jobs
    skill_length = len(data["qualifications"])  # Number of skills
    day_length = data["horizon"]  # Number of days

    # Define jobs parameters
    gains_job = np.array([job["gain"] for job in data["jobs"]])
    penalties_job = np.array([job["daily_penalty"] for job in data["jobs"]])
    due_dates_job = np.array([job["due_date"] for job in data["jobs"]])
    work_days_job_skill = np.array(
        [
            [
                job["working_days_per_qualification"][skill]
                if skill in job["working_days_per_qualification"]
                else 0
                for skill in data["qualifications"]
            ]
            for job in data["jobs"]
        ]
    )

    # Define staff parameters
    qualifications_worker_skill = np.array(
        [
            [
                1 if skill in worker["qualifications"] else 0
                for skill in data["qualifications"]
            ]
            for worker in data["staff"]
        ]
    )
    vacations_worker_day = np.array(
        [
            [1 if 1 + day in worker["vacations"] else 0 for day in range(day_length)]
            for worker in data["staff"]
        ]
    )

    ## DECISION VARIABLES ##

    # 4-D array of binary variables : 1 if a worker is assigned to a certain project for a certain skill on a certain day, else 0
    works_worker_job_skill_day = model.addVars(
        worker_length,
        job_length,
        skill_length,
        day_length,
        vtype=GRB.BINARY,
        name="work",
    )

    is_realized_job = model.addVars(
        job_length, vtype=GRB.BINARY, name="is_realized"
    )  # 1 if a job is realized, else 0

    started_after_job_day = model.addVars(
        job_length, day_length, vtype=GRB.BINARY, name="started_after"
    )  # 1 if a job is started after a certain day, else 0
    finished_before_job_day = model.addVars(
        job_length, day_length, vtype=GRB.BINARY, name="finished_before"
    )  # 1 if a job is finished before a certain day, else 0
    max_duration = model.addVar(
        vtype=GRB.INTEGER, name="max_duration"
    )  # Integer that represents the maximum duration for any job

    is_assigned_worker_job = model.addVars(
        worker_length, job_length, vtype=GRB.BINARY, name="is_assigned"
    )  # 1 if a certain worker is assigned on a certain job, else 0
    max_assigned = model.addVar(
        vtype=GRB.INTEGER, name="max_assigned"
    )  # Integer that represents the maximum number of assigned jobs for any worker

    model = add_constraints(
        model,
        worker_length,
        job_length,
        skill_length,
        day_length,
        work_days_job_skill,
        qualifications_worker_skill,
        vacations_worker_day,
        works_worker_job_skill_day,
        is_realized_job,
        started_after_job_day,
        finished_before_job_day,
        max_duration,
        is_assigned_worker_job,
        max_assigned,
    )

    model = add_objective(
        model,
        job_length,
        day_length,
        gains_job,
        penalties_job,
        due_dates_job,
        is_realized_job,
        finished_before_job_day,
        max_duration,
        max_assigned,
        with_epsilon_constraint,
    )

    return model


def add_constraints(
    model,
    worker_length,
    job_length,
    skill_length,
    day_length,
    work_days_job_skill,
    qualifications_worker_skill,
    vacations_worker_day,
    works_worker_job_skill_day,
    is_realized_job,
    started_after_job_day,
    finished_before_job_day,
    max_duration,
    is_assigned_worker_job,
    max_assigned,
):

    model.addConstrs(
        (
            works_worker_job_skill_day[worker, job, skill, day]
            <= qualifications_worker_skill[worker, skill]
            for worker in range(worker_length)
            for job in range(job_length)
            for skill in range(skill_length)
            for day in range(day_length)
        ),
        name="qualification",
    )

    model.addConstrs(
        (
            grb.quicksum(
                works_worker_job_skill_day[worker, job, skill, day]
                for job in range(job_length)
                for skill in range(skill_length)
            )
            <= 1 - vacations_worker_day[worker, day]
            for worker in range(worker_length)
            for day in range(day_length)
        ),
        name="vacation",
    )

    model.addConstrs(
        (
            grb.quicksum(
                works_worker_job_skill_day[worker, job, skill, day]
                for worker in range(worker_length)
                for day in range(day_length)
            )
            == is_realized_job[job] * work_days_job_skill[job, skill]
            for job in range(job_length)
            for skill in range(skill_length)
        ),
        name="job_coverage",
    )

    # started_after == 0 => works == 0
    model.addConstrs(
        (
            works_worker_job_skill_day[worker, job, skill, day]
            <= started_after_job_day[job, day]
            for worker in range(worker_length)
            for job in range(job_length)
            for skill in range(skill_length)
            for day in range(day_length)
        ),
        name="started_after",
    )
    # increasing sequence
    model.addConstrs(
        (
            started_after_job_day[job, day] <= started_after_job_day[job, day + 1]
            for job in range(job_length)
            for day in range(day_length - 1)
        ),
        name="started_after_increasing",
    )
    # is_realized_job == 0 => started_after == 1
    model.addConstrs(
        (
            1 - started_after_job_day[job, day] <= is_realized_job[job]
            for job in range(job_length)
            for day in range(day_length)
        ),
        name="started_after_not_realized",
    )

    # finished before == 1 => works == 0
    model.addConstrs(
        (
            works_worker_job_skill_day[worker, job, skill, day]
            <= 1 - finished_before_job_day[job, day]
            for worker in range(worker_length)
            for job in range(job_length)
            for skill in range(skill_length)
            for day in range(day_length)
        ),
        name="finished_before",
    )
    # increasing sequence
    model.addConstrs(
        (
            finished_before_job_day[job, day] <= finished_before_job_day[job, day + 1]
            for job in range(job_length)
            for day in range(day_length - 1)
        ),
        name="finished_before_increasing",
    )
    # is_realized_job == 0 => finished_before == 1
    model.addConstrs(
        (
            1 - finished_before_job_day[job, day] <= is_realized_job[job]
            for job in range(job_length)
            for day in range(day_length)
        ),
        name="finished_before_not_realized",
    )

    model.addConstrs(
        (
            grb.quicksum(
                started_after_job_day[job, day] - finished_before_job_day[job, day]
                for day in range(day_length)
            )
            <= max_duration
            for job in range(job_length)
        ),
        name="max_duration",
    )

    # exists_skill_day works == 1 => is_assigned == 1
    model.addConstrs(
        (
            works_worker_job_skill_day[worker, job, skill, day]
            <= is_assigned_worker_job[worker, job]
            for worker in range(worker_length)
            for job in range(job_length)
            for skill in range(skill_length)
            for day in range(day_length)
        ),
        name="is_assigned_worker_job",
    )
    # forall_skill_day works == 0 => is_assigned == 0
    model.addConstrs(
        (
            is_assigned_worker_job[worker, job]
            <= grb.quicksum(
                works_worker_job_skill_day[worker, job, skill, day]
                for skill in range(skill_length)
                for day in range(day_length)
            )
            for worker in range(worker_length)
            for job in range(job_length)
        ),
        name="is_assigned_worker_job_bis",
    )

    model.addConstrs(
        (
            grb.quicksum(
                is_assigned_worker_job[worker, job] for job in range(job_length)
            )
            <= max_assigned
            for worker in range(worker_length)
        ),
        name="max_assigned",
    )

    return model


def add_objective(
    model,
    job_length,
    day_length,
    gains_job,
    penalties_job,
    due_dates_job,
    is_realized_job,
    finished_before_job_day,
    max_duration,
    max_assigned,
    with_epsilon_constraint,
):
    if not with_epsilon_constraint : 
        # Add primary objective
        model.ModelSense = GRB.MAXIMIZE
        model.setObjectiveN(
            grb.quicksum(
                gains_job[job] * is_realized_job[job]
                - penalties_job[job]
                * grb.quicksum(
                    1 - finished_before_job_day[job, day]
                    for day in range(due_dates_job[job], day_length)
                )
                for job in range(job_length)
            ),
            0,
            priority=2,
        )
        # Add multi-objective functions
        model.setObjectiveN(
            -max_assigned,
            1,
            priority=1,
        )
        model.setObjectiveN(
            -max_duration,
            2,
            priority=0,
        )
    else : 
        # Add primary objective
        model.setObjective(
        grb.quicksum(
                gains_job[job] * is_realized_job[job]
                - penalties_job[job]
                * grb.quicksum(
                    1 - finished_before_job_day[job, day]
                    for day in range(due_dates_job[job], day_length)
                )
                for job in range(job_length)
            ) + 0.005 * max_duration + 0.001 * max_assigned,
        sense=GRB.MAXIMIZE,
    )


    return model


def get_instance(instance_filename):
    file = os.path.join("instances", instance_filename)
    data = json.load(open(file, "r"))
    return data


if __name__ == "__main__":
    instance_filename = "toy_instance.json"
    data = get_instance(instance_filename)
    build_model(data)
