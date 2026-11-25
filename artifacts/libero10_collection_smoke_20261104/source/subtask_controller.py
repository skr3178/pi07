"""Explicit simulator-state oracle for LIBERO-10 subtask diagnostics.

This is privileged-state assistance, NOT a learned planner. Selects the first
unmet semantic condition from current simulator predicates; no demo timings.
Container insertion precedes closing even when BDDL lists closure first.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Subtask:
    predicate_index: int
    predicate: tuple
    text: str
    priority: int


def build_plan(predicates):
    objects={p[1] for p in predicates}
    plan=[]
    for i,p in enumerate(predicates):
        kind=p[0].lower();obj=p[1];target=p[2] if len(p)>2 else None
        priority=1
        if kind=="turnon" and obj=="flat_stove_1":
            text="turn on the stove";priority=0
        elif kind=="close" and obj=="white_cabinet_1_bottom_region":
            text="close the drawer";priority=2
        elif kind=="close" and obj=="microwave_1":
            text="close the microwave";priority=2
        elif kind=="in" and target=="white_cabinet_1_bottom_region" and obj=="akita_black_bowl_1":
            text="put the black bowl in the bottom drawer"
        elif kind=="in" and target=="microwave_1_heating_region" and obj=="white_yellow_mug_1":
            text="put the yellow and white mug in the microwave"
        elif kind=="in" and target=="basket_1_contain_region":
            names={"alphabet_soup_1":"alphabet soup","cream_cheese_1":"cream cheese box",
                   "tomato_sauce_1":"tomato sauce","butter_1":"butter"}
            if obj not in names:raise ValueError(f"Unsupported basket object: {obj}")
            text=f"put the {names[obj]} in the basket"
        elif kind=="on" and target=="flat_stove_1_cook_region" and obj.startswith("moka_pot_"):
            # Unique instance wording is explicit; its training alignment is not
            # assumed. It must be checked in the image-language probe.
            name=("right moka pot" if obj=="moka_pot_1" else "left moka pot") if "moka_pot_2" in objects else "moka pot"
            text=f"put the {name} on the stove"
        elif kind=="on" and obj=="porcelain_mug_1" and target=="plate_1":
            text="put the white mug on the left plate" if "white_yellow_mug_1" in objects else "put the white mug on the plate"
        elif kind=="on" and obj=="white_yellow_mug_1" and target=="plate_2":
            text="put the yellow and white mug on the right plate"
        elif kind=="on" and obj=="chocolate_pudding_1" and target=="living_room_table_plate_right_region":
            text="put the chocolate pudding to the right of the plate"
        elif kind=="in" and obj=="black_book_1" and target=="desk_caddy_1_back_contain_region":
            text="pick up the book and place it in the back compartment of the caddy"
        else:raise ValueError(f"Unsupported oracle predicate: {p}")
        plan.append(Subtask(i,tuple(p),text,priority))
    return sorted(plan,key=lambda item:(item.priority,item.predicate_index))


def select_subtask(plan,current_truth):
    if len(plan)!=len(current_truth) or any(type(x) is not bool for x in current_truth):
        raise ValueError("One current boolean per predicate is required")
    return next((step for step in plan if not current_truth[step.predicate_index]),None)


def from_environment(env):
    domain=env._env.env
    predicates=domain.parsed_problem["goal_state"]
    truth=[bool(domain._eval_predicate(p)) for p in predicates]
    return select_subtask(build_plan(predicates),truth)
