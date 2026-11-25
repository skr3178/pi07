"""Section V prompt fields and dropout, independent of policy/model loading.

Unknown annotations are omitted, never converted to good quality/no mistake.
The local controller supports only end-effector actions (ee).
"""
from dataclasses import asdict, dataclass, field
import hashlib
import math
import random


@dataclass(frozen=True)
class Evidence:
    value: object = None
    source: str = "unavailable"
    confidence: float | None = None

    def __post_init__(self):
        if self.value is not None and (not self.source.strip() or self.source == "unavailable"):
            raise ValueError("Available annotations require provenance")
        if self.confidence is not None and not (math.isfinite(self.confidence) and 0 <= self.confidence <= 1):
            raise ValueError("Confidence must be unknown or between zero and one")


@dataclass(frozen=True)
class PromptContext:
    task: str
    subtask: Evidence = field(default_factory=Evidence)
    speed_steps: Evidence = field(default_factory=Evidence)
    quality: Evidence = field(default_factory=Evidence)
    mistake: Evidence = field(default_factory=Evidence)
    control_mode: str = "ee"

    def __post_init__(self):
        if not isinstance(self.task, str) or not self.task.strip():
            raise ValueError("A nonempty full task instruction is required")
        if self.control_mode != "ee":
            raise ValueError("This LIBERO implementation supports ee only; a joint tag cannot change the controller")
        if self.subtask.value is not None and (not isinstance(self.subtask.value,str) or not self.subtask.value.strip()):
            raise ValueError("Subtask must be nonempty text or unavailable")
        if self.speed_steps.value is not None and (type(self.speed_steps.value) is not int or self.speed_steps.value <= 0):
            raise ValueError("Duration must be a positive integer count of recorded steps")
        if self.quality.value is not None and (type(self.quality.value) is not int or not 1 <= self.quality.value <= 5):
            raise ValueError("Quality must be an annotated integer from 1 to 5")
        if self.mistake.value is not None and type(self.mistake.value) is not bool:
            raise ValueError("Mistake must be true, false or unavailable")


@dataclass(frozen=True)
class PromptOptions:
    subtasks: bool = True
    goals: bool = True
    metadata: bool = True
    goal_probability: float = .25
    subtask_drop_given_goal: float = .30
    metadata_drop: float = .15
    metadata_field_drop: float = .05
    # LIBERO durations are much shorter than the paper's robot episodes.
    # Use 500 for the paper's bin width; 50 is the declared local adaptation.
    speed_bin_steps: int = 50

    def __post_init__(self):
        for name in ("goal_probability", "subtask_drop_given_goal", "metadata_drop", "metadata_field_drop"):
            if not 0 <= getattr(self,name) <= 1:
                raise ValueError(f"Invalid probability: {name}")
        if type(self.speed_bin_steps) is not int or self.speed_bin_steps <= 0:
            raise ValueError("Speed bin width must be a positive integer")

    @classmethod
    def profile(cls, name, **overrides):
        settings = {"task": (False,False,False), "subtask": (True,False,False),
                    "goal": (False,True,False), "subtask_goal": (True,True,False),
                    "full": (True,True,True)}
        if name not in settings:
            raise ValueError(f"Unknown prompt profile: {name}")
        a,b,c = settings[name]
        return cls(subtasks=a,goals=b,metadata=c,**overrides)


@dataclass(frozen=True)
class BuiltPrompt:
    text: str
    goal_present: bool
    fields: dict
    decisions: dict


def speed_bin(steps, width):
    """Round half up, with at least one bin for short positive durations."""
    return max(width, ((2*steps+width)//(2*width))*width)


def build_prompt(context, options, *, training, seed=0, sample_key=0, goal_available=True):
    # Local RNG and a fixed set of draws: independent of decoding, DataLoader
    # order, global RNG, field missingness and profile switches.
    key = f"section-v-v1:{seed}:{sample_key}".encode()
    rng = random.Random(int.from_bytes(hashlib.sha256(key).digest()[:16],"big"))
    goal_draw, subtask_draw, meta_draw, *field_draws = [rng.random() for _ in range(6)]
    goal = options.goals and goal_available and (not training or goal_draw < options.goal_probability)
    drop_subtask = training and goal and subtask_draw < options.subtask_drop_given_goal
    drop_metadata = training and meta_draw < options.metadata_drop
    fields = {}
    text = [f"Task: {context.task.strip().rstrip('.')}."]
    for i,name in enumerate(("subtask","speed_steps","quality","mistake")):
        evidence = getattr(context,name)
        enabled = options.subtasks if name == "subtask" else options.metadata
        dropped = drop_subtask if name == "subtask" else drop_metadata or (
            training and field_draws[i-1] < options.metadata_field_drop)
        status = ("unavailable" if evidence.value is None else "disabled" if not enabled
                  else "dropped" if dropped else "present")
        fields[name] = {**asdict(evidence), "status": status}
        if status != "present":
            continue
        if name == "subtask":
            text.append(f"Subtask: {evidence.value.strip().rstrip('.')}.")
        elif name == "speed_steps":
            value = speed_bin(evidence.value, options.speed_bin_steps)
            fields[name]["bin"] = value
            text.append(f"Speed: {value} recorded steps.")
        elif name == "quality":
            text.append(f"Quality: {evidence.value}.")
        else:
            text.append(f"Mistake: {str(evidence.value).lower()}.")
    text.append(f"Control Mode: {context.control_mode}.")
    return BuiltPrompt(" ".join(text), bool(goal), fields,
        {"training": training,"subtask_drop":bool(drop_subtask),
         "all_metadata_drop":bool(drop_metadata),"control_mode":"ee"})


def runtime_prompt(task, *, options=None, subtask=None, subtask_source=None,
                   desired_speed_steps=None, desired_quality=None, desired_mistake=None,
                   goal_available=False):
    """Construct runtime requests without episode/frame labels or future data.

    A subtask must be supplied by a human or controller, not replayed demo time.
    Metadata here expresses requested behavior, not observed execution quality.
    """
    if subtask is not None and not subtask_source:
        raise ValueError("Runtime subtask requires a human/controller source")
    context = PromptContext(task,
        subtask=Evidence(subtask,subtask_source or "unavailable"),
        speed_steps=Evidence(desired_speed_steps,"inference_request"),
        quality=Evidence(desired_quality,"inference_request"),
        mistake=Evidence(desired_mistake,"inference_request"))
    return build_prompt(context, options or PromptOptions(), training=False, goal_available=goal_available)


class CheckedPromptProcessor:
    """Fail on overlength text instead of silently dropping later prompt fields."""
    def __init__(self, pipeline, max_length):
        self.pipeline, self.max_length = pipeline, max_length
        self.tokenizer = next(s.input_tokenizer for s in pipeline.steps if hasattr(s,"input_tokenizer"))

    def token_counts(self, texts):
        if isinstance(texts,str): texts=[texts]
        texts=[t if t.endswith("\n") else t+"\n" for t in texts]
        return [len(ids) for ids in self.tokenizer(texts,truncation=False,padding=False)["input_ids"]]

    def __call__(self, batch):
        counts = self.token_counts(batch["task"])
        if max(counts) > self.max_length:
            raise ValueError(f"Prompt needs {max(counts)} tokens; configured maximum is {self.max_length}")
        return self.pipeline(batch)

    def save_pretrained(self, *args, **kwargs):
        return self.pipeline.save_pretrained(*args,**kwargs)


def make_section_v_processors(config, checkpoint, *, max_length=128):
    from lerobot.policies.factory import make_pre_post_processors
    config.tokenizer_max_length = max_length
    pre,post = make_pre_post_processors(config,pretrained_path=str(checkpoint),
        preprocessor_overrides={"tokenizer_processor":{"max_length":max_length,"truncation":False},
                               "device_processor":{"device":config.device}},
        postprocessor_overrides={"device_processor":{"device":"cpu"}})
    return CheckedPromptProcessor(pre,max_length),post
