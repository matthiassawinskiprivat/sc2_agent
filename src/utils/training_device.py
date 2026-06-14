"""

Resolve PyTorch device for SB3 training (sync universal + async learner/collector).



Precedence: env SC2_TRAINING_DEVICE > YAML training.device > auto.



**GPU only:** ``auto``, ``cuda``, and ``gpu`` all resolve to ``cuda:0`` and raise if

CUDA is unavailable. Do not add CPU fallbacks or split collector/learner devices.

"""



from __future__ import annotations



import os

from typing import Any, Mapping, Optional



import torch





def _require_cuda() -> torch.device:

    if not torch.cuda.is_available():

        raise RuntimeError(

            "This project requires CUDA for all policy training and inference "

            "(collector, learner, opponent). torch.cuda.is_available() is False. "

            "Install a CUDA-enabled PyTorch build, check drivers, and CUDA_VISIBLE_DEVICES."

        )

    return torch.device("cuda:0")





def resolve_training_device(training_config: Optional[Mapping[str, Any]] = None) -> torch.device:

    cfg = training_config or {}

    raw = os.environ.get("SC2_TRAINING_DEVICE", "").strip().lower()

    if not raw:

        raw = str(cfg.get("device", "auto")).strip().lower()



    if raw == "cpu":

        raise RuntimeError(

            "training.device / SC2_TRAINING_DEVICE=cpu is not supported. "

            "All SB3 models (collector, learner, opponent) must run on CUDA."

        )



    if raw in ("cuda", "gpu", "auto", "", "default"):

        return _require_cuda()



    raise ValueError(

        f"Invalid training device {raw!r}. Use 'auto' or 'cuda' "

        "(SC2_TRAINING_DEVICE or training.device). CPU is not supported."

    )





def resolve_opponent_device(training_config: Optional[Mapping[str, Any]] = None) -> torch.device:

    """

    Device for opponent inference inside vec-env worker subprocesses.



    Always CUDA (same as collector/learner). Override keys are ignored if they request CPU.

    """

    cfg = dict(training_config or {})

    env_cfg = cfg.get("env") or {}

    raw = os.environ.get("SC2_OPPONENT_DEVICE", "").strip().lower()

    if not raw:

        raw = str(

            cfg.get("opponent_device") or env_cfg.get("opponent_device") or "cuda"

        ).strip().lower()



    if raw == "cpu":

        raise RuntimeError(

            "opponent_device / SC2_OPPONENT_DEVICE=cpu is not supported. "

            "Opponent policies must run on CUDA."

        )



    if raw in ("cuda", "gpu", "auto", "", "default"):

        return resolve_training_device(cfg)



    raise ValueError(

        f"Invalid opponent device {raw!r}. Use 'cuda' or 'auto'. CPU is not supported."

    )





def warmup_cuda_if_needed(device: torch.device) -> None:

    """Eager CUDA context creation in the current process (helps lazy init on some Windows setups)."""

    if device.type != "cuda":

        raise RuntimeError(

            f"warmup_cuda_if_needed expected a CUDA device, got {device!r}. "

            "This codebase does not support CPU training."

        )

    torch.cuda.init()

    torch.zeros(1, device=device)


