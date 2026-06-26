"""Export a recorded OpenOutcry rollout trace to a `Minari <https://minari.farama.org>`_
offline-RL dataset.

Minari (Farama) is the offline-RL trajectory dataset standard. This module converts the
versioned JSONL rollout trace produced by :mod:`openoutcry.trace`
(:class:`~openoutcry.trace.RolloutTraceWriter`) into a :class:`minari.MinariDataset`
**without spawning a live env**: it assembles one ``EpisodeBuffer`` per episode by direct
field assignment and calls :func:`minari.create_dataset_from_buffers`.

The field map is::

    trace ``decision``     -> Minari ``actions``
    trace ``observation``  -> Minari ``observations``   (n+1 convention, see below)
    trace ``reward``       -> Minari ``rewards``
    (synthesized)          -> Minari ``terminations`` / ``truncations``
    trace ``info`` (subset)-> Minari ``infos``          (scalar fields only)

**The n+1 vs n convention.** Minari stores ``T+1`` observations against ``T`` actions: a
reset observation plus one observation after each of the ``T`` steps. The OpenOutcry trace
records the ``T`` *point-in-time* observations the agent actually saw (one per decision); it
does not capture the post-final observation. We therefore emit the ``T`` recorded
observations as ``observations[0..T-1]`` and synthesize the terminal ``observations[T]`` by
repeating the last recorded observation (override with ``terminal_observation=``). The
invariant ``len(observations) == len(actions) + 1`` always holds.

**Termination is run-level.** The trace records a single rollout outcome, not a per-step
terminated/truncated flag. We synthesize: every non-final step is ``terminated=False,
truncated=False``; the final step is ``terminated=True`` iff the run truly terminated (the
gym env's bankruptcy/absorbing state, inferred from ``info['nav'] <= 0`` or an explicit
``terminated=`` override), else ``truncated=True`` (it merely ran out of bars). Every episode
thus ends on exactly one of the two flags, as Minari requires.

**Leak-safety** carries over from :mod:`openoutcry.trace`: a raw dataset/env handle in any
record is rejected (it would carry the full, future-inclusive series), and only a whitelist
of scalar ``info`` fields is serialized — never a full series.

``minari`` is an OPTIONAL dependency (install the ``minari`` extra). This module imports
cleanly without it; :func:`to_minari` raises a clear error if called when it is missing.
``minari.create_dataset_from_buffers`` is the only entry used, so the heavy ``jax`` dependency
of ``EpisodeBuffer.add_step_data`` is avoided by assembling buffers by direct field
assignment.

Verified against ``minari`` 0.5.3: ``EpisodeBuffer(id, seed, observations, actions, rewards,
terminations, truncations, infos)``; ``create_dataset_from_buffers(dataset_id, buffer,
env=None, observation_space=, action_space=, ...)``; per-step ``StepData = {observation,
action, reward, terminated, truncated, info}``.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import numpy as np

from .trace import _is_leaky, load_trace

try:  # pragma: no cover - exercised only when minari is installed
    import minari  # noqa: F401
    from minari import create_dataset_from_buffers
    from minari.data_collector.episode_buffer import EpisodeBuffer

    _HAS_MINARI = True
except Exception:  # noqa: BLE001 - any import failure means "not available"
    minari = None  # type: ignore[assignment]
    create_dataset_from_buffers = None  # type: ignore[assignment]
    EpisodeBuffer = None  # type: ignore[assignment,misc]
    _HAS_MINARI = False


# Scalar `info` fields safe to serialize into Minari `infos`. Anything not on this list and
# not a bare scalar is dropped, so a full (future-inclusive) series can never leak through.
_DEFAULT_INFO_FIELDS = ("scenario_seed", "nav", "reward", "invalid", "step")

_SCALAR_TYPES = (bool, int, float, str)


def _require_minari() -> None:
    if not _HAS_MINARI:
        raise RuntimeError(
            "minari is not installed. Install the 'minari' extra "
            "(pip install 'openoutcry[minari]') to export rollout traces to an offline-RL "
            "dataset; the rest of the openoutcry package works without it."
        )


# ---------------------------------------------------------------------------
# Trace ingestion
# ---------------------------------------------------------------------------

def _resolve_trace(trace: Any) -> tuple[list[dict], dict]:
    """Normalize the ``trace`` argument into ``(step_records, meta)``.

    Accepts a JSONL path (``str``/``os.PathLike``), an in-memory ``(records, meta)`` tuple,
    or a bare list of step records (``meta`` defaults to ``{}``).
    """
    if isinstance(trace, (str, bytes)) or hasattr(trace, "__fspath__"):
        return load_trace(str(trace))
    if isinstance(trace, tuple) and len(trace) == 2:
        records, meta = trace
        return list(records), dict(meta or {})
    if isinstance(trace, Sequence):
        return list(trace), {}
    raise TypeError(
        "trace must be a JSONL path, an in-memory (records, meta) tuple, or a list of "
        f"step records; got {type(trace).__name__!r}"
    )


def _step_records(records: Sequence[dict]) -> list[dict]:
    """Drop the run ``meta`` record and any non-step rows, preserving recorded order."""
    return [r for r in records if isinstance(r, dict) and r.get("kind") != "meta"]


def _split_episodes(records: Sequence[dict]) -> list[list[dict]]:
    """Partition step records into episodes.

    A new episode starts when the per-step ``step`` counter resets to 0 (a fresh rollout) or
    when the ``info['scenario_seed']`` changes. A trace from a single rollout yields one
    episode; a concatenated multi-rollout trace yields several.
    """
    episodes: list[list[dict]] = []
    current: list[dict] = []
    cur_seed: Any = object()  # sentinel that compares unequal to any real seed
    for rec in records:
        step = rec.get("step")
        seed = (rec.get("info") or {}).get("scenario_seed")
        boundary = bool(current) and ((step == 0) or (seed != cur_seed))
        if boundary:
            episodes.append(current)
            current = []
        current.append(rec)
        cur_seed = seed
    if current:
        episodes.append(current)
    return episodes


# ---------------------------------------------------------------------------
# Field mapping
# ---------------------------------------------------------------------------

def _reject_leaky_record(rec: dict) -> None:
    """Refuse to export a record carrying a raw dataset/env handle (full-series leak)."""
    for field in ("observation", "decision"):
        value = rec.get(field)
        if _is_leaky(value):
            raise TypeError(
                f"refusing to export {field}={type(value).__name__!r}: a raw dataset/env "
                "handle would leak future bars into the dataset."
            )


def _coerce_action(decision: Any, action_space: Any, action_fn: Any) -> np.ndarray:
    """Map a recorded ``decision`` onto an action array matching ``action_space``."""
    raw = action_fn(decision) if action_fn is not None else decision
    try:
        arr = np.asarray(raw, dtype=action_space.dtype)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"cannot map decision {decision!r} to an action of dtype "
            f"{action_space.dtype}; pass action_fn= to convert raw decisions "
            "(e.g. a label) into a target-weight vector."
        ) from exc
    if action_space.shape is not None and arr.shape != tuple(action_space.shape):
        raise ValueError(
            f"mapped action shape {arr.shape} != action_space.shape "
            f"{tuple(action_space.shape)}; pass action_fn= to reshape."
        )
    return arr


def _assemble_observations(samples: Sequence[Any], space: Any) -> Any:
    """Stack per-step observation ``samples`` into Minari's tree-of-lists structure.

    For a ``Dict`` observation space the result is a dict whose leaves are per-key lists; for
    a ``Box`` (or any leaf space) it is a flat list of coerced arrays. Mirrors the structure
    ``EpisodeBuffer.add_step_data`` builds, but without importing jax.
    """
    spaces_dict = getattr(space, "spaces", None)
    if isinstance(spaces_dict, dict):
        return {
            key: _assemble_observations([s[key] for s in samples], sub)
            for key, sub in spaces_dict.items()
        }
    dtype = getattr(space, "dtype", None)
    if dtype is not None:
        return [np.asarray(s, dtype=dtype) for s in samples]
    return list(samples)


def _whitelisted_infos(
    records: Sequence[dict], info_fields: Sequence[str]
) -> Optional[dict]:
    """Per-step ``infos`` as a dict of per-field lists, keeping only scalar whitelisted keys.

    A field is emitted only if it is a bare scalar (bool/int/float/str) in every step where it
    appears; missing values read as ``None``. Lists/dicts (a potential full-series leak) are
    dropped. Returns ``None`` when nothing survives so the buffer's ``infos`` stays empty.
    """
    allow = set(info_fields)
    out: dict[str, list] = {}
    for field in info_fields:
        column: list = []
        seen = False
        for rec in records:
            value = (rec.get("info") or {}).get(field)
            if isinstance(value, _SCALAR_TYPES):
                seen = True
                column.append(value)
            else:
                column.append(None)
        if seen and field in allow:
            out[field] = column
    return out or None


def _flatten_scores(scores: Any, prefix: str = "sharpebench") -> dict:
    """Flatten the (possibly nested) SharpeBench score dict to ``prefix_path -> scalar``.

    HDF5 / JSON attribute stores keep flat scalars; nested dicts are joined with ``_`` and
    non-scalar leaves are coerced to ``float`` when possible, else dropped.
    """
    flat: dict[str, Any] = {}

    def _walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                _walk(v, f"{path}_{k}" if path else str(k))
            return
        if isinstance(node, bool):
            flat[path] = node
        elif isinstance(node, (int, float)):
            flat[path] = node
        elif isinstance(node, str):
            flat[path] = node
        else:
            try:
                flat[path] = float(node)
            except (TypeError, ValueError):
                pass

    _walk(scores or {}, prefix)
    return flat


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def to_minari(
    trace: Union[str, "os.PathLike[str]", tuple, Sequence[dict]],  # noqa: F821
    dataset_id: str,
    *,
    observation_space: Any,
    action_space: Any,
    terminated: Optional[bool] = None,
    terminal_observation: Any = None,
    action_fn: Any = None,
    info_fields: Sequence[str] = _DEFAULT_INFO_FIELDS,
    env: Any = None,
    **kwargs: Any,
):
    """Convert a recorded OpenOutcry rollout ``trace`` into a :class:`minari.MinariDataset`.

    Parameters
    ----------
    trace:
        A JSONL trace path, an in-memory ``(records, meta)`` tuple (e.g. from
        :func:`openoutcry.trace.load_trace`), or a bare list of step records.
    dataset_id:
        Minari dataset id of the form ``namespace/name-vN``.
    observation_space, action_space:
        Gymnasium spaces describing the trajectories — required so Minari can build the
        dataset WITHOUT a live env. Use ``OpenOutcryEnv(...).observation_space`` /
        ``.action_space``.
    terminated:
        Override the run-level outcome. ``None`` (default) infers it per episode from the
        final step's ``info['nav']`` (``<= 0`` ⇒ terminated, else truncated).
    terminal_observation:
        The synthesized ``observations[T]`` (post-final). ``None`` repeats the last recorded
        observation.
    action_fn:
        Optional ``decision -> action`` callable for traces whose ``decision`` is not already
        an action-shaped vector (e.g. a textual label).
    info_fields:
        Whitelist of scalar ``info`` keys to carry into per-step ``infos``.
    env:
        Optional environment id / spec recorded on the dataset so
        :meth:`minari.MinariDataset.recover_environment` can re-instantiate it. Leave as
        ``None`` to build the dataset purely from buffers (no live env); pass a registered id
        such as ``"OpenOutcry/<Tier>-v1"`` once the gymnasium ids are registered to make
        ``recover_environment()`` work out of the box.
    **kwargs:
        Forwarded to :func:`minari.create_dataset_from_buffers` (e.g. ``author``,
        ``description``, ``code_permalink``, ``data_format``).

    Returns
    -------
    minari.MinariDataset
        The created dataset. The flattened SharpeBench scores from the run ``meta`` are
        written to dataset-level metadata (and per-episode metadata) as flat scalars.
    """
    _require_minari()

    records, meta = _resolve_trace(trace)
    steps = _step_records(records)
    if not steps:
        raise ValueError("trace contains no step records to export")

    score_attrs = _flatten_scores(meta.get("scores"))

    buffers = []
    for ep_id, ep in enumerate(_split_episodes(steps)):
        for rec in ep:
            _reject_leaky_record(rec)

        actions = [_coerce_action(r.get("decision"), action_space, action_fn) for r in ep]
        rewards = [float(r.get("reward", 0.0)) for r in ep]

        obs_samples = [r.get("observation") for r in ep]
        last_info = ep[-1].get("info") or {}
        term = (
            terminated
            if terminated is not None
            else float(last_info.get("nav", 1.0)) <= 0.0
        )
        terminations = [False] * len(ep)
        truncations = [False] * len(ep)
        terminations[-1] = bool(term)
        truncations[-1] = not bool(term)

        terminal = terminal_observation if terminal_observation is not None else obs_samples[-1]
        observations = _assemble_observations([*obs_samples, terminal], observation_space)

        seed = None
        raw_seed = last_info.get("scenario_seed")
        if isinstance(raw_seed, (int, float)) and not isinstance(raw_seed, bool):
            seed = int(raw_seed)

        buffers.append(
            EpisodeBuffer(
                id=ep_id,
                seed=seed,
                observations=observations,
                actions=actions,
                rewards=rewards,
                terminations=terminations,
                truncations=truncations,
                infos=_whitelisted_infos(ep, info_fields),
            )
        )

    dataset = create_dataset_from_buffers(
        dataset_id,
        buffer=buffers,
        env=env,
        observation_space=observation_space,
        action_space=action_space,
        **kwargs,
    )

    # Carry SharpeBench scores into dataset- and episode-level metadata as flat scalars.
    if score_attrs:
        try:
            dataset.storage.update_metadata({"sharpebench_scores": score_attrs})
            for ep_id in range(len(buffers)):
                dataset.storage.update_episode_metadata(
                    [score_attrs], episode_indices=[ep_id]
                )
        except Exception:  # noqa: BLE001 - metadata is best-effort; the dataset is the artifact
            pass

    return dataset


__all__ = ["to_minari"]
