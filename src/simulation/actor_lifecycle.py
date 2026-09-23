"""
actor_lifecycle.py

Stale-actor classification shared by every per-frame serializer
(src/data/world_state.py, src/data/annotation.py) and the canonical traffic
cleanup (src/simulation/canonical_traffic.py). Pure Python: no CARLA import.

A background actor can leave the server registry between the world snapshot
and a per-actor RPC (e.g. Vehicle.get_light_state() -> server
"get_vehicle_light_state"). Only the two CARLA messages below mean "this
actor no longer exists"; every other RuntimeError is a real failure and must
propagate.
"""


STALE_ACTOR_ERROR_MARKERS = (
    # Server side (ECarlaServerResponse::ActorNotFound), e.g.
    # "Responding error from function get_vehicle_light_state: Actor could
    # not be found in the registry. Actor Id: 523"
    "Actor could not be found in the registry",
    # Client side (LibCarla): the Python wrapper's episode was cleared by
    # a successful actor.destroy() in this process.
    "trying to operate on a destroyed actor",
)


class EgoActorLostError(RuntimeError):
    """The ego vehicle is gone: the canonical sequence cannot continue."""


def is_stale_actor_error(exc):
    return isinstance(exc, RuntimeError) and any(
        marker in str(exc) for marker in STALE_ACTOR_ERROR_MARKERS
    )


def require_ego_alive(ego, context):
    """Ego is never silently skipped (unlike background actors)."""

    if ego is None or not ego.is_alive:
        raise EgoActorLostError(
            f"Ego actor {getattr(ego, 'id', None)} is no longer alive ({context}); "
            f"canonical sequence aborted."
        )
