"""Independent optimization prototypes for the EE 5239 collision project."""

from .active import ActiveConfig, ActiveReflection, MomentumRestart, MomentumSGD
from .core import State, elastic_pair_collision, specular_reflection
from .hybrid import ContactResolutionError, HybridConfig, HybridContact, first_contact_time
from .gravity import GravityConfig, GravityContact, ballistic_contact_time
from .objectives import (
    Affine, Budget, CountedObjective, EvaluationBudgetExceeded,
    Quadratic, Rosenbrock, TiltedDoubleWell,
)
from .runner import RunResult, run
from .population import (
    PopulationConfig, PopulationGravity, PopulationOracle, PopulationRun,
    PopulationState, pair_contact_time, run_population,
)
from .strategic import StrategicConfig, StagnationMonitor, StrategicRun, run_strategic

__all__ = [
    "ActiveConfig", "ActiveReflection", "MomentumRestart", "MomentumSGD",
    "State", "elastic_pair_collision", "specular_reflection",
    "ContactResolutionError", "HybridConfig", "HybridContact", "first_contact_time",
    "GravityConfig", "GravityContact", "ballistic_contact_time",
    "Affine", "Budget", "CountedObjective", "EvaluationBudgetExceeded",
    "Quadratic", "Rosenbrock", "TiltedDoubleWell", "RunResult", "run",
    "PopulationConfig", "PopulationGravity", "PopulationOracle", "PopulationRun",
    "PopulationState", "pair_contact_time", "run_population",
    "StrategicConfig", "StagnationMonitor", "StrategicRun", "run_strategic",
]
