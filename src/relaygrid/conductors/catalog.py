from __future__ import annotations
from dataclasses import dataclass
@dataclass(frozen=True)
class Conductor:
    code: str
    r_ohm_km: float
    x_ohm_km: float
    ampacity_a: float
    diameter_mm: float = 18.0
    max_temp_c: float = 75.0
CATALOG={
    "ACSR_1_0": Conductor("ACSR_1_0",0.321,0.402,230,10.1,75.0),
    "ACSR_4_0": Conductor("ACSR_4_0",0.198,0.384,340,14.3,75.0),
    "AAAC_336": Conductor("AAAC_336",0.161,0.372,420,18.3,75.0),
    "CU_500": Conductor("CU_500",0.128,0.355,510,19.5,75.0),
}
def get(code: str): return CATALOG[code]
def parallel_impedance(code: str,length_km: float,circuits: int=1):
    c=get(code)
    if length_km<0 or circuits<=0: raise ValueError("invalid geometry")
    return c.r_ohm_km*length_km/circuits,c.x_ohm_km*length_km/circuits
def thermal_mva(code: str,kv_ll: float):
    c=get(code); return 1.7320508075688772*kv_ll*c.ampacity_a/1000
