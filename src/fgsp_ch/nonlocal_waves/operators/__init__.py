from fgsp_ch.nonlocal_waves.operators.bo_fourier import BOFourierOperator
from fgsp_ch.nonlocal_waves.operators.kdv_fourier import KdVFourierOperator
from fgsp_ch.nonlocal_waves.operators.ilw_fourier import ILWFourierOperator
from fgsp_ch.nonlocal_waves.operators.power_law_fourier import PowerLawFourierOperator
from fgsp_ch.nonlocal_waves.operators.wave_traits import DispersiveWaveTraits

__all__ = [
    "BOFourierOperator",
    "KdVFourierOperator",
    "ILWFourierOperator",
    "PowerLawFourierOperator",
    "DispersiveWaveTraits",
]
