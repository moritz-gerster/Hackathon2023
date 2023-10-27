from scripts.spectral_analysis import correlations
from scripts.phase_analysis import phase_spiking
from scripts.utils import _get_brain_areas, _load_data


def main(fname="steinmetz_2016-12-14_Cori.nc"):
    data = _load_data(fname)
    brain_areas = _get_brain_areas(data)
    for brain_area in brain_areas:
        correlations(brain_area=brain_area)
        phase_spiking(brain_area=brain_area)


if __name__ == '__main__':
    main()