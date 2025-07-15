import cProfile
import pstats


from ponds.blobgen import GAUSS as GAUSS_new
from ponds.core import PONDS as PONDS_new
from ponds.shiftgen import SIGMOID as SIGMOID_new

def main():

    pd = PONDS_new(
        lat_size=200,
        lon_size=400,
        time_size=500,
        background_noise=0.01,
        background_trend=0.002,
    )

    #pd.add_shift(random_seed=1,
    #             shape_method=GAUSS,
    #             shift_method=SIGMOID_new)
    

    profiler = cProfile.Profile()
    profiler.enable()
    #shift_dict = {
        #"shift_steepness": 0.2,
        #"shift_magnitude": 0.5,
    #}
    pd.add_shift(
        random_seed=4,
        shape_method=GAUSS_new(),
        shift_method=SIGMOID_new(),
    #    kwargs=shift_dict,
    )
    
    profiler.disable()
    profiler.dump_stats("profile_output_new.prof")

if __name__ == "__main__":
    #profiler = cProfile.Profile()
    #profiler.enable()

    main()

    #profiler.disable()
    #profiler.dump_stats("profile_output.prof")
