import cProfile
import pstats


from ponds.blobgen import GAUSS
from ponds.core_new import PONDS as PONDS_new
from ponds.shiftgen_new import SIGMOID as SIGMOID_new

def main():

    pd = PONDS_new(
        lat_size=20,
        lon_size=40,
        time_size=200,
        background_noise=0.01,
        background_trend=0.002,
    )

    #pd.add_shift(random_seed=1,
    #             shape_method=GAUSS,
    #             shift_method=SIGMOID_new)
    

    profiler = cProfile.Profile()
    profiler.enable()
    pd.add_shift(random_seed=4,
                 shape_method=GAUSS,
                 shift_method=SIGMOID_new)
    
    profiler.disable()
    profiler.dump_stats("profile_output.prof")

if __name__ == "__main__":
    #profiler = cProfile.Profile()
    #profiler.enable()

    main()

    #profiler.disable()
    #profiler.dump_stats("profile_output.prof")
