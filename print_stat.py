import pstats
from pstats import SortKey

#the code to run the cProfiles is:
#python -m cProfile -o stat.txt imas_gpr1d.py 55564 -ids=interferometer -k=Gibbs_Kernel

p = pstats.Stats('stat_optimizatio.txt')
#p = pstats.Stats('stat.txt')



#p.strip_dirs().sort_stats(-1).print_stats()
#p.sort_stats(SortKey.NAME)
#p.print_stats()

print('if you want to understand what algorithms are taking time')
p.sort_stats(SortKey.CUMULATIVE).print_stats(10)

print('If you were looking to see what functions were looping a lot, and taking a lot of time:')
p.sort_stats(SortKey.TIME).print_stats(10)



#p.sort_stats(SortKey.FILENAME).print_stats('__init__')
#


#p.sort_stats(SortKey.TIME, SortKey.CUMULATIVE).print_stats(.5, 'init')
