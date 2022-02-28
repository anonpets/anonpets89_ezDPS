import matplotlib.pyplot as plt
from pylab import *


names = ['7', '8', '12', '19', '217', '256', '311', '610', '901']
x = range(len(names))
# Ptime
y = [1757, 1673, 1859, 2480.712227, 3518.886928, 3911.039248, 4286.45422, 4921.262836, 5627.340352]
y1 = [11438, 11490.25967, 11800.80067, 16613.62196, 37325.63103, 48353.51927, 66719.54194, 226419.0579, 481321.1837]
# Vtime
# y = [5158.746033, 5090.426366, 5304.448736, 6120.223106, 6073.956136, 6317.637142, 6538.321439, 6888.2681, 7249.490393]
# y1 = [8919.962192, 8931.187684, 9073.549428, 10752.34445, 11666.7153, 12528.11045, 13736.58245, 20328.87143, 26806.5982]
# Psize
# y = [2706.530562, 2678.284872, 2766.769049, 3196.037866, 2891.960288, 2992.706361, 3083.944808, 3228.624743, 3377.96641]
# y1 = [8919.962192, 8931.187684, 9073.549428, 10752.34445, 11666.7153, 12528.11045, 13736.58245, 20328.87143, 26806.5982]

#plt.plot(x, y, 'ro-')
#plt.plot(x, y1, 'bo-')
plt.xlim(-0.5, 8.5)  # 限定横轴的范围
plt.ylim(-10000, 500000)  # 限定纵轴的范围
plt.plot(x, y, marker='o', mec='r', mfc='w', label=u'Ours')
plt.plot(x, y1, marker='*', ms=10, label=u'Baseline')
plt.legend()  # 让图例生效
plt.xticks(x, names, rotation=45)

ax = plt.gca()
ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(u"Number of classes (parameter s)")  # X轴标签
plt.ylabel("Prover's Time (Partly Estimated) (s)")  # Y轴标签
# plt.title("Comparison in Prover's time")  # 标题

plt.savefig('Ptime_compare')
# plt.savefig('Vtime_compare')
# plt.savefig('Psize_compare')
plt.show()

