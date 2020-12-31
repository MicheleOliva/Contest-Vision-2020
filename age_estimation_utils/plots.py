import matplotlib.pyplot as plt
import numpy as np


def pie_plot(labels, sizes, colors, explode, path):
  
  plt.pie(sizes, explode=explode, labels=labels, colors=colors,
  autopct='%1.1f%%', shadow=True, startangle=140)

  plt.axis('equal')
  plt.savefig(path)
  plt.show()


def histogram(labels, flist, slist, path, titles):

  x = np.arange(len(labels))  # the label locations
  width = 0.35  # the width of the bars

  fig, ax = plt.subplots()
  rects1 = ax.bar(x - width/2, flist, width, label=titles[0])
  rects2 = ax.bar(x + width/2, slist, width, label=titles[1])

  # Add some text for labels, title and custom x-axis tick labels, etc.
  ax.set_ylabel(titles[2])
  ax.set_title(titles[3])
  ax.set_xticks(x)
  ax.set_xticklabels(labels)
  ax.legend()

  def autolabel(rects):
    
      for rect in rects:
          height = rect.get_height()
          ax.annotate('{}'.format(height),
                      xy=(rect.get_x() + rect.get_width() / 2, height),
                      xytext=(0, 3),  # 3 points vertical offset
                      textcoords="offset points",
                      ha='center', va='bottom')

  autolabel(rects1)
  autolabel(rects2)

  fig.tight_layout()
  plt.savefig(path)
  plt.show()


