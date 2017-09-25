import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches
import io
from matplotlib import colors
import xml.dom.minidom
import numpy as np
import collections
from PIL import Image
from PIL import ImageEnhance
import matplotlib.ticker as plticker

class TopDownViewer():
  def __init__(self):
    self.block_mapping = collections.OrderedDict()
    self.default_block_type_id = -1
    self.ax = plt.axes()
    self.fig = plt.gcf()
    #self.fig.set_size_inches(8,8)
    linecolor = 'black'
    self.linecolor = linecolor
    self.linewidth = 1

    self.ax.spines['top'].set_linewidth(self.linewidth)
    self.ax.spines['top'].set_color(linecolor)
    self.ax.spines['left'].set_linewidth(self.linewidth)
    self.ax.spines['left'].set_color(linecolor)
    self.ax.spines['bottom'].set_linewidth(self.linewidth)
    self.ax.spines['bottom'].set_color(linecolor)
    self.ax.spines['right'].set_linewidth(self.linewidth)
    self.ax.spines['right'].set_color(linecolor)
    self.ax.xaxis.label.set_color(linecolor)
    self.ax.tick_params(axis='x', colors=linecolor)
    self.ax.yaxis.label.set_color(linecolor)
    self.ax.tick_params(axis='y', colors=linecolor)
    self.ax.tick_params(direction='in')
    self.reset()

  def reset(self):
    plt.cla()
    self.player = plt.Circle((0, 0), .25, color='k')
    self.arrow_img = matplotlib.patches.Arrow(x=0, y=0, dx=1, dy=0, width=1, 
        fill = True, color='k', visible = True)
    self.ax = plt.axes()
    self.ax.xaxis.set_animated(True)
    self.ax.yaxis.set_animated(True)
    self.ax.add_artist(self.player)
    self.ax.add_artist(self.arrow_img)

    self.ax.yaxis.set_ticks([])
    self.ax.xaxis.set_ticks([])
    self.ax.yaxis.set_ticklabels([])
    self.ax.xaxis.set_ticklabels([])

  def initialize(self, block_xml, size):
    self.img_size = size
    self.default_block_type_id = 0
    self.images = [plt.imread("MazeBase/images/block_fill.png", format='png'), 
      plt.imread("MazeBase/images/sheep_fill.png", format='png'),
      plt.imread("MazeBase/images/horse_fill.png", format='png'),
      plt.imread("MazeBase/images/pig_fill.png", format='png'),
      plt.imread("MazeBase/images/chest_fill.png", format='png'),
      plt.imread("MazeBase/images/cat_fill.png", format='png'),
      plt.imread("MazeBase/images/monster_fill2.png", format='png'),
      plt.imread("MazeBase/images/ice_fill.png", format='png'),
      ]

  def draw_topology(self, grid):
    self.grid = grid
    for x in range(0, self.grid.shape[0]):
        for y in range(0, self.grid.shape[1]):
            if self.grid[x, y] == 1:
                self.grid[x, y] = 0
            else:
                self.grid[x, y] = 1

  def clear(self):
    plt.cla()
    self.player = plt.Circle((0, 0), .25, color='k')
    self.arrow_img = matplotlib.patches.Arrow(x=0, y=0, dx=1, dy=0, width=1, 
        fill = True, color='k', visible = True)
    self.ax = plt.axes()
    self.ax.xaxis.set_animated(True)
    self.ax.yaxis.set_animated(True)
    self.ax.add_artist(self.player)
    self.ax.add_artist(self.arrow_img)

    self.ax.yaxis.set_ticks([])
    self.ax.xaxis.set_ticks([])
    self.ax.yaxis.set_ticklabels([])
    self.ax.xaxis.set_ticklabels([])

  def draw_objects(self, objects, changed):
    if changed:
      self.clear()
      self.arrow_img.set_visible(False)
      for x in range(0, self.grid.shape[0]):
          for y in range(0, self.grid.shape[1]):
              if objects[x, y] != 0:
                  plt.imshow(self.images[int(objects[x, y])], extent=[y,y+1,x,x+1], 
                      origin='lower', interpolation='nearest')
              elif self.grid[x, y] == 1:
                  plt.imshow(self.images[0], extent=[y,y+1,x,x+1], origin='lower', interpolation='nearest')

      self.ax.set_xlim([0, self.grid.shape[1]])
      self.ax.set_ylim([self.grid.shape[0], 0])
      loc = plticker.MultipleLocator(base=1)
      self.ax.xaxis.set_major_locator(loc)
      self.ax.yaxis.set_major_locator(loc)
      self.ax.grid(which='major', axis='both', linestyle='-', 
          linewidth=self.linewidth, color=self.linecolor)


  def update_frame(self, pos_x, pos_y, facing):
    img_size = int(self.img_size)
    self.player.center = (pos_x + 0.5, pos_y + 0.5)
    self.arrow_img.remove()
    self.arrow_img.set_visible(False)

    if facing == 1:
        Dx = 0
        Dy = 1
    elif facing == 2:
        Dx = -1
        Dy = 0
    elif facing == 3:
        Dx = 0
        Dy = -1
    elif facing == 4:
        Dx = 1
        Dy = 0
    else:
        assert False

    self.arrow_img = matplotlib.patches.Arrow(x = pos_x + 0.5, 
        y = pos_y + 0.5, dx = Dx, dy = Dy, width = 1.0, fill = True, 
        color='k', visible = True)
    self.ax.add_artist(self.arrow_img)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent="True", pad_inches=0)
    buf.seek(0)
    top_down_view = Image.open(buf)
    top_down_view = top_down_view.convert(mode="RGB")
    top_down_view = top_down_view.resize(( \
        img_size, int(img_size * float(self.grid.shape[0])/float(self.grid.shape[1]))),
        Image.ANTIALIAS)

    return top_down_view

def create_viewer():
  return TopDownViewer()
