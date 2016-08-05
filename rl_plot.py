import numpy as np
from bokeh.plotting import figure, show, output_file


alpha_beta = np.load('all_sess_alpha.npy')
label = np.load('all_sess_label.npy')

N = label.size
x = alpha_beta[:,1]
y = alpha_beta[:,0]

radii = .6 #np.random.random(size=N) * 1.5


palette = ["#053061", "#2166ac", "#4393c3", "#92c5de",
           "#f7f7f7", "#fddbc7", "#f4a582", "#d6604d", "#b2182b", "#67001f"]
colors = [palette[np.int16(i)] for i in label[:,0]-1]
TOOLS="crosshair,pan,wheel_zoom,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select"

p = figure(tools=TOOLS)

p = figure(title = "fit values across animals and sessions for states = odorXwell")

p.scatter(x, y, radius=radii,
          fill_color=colors, fill_alpha=0.6,
          line_color=None)

p.grid.grid_line_color="white"
p.background_fill_color= "#777777"



p.xaxis.axis_label = "beta value"
p.yaxis.axis_label = "alpha value"

output_file("alpha_beta_by_animal_simpleRL.html", title="color_scatter.py example")

show(p)  # open a browser
