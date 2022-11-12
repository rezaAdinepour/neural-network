import matplotlib.pyplot as plt
import numpy as np


def display(event):
    datas = []
    if event.button == 1:
        #print("Left Mouse button was clicked!")
        #print(event.xdata, event.ydata)
        plt.scatter(event.xdata, event.ydata, color='red')
        fig.canvas.draw()
        fig.canvas.flush_events()
    elif event.button == 3:
        #print("Right Mouse button was clicked!")
        #print(event.xdata, event.ydata)
        plt.scatter(event.xdata, event.ydata, color='blue')
        fig.canvas.draw()
        fig.canvas.flush_events()
 

fig = plt.figure(1)
plt.xlim(-5, 5)
plt.ylim(-5, 5)
 
fig.canvas.mpl_connect("button_press_event", display)

plt.show()







# class LineBuilder:
#     def __init__(self, line):
#         self.line = line
#         self.xs = list(line.get_xdata())
#         self.ys = list(line.get_ydata())
#         self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

#     def __call__(self, event):
#         print('click', event)
#         if(event.inaxes!=self.line.axes):
#             return
#         self.xs.append(event.xdata)
#         self.ys.append(event.ydata)
#         self.line.set_data(self.xs, self.ys)
#         self.line.figure.canvas.draw()


    
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_title('click to add points')
# line, = ax.plot([], [], linestyle="none", marker="o")
# linebuilder = LineBuilder(line)
# plt.show()