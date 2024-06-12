import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()

def update(frame):
    ax.clear()
    ax.imshow(frame)
    ax.axis('off')

def display(I):
    ani = animation.FuncAnimation(fig, update, frames=I, repeat=False)
    ani.save("./CodeSnake/Video/interpolation.mp4",writer='ffmpeg')
    plt.show()
