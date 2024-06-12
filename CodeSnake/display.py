import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()

def update(frame):
    ax.clear()
    ax.imshow(frame)
    ax.axis('off')

def display(I, name):
    ani = animation.FuncAnimation(fig, update, frames=I, repeat=False)
    ani.save("./CodeSnake/Video/" + name + ".png",writer='ffmpeg')
    plt.show()
