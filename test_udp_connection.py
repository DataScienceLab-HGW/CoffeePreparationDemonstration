import socket
import sys
import time
from struct import unpack
from random import randint

import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation


simulation = True

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
timex = []
plotline_cb = []

# This function is called periodically from FuncAnimation
def animate_plot(i, timex, line):

    # Add x and y to lists

    timex.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
    line.append(randint(25,40))
    # Limit x and y lists to 20 items
    timex[-20:]
    line[-20:]

    # Draw x and y lists
    ax.clear()
    ax.plot(timex, line)

    # Format plot
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title('Weight of Coffee Box Scale')
    plt.ylabel('Temperature (deg C)')
    #plt.show()

def animate_text(i, timex, line):
    txt = ax.text(0.25, 0.5, "Please add coffee to the French Press", clip_on=False)
    txt.set_clip_on(False) 

# Set up plot to call animate() function periodically
#ani = animation.FuncAnimation(fig, animate_plot, fargs=(timex, plotline_cb), interval=1000)
ani = animation.FuncAnimation(fig, animate_text, fargs=(timex, plotline_cb), interval=1000)

plt.show()


# ----- UDP setup -----
# Create a UDP socket

if not simulation:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Bind the socket to the port
    host, port = '141.53.32.82', 65000
    server_address = (host, port)

    print(f'Starting UDP server on {host} port {port}')
    sock.bind(server_address)


# while True: 
#     # Wait for message
#     if simulation:
#         sum_coffeebox, sum_frenchpress, sum_kettle = 15, 20, 25
#     else:
#         message, address = sock.recvfrom(4096)

#         sum_coffeebox, sum_frenchpress, sum_kettle = unpack('3f', message)

#     print(f'{sum_coffeebox} | {sum_frenchpress} | {sum_kettle}')
#     timex.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
#     plotline_cb.append(sum_coffeebox)
    
#     animate(0, timex, plotline_cb)
#     time.sleep(1)