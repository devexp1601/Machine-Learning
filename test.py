import turtle

# Setup turtle
t = turtle.Turtle()
t.speed(3)  # speed of drawing
t.color("gold")

# Draw a 5-pointed star
for _ in range(5):
    t.forward(100)  # length of each star side
    t.right(144)    # angle for star

# Finish
turtle.done()