# Run this script to run the program
from projectile_motion.raw import almost



def main():

    r = 0.1
    v = 50.
    t = 3
    fps = 240

    x = 100
    z = 1
    y = 29
    goal = almost.AABB(x-r, y-r, z-r, x+r, y+r, z+r)

    almost.single_test(goal, v, t, fps, debug=False, graph=True)
    # save.test_aabb_graph(v, r, t, fps, debug=True)
    # save.test_everything(v, r, t, fps, 5, debug=False)



if __name__ == "__main__":
    main()
