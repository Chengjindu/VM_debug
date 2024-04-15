def init():
    global RESCALE, N_, M_, x0_, y0_, dx_, dy_, fps_
    RESCALE = 2

    """
    N_, M_: the row and column of the marker array
    x0_, y0_: the coordinate of upper-left marker (original size before deformation)
    dx_, dy_: the horizontal and vertical interval between adjacent markers (original size before deformation)
    fps_: the desired frame per second, the algorithm will find the optimal solution in 1/fps seconds
    """
    
    N_ = 9
    M_ = 7

    fps_ = 30

    x0_ = 55 / RESCALE
    y0_ = 70 / RESCALE
    dx_ = 90 / RESCALE
    dy_ = 90 / RESCALE

