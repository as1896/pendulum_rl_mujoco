import time
import numpy as np
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("model/pendulum.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.type         = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.distance     = 4.0
    viewer.cam.azimuth      = 90.0
    viewer.cam.elevation    = 0.0
    viewer.cam.lookat[:] = np.array([0.0, 0.0, 1.0], dtype=float)

    while viewer.is_running():

        mujoco.mj_step(model, data)
        time.sleep(model.opt.timestep)
        viewer.sync()