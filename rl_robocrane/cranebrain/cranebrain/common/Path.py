import numpy as np
import mujoco
import time
import pinocchio as pin
import arcpy.core as ac
from scipy.spatial.transform import Rotation
import os
import sys
pinutil_path = os.path.abspath(os.path.join("/home/ubuntu/", "python"))
sys.path.append(pinutil_path)
import create_robocrane_env as cre
import cranebrain.cranebrain.utils.pinutil as pu

from sspp import _sspp as sp
from ..common.SteadyState import SteadyState
from ..utils.util import load
from ..utils.mujoco_util import addTriad, visualizePath


class Path: 

    def __init__(self, 
                pin_model, 
                pin_data, 
                tool_frame_id, 
                xml_path, 
                sigma = 0.15, 
                limits = np.array([170, 120, 170, 120, 170, 120, 175, 25, 25]) * np.pi / 180,
                n_ctrl_pts=7, 
                sample_count=200, 
                check_points=100):

        self.pin_model = pin_model
        self.pin_data = pin_data
        self.tool_frame_id = tool_frame_id

        self.ss_obj = SteadyState(pin_model, pin_data, tool_frame_id)
        self.planner = sp.SamplingPathPlanner9(xml_path)
        self.spline_ls = []
        
        self.sigma = sigma
        self.limits = limits
        self.n_ctrl_pts = n_ctrl_pts        # nr. of spline control points
        self.sample_count = sample_count    # nr of trajectory samples
        self.check_points = check_points    # eval points on each trajectory for collission checking


    def get_site_homtrans(self, mj_model, mj_data, site_name):
        site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        pos = mj_data.site_xpos[site_id]
        rotmat = mj_data.site_xmat[site_id]
        hom_trans = np.eye(4)
        hom_trans[0:3,3] = pos
        hom_trans[0:3,0:3] = np.array(rotmat).reshape((3,3))
        return hom_trans


    def inverse_kinematics(self, H_des, q_init):
        q_ik, success = self.ss_obj.inverse_kinematics(H_des, q_init)
        if not success:
            raise Exception("IK failed")
        return q_ik


    def forward_kinematics(self, q):
        pin.forwardKinematics(self.pin_model, self.pin_data, q.T)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        oMf_tool = self.pin_data.oMf[self.tool_frame_id]
        return oMf_tool.homogeneous


    def plan(self, q_start, q_end):


        if len(q_start) != 9 or len(q_end) != 9:
            raise Exception("Invalid joint configuration")

        success, succ_paths = self.planner.plan(q_start, 
                                    q_end, 
                                    self.sigma, 
                                    self.limits, 
                                    sample_count = self.sample_count, 
                                    check_points = self.check_points, 
                                    init_points = self.n_ctrl_pts)

        self.spline_ls = succ_paths

        #print("# of successfull paths: ", len(succ_paths))
        # if not success:
        #     raise Exception("Path planning failed")
        return success

    def get_ctrl_pts(self):
        return self.planner.get_ctrl_pts()


    def evaluate(self, u, spline=None):
        if spline is None:
            return self.planner.evaluate(u)
        return self.planner.evaluate(spline, u)


    # def find_steadystate(self, q_a):
    #     q_init = np.concatenate([q_a, [0,0]])
    #     q_result = self.ss_obj.find_steady_state(q_init)
    #     return q_result


    def get_path(self, n_path_pts):
        u_vec = np.linspace(0,1,n_path_pts)
        H_ls = []

        for u in u_vec:
            q = self.evaluate(u)
            H = self.forward_kinematics(q)
            H_ls.append(H)

        return H_ls


    def get_all_paths(self, n_path_pts):
        u_vec = np.linspace(0,1,n_path_pts)
        path_ls = []

        for spline in self.spline_ls:
            H_ls = []

            print("shape spline::ctrls: ", spline.ctrls().shape)
            for u in u_vec:
                q = self.evaluate(u, spline)
                H = self.forward_kinematics(q)
                H_ls.append(H)

            path_ls.append(H_ls)

        return path_ls



   
def main():
    mj_model, mj_data, xml_path, pin_model, pin_data, env, tool_frame_id = load()
    path = Path(pin_model, pin_data, tool_frame_id, xml_path)
    # mj_model, mj_data, env = cre.create_robocrane_mujoco_models()

    # get position of wall/site_left wall and wall/site_right_wall
    H_left = path.get_site_homtrans(mj_model, mj_data, "wall/site_left_wall")
    H_right = path.get_site_homtrans(mj_model, mj_data, "wall/site_right_wall")

    # print("tool frame name: ", mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, tool_frame_id))
    # exit()
    

    q_init = [0, 0, 0, -1.5708,0,1.5708,0,0,0 ]
    q_sol = []
    q_sol.append(path.inverse_kinematics(H_left, q_init))
    q_sol.append(path.inverse_kinematics(H_right, q_init))

    path.plan(q_sol[0], q_sol[1]) # plan only actuated DoFs

    # H_ls = path.get_path(25)
    ctrls = path.get_ctrl_pts()
    print("shape of ctrl pots: ", ctrls.shape)
    path_ls = path.get_all_paths(25)
    

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:

        # config viewer
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = False
        viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE # mjLABEL_BODY
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE # mjFRAME_BODY

        ind = 0

        for H_ls in path_ls:
            path_ngeoms = len(H_ls) * [1]
            update = False
            visualizePath(viewer.user_scn, H_ls, path_ngeoms, update)


        N = 50
        for i in range(N):
            u = i / N
            q = path.evaluate(u)
            mj_data.qpos[:9] = q
            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()

            if not viewer.is_running():
                break

            time.sleep(0.1)

        input("Press Enter to continue...")

        viewer.close()



if __name__ == "__main__":
    main()

