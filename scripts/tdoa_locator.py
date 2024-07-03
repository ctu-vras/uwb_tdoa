#!/usr/bin/env python
# 3 TDoA anchors -> 1 anchor

import rospy
import copy

from uwb_tdoa.msg import TDoAAnchor, TDoAMeas
from dwm1001_ros.msg import UWBMeas
from xplraoa_ros.msg import Angles
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from std_msgs.msg import Bool, String

import numpy as np
from numpy.polynomial import Polynomial
import tf2_ros
import tf2_py as tf2
import ros_numpy
import genpy
import threading
from scipy.optimize import least_squares
from scipy.signal import lfilter, lfilter_zi, butter
from collections import deque, defaultdict
from scipy.optimize import least_squares
from scipy.signal import lfilter, lfilter_zi, butter
from collections import deque

C = 299792458
CUTOFF = 1.5
RANGE = 4.0
CALIB_LEN = 50
R_const = 100
Q_const = 1


def get_angle(Va, Vb, Vn):
    """returns oriented angle of rotation from Va to Vb"""
    # https://stackoverflow.com/a/33920320
    return float(
        np.arctan2(np.matmul(np.cross(Va, Vb, axis=0).T, Vn), np.matmul(Va.T, Vb))
    )


class Kalman:
    def __init__(self, x0, P0, dt):
        # x = [x, y, z, v_x, v_y, a_x, a_y]
        # measurement = [x, y]
        # input = [v_x, v_y]
        self.x = x0
        self.x_cov = P0
        self.dt = dt
        self.A = np.array(
            [
                [1, 0, dt, 0, dt**2 / 2, 0],
                [0, 1, 0, dt, 0, dt**2 / 2],
                [0, 0, 1, 0, dt, 0],
                [0, 0, 0, 1, 0, dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
        self.B = np.array(
            [
                [0, 0],
                [0, 0],
                [1, 0],
                [0, 1],
                [0, 0],
                [0, 0],
            ]
        )
        self.Q = Q_const * np.eye(6)
        self.R = R_const * np.eye(2)
        self.H = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
            ]
        )

    def set_initial(self, x0, P0):
        self.x = x0
        self.x_cov = P0

    def predict(self, input):
        self.x = np.matmul(self.A, self.x) + np.matmul(self.B, input)
        self.x_cov = np.matmul(np.matmul(self.A, self.x_cov), self.A.T) + self.Q

        return self.x, self.x_cov

    def correct(self, measurement):
        K = np.matmul(
            np.matmul(self.x_cov, self.H.T),
            np.linalg.inv(np.matmul(np.matmul(self.H, self.x_cov), self.H.T) + self.R),
        )
        self.x = self.x + np.matmul(K, measurement - np.matmul(self.H, self.x))
        self.x_cov = np.matmul(np.eye(7) - np.matmul(K, self.H), self.x_cov)

        return self.x, self.x_cov


class Locator:
    def __init__(self) -> None:
        rospy.init_node("tdoa_locator")
        self.msg_lock = threading.Lock()
        self.stamps = {}
        self.vars = {}
        self.frame_sn = {}

        # tf2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.sleep(0.2)

        # TDoA config
        self.anchors_address = rospy.get_param("uwb/tdoa/anchors/address")
        self.coords = rospy.get_param("uwb/tdoa/anchors/position")
        self.num_anchors = rospy.get_param("uwb/tdoa/num_anchors")
        self.positions = {}
        self.tof = {}
        self.lp_filter = butter(3, CUTOFF, fs=10.0)
        self.zi = {}
        self.calib_queue = defaultdict(dict)
        self.calib = {}
        for i in range(len(self.anchors_address)):
            address = self.anchors_address[i]
            self.positions[address] = np.array(
                [
                    [self.coords[3 * i]],
                    [self.coords[3 * i + 1]],
                    [self.coords[3 * i + 2]],
                ]
            )
            self.zi[address] = lfilter_zi(self.lp_filter[0], self.lp_filter[1])
            self.calib[address] = Polynomial([0.0, 1.0])

        # TWR config
        self.target_anchor = rospy.get_param("uwb/tdoa/target/anchor")
        self.target_tags = rospy.get_param("uwb/tdoa/target/tags")
        self.target_ids = rospy.get_param("uwb/twr/ids")
        self.target_coords = rospy.get_param("uwb/twr/positions")
        self.target_positions = {}
        self.subscribers = {}
        for i in range(len(self.target_ids)):
            id_l = self.target_ids[i].lower()
            self.target_positions[id_l] = np.array(
                [
                    [self.target_coords[3 * i]],
                    [self.target_coords[3 * i + 1]],
                    [self.target_coords[3 * i + 2]],
                ]
            )

        # rviz visualization
        self.marker = Marker()
        self.marker.header.frame_id = "locator"
        self.marker.type = self.marker.ARROW
        self.marker.action = self.marker.ADD
        self.marker.scale.x = 0.1
        self.marker.scale.y = 0.2
        self.marker.scale.z = 0.3
        self.marker.color.a = 1.0
        self.marker.color.r = 1.0
        self.marker.color.g = 1.0
        self.marker.color.b = 0.0
        self.marker.pose.position.x = 0.0
        self.marker.pose.position.y = 0.0
        self.marker.pose.position.z = 0.0

        # position estimation
        self.last_estimate = None
        self.tim2 = rospy.Timer(rospy.Duration(0.1), self.get_estimate)

        # estimate filtration
        self.filter = Kalman(np.array([[0], [0], [0], [0], [0], [0]]), np.eye(6), 0.1)
        self.initialised = False

        # angle measurements
        self.angle_zi = lfilter_zi(self.lp_filter[0], self.lp_filter[1])
        self.angle_pub = rospy.Publisher("uwb/tdoa/angle", Angles, queue_size=1)

        self.anchors = list(self.positions.keys())

        self.marker_pub = rospy.Publisher("heading", Marker, queue_size=1)

        self.subs1 = rospy.Subscriber(
            "uwb/tdoa/measurements", TDoAMeas, self.tdoa_cb, queue_size=1
        )

        self.tim = rospy.Timer(rospy.Duration(1.0), self.auto_calibrate)
        self.tim2 = rospy.Timer(rospy.Duration(0.1), self.estimate_angle)
        print("setup done")

        rospy.on_shutdown(self.shutdown)

    def shutdown(self):
        self.tim.shutdown()
        rospy.sleep(0.2)

    def get_transform(self, tf_from, tf_to, out="matrix", time=None, dur=0.1):
        """returns the latest transformation between the given frames
        the result of multiplying point in frame tf_to by the output matrix is in the frame tf_from

        :param tf_from: find transform from this frame
        :param tf_to: find transform to this frame
        :param out: the return type
                    - 'matrix' - returns numpy array with the tf matrix
                    - 'tf' - returns TransformStamped
        :param time: the desired timestamp of the transform (ROS Time)
        :param dur: the timeout of the lookup (float)
        :return: as selected by out parameter or None in case of tf2 exception
                    - only ConnectivityException is logged
        """
        if time is None:
            tf_time = rospy.Time(0)
        else:
            if not isinstance(time, rospy.Time) and not isinstance(time, genpy.Time):
                raise TypeError("parameter time has to be ROS Time")
            tf_time = time

        try:
            t = self.tf_buffer.lookup_transform(
                tf_from, tf_to, tf_time, rospy.Duration(dur)
            )
        except (tf2.LookupException, tf2.ExtrapolationException):
            return None
        except tf2.ConnectivityException as ex:
            rospy.logerr(ex)
            return None

        # return the selected type
        if out == "matrix":
            return ros_numpy.numpify(t.transform)
        elif out == "tf":
            return t
        else:
            raise ValueError("argument out should be 'matrix' or 'tf'")

    def get_estimate(self, _):
        t = self.get_transform("locator", "position")
        if t is not None:
            est = np.array([t[0, 3], t[1, 3]])
            est /= np.linalg.norm(est)
            self.last_estimate = est

    def tdoa_cb(self, msg: TDoAMeas):
        self.msg_lock.acquire()
        tag = False
        stamps = {}
        for m in msg.measurements:
            if m.target == self.target_anchor:
                if m.meas_valid:
                    self.stamps[m.address] = m.stamp
                    self.vars[m.address] = m.variance
                    self.frame_sn[m.address] = m.frame_sn
            else:
                if m.meas_valid:
                    tag = True
                    stamps[m.address] = m.stamp
        self.msg_lock.release()
        if tag:
            for i in range(len(self.anchors)):
                id1 = self.anchors[i]
                id2 = self.anchors[(i + 1) % len(self.anchors)]
                d1 = np.linalg.norm(
                    self.positions[id1] - self.target_positions[m.target]
                )
                d2 = np.linalg.norm(
                    self.positions[id2] - self.target_positions[m.target]
                )
                expected_diff = d1 - d2
                diff = C * (stamps[id1] - stamps[id2])
                if m.target not in self.calib_queue[id1].keys():
                    self.calib_queue[id1][m.target] = deque()
                self.calib_queue[id1][m.target].append((expected_diff, diff))

    def auto_calibrate(self, _):
        for a in self.anchors:
            l = []
            for t in self.target_tags:
                if t not in self.calib_queue[a].keys():
                    continue
                while len(self.calib_queue[a][t]) >= CALIB_LEN:
                    self.calib_queue[a][t].popleft()
                if len(self.calib_queue[a][t]) != 0:
                    mean = np.mean(self.calib_queue[a][t], axis=0)
                    l += [(mean[1], mean[0])]
            l = np.array(l)
            if len(l) == 0:
                return
            elif len(l) == 1:
                c = np.mean(l[:, 0] - l[:, 1])
                self.calib[a] = Polynomial([c, 1.0])
            else:
                p = Polynomial.fit(l[:, 1], l[:, 0], 1)
                p = p.convert()
                self.calib[a] = p
            print(self.calib[a])
        print()

    def intersectionPoint(self, guess, p_tdoa, d_tdoa, w):
        x_t1 = []
        y_t1 = []
        x_t2 = []
        y_t2 = []
        for i in range(len(p_tdoa)):
            x_t1 += [[p_tdoa[i][0][0, 0]]]
            y_t1 += [[p_tdoa[i][0][1, 0]]]
            x_t2 += [[p_tdoa[i][1][0, 0]]]
            y_t2 += [[p_tdoa[i][1][1, 0]]]
        x_t1_tdoa = np.array(x_t1)
        y_t1_tdoa = np.array(y_t1)
        x_t2_tdoa = np.array(x_t2)
        y_t2_tdoa = np.array(y_t2)
        d_tdoa = np.array(d_tdoa)

        last = self.last_estimate

        def eq(g):
            x, y = g

            f_tdoa = (
                np.sqrt((x - x_t1_tdoa) ** 2 + (y - y_t1_tdoa) ** 2)
                - np.sqrt((x - x_t2_tdoa) ** 2 + (y - y_t2_tdoa) ** 2)
                - d_tdoa
            )
            f_tdoa = w * f_tdoa
            f = np.vstack((f_tdoa, 10 * (np.linalg.norm(g) - 1)))
            if last is not None:
                f = np.vstack((f, 0.5 * np.linalg.norm(g - last)))
            return f.flatten().tolist()

        ans = least_squares(eq, guess, loss="soft_l1", verbose=0)

        if ans.success:
            return ans.x
        else:
            return None

    def estimate_angle(self, _):
        if len(self.stamps.keys()) != self.num_anchors:
            rospy.logwarn("No data")
            return

        self.msg_lock.acquire()
        stamps = copy.deepcopy(self.stamps)
        self.msg_lock.release()

        d = []
        pos = []
        w = []
        for i in range(len(self.anchors)):
            id1 = self.anchors[i]
            id2 = self.anchors[(i + 1) % len(self.anchors)]
            a1 = self.positions[id1]
            a2 = self.positions[id2]
            d_a2a = np.linalg.norm(a1 - a2)
            diff = self.calib[id1](C * (stamps[id1] - stamps[id2]))

            if diff > d_a2a:
                diff = d_a2a

            w += [[1 - np.abs(diff) / d_a2a]]

            var = C * (self.vars[id1] + self.vars[id2])
            diff_avg, self.zi[id1] = lfilter(
                self.lp_filter[0], self.lp_filter[1], np.array([diff]), zi=self.zi[id1]
            )
            diff_avg = float(diff_avg)

            d += [[diff_avg]]
            pos += [[a1, a2]]

        # solve NLS
        if self.last_estimate is None:
            guess = np.array([1.0, 0.0])
        else:
            guess = self.last_estimate
        est = self.intersectionPoint(guess, pos, d, w)
        if est is None:
            rospy.logwarn("No solution")
        else:
            Va = np.array([[1.0], [0.0], [0.0]])
            Vb = np.array([[est[0]], [est[1]], [0.0]])
            Vn = np.array([[0.0], [0.0], [1.0]])
            angle = get_angle(Va, Vb, Vn)
            msg = Angles(angle, 0.0, 0.0)
            self.angle_pub.publish(msg)
            self.marker.header.stamp = rospy.Time.now()
            self.marker.points = [
                Point(0, 0, 0),
                Point(2 * np.cos(angle), 2 * np.sin(angle), 0),
            ]
            self.marker_pub.publish(self.marker)


if __name__ == "__main__":
    l = Locator()
    rospy.spin()
